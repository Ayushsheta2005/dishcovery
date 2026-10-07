"""Reading and writing qrels and runs.

File formats (JSON Lines, one object per line; ``.gz`` paths are transparently
compressed):

* **qrels** — ``{"qid", "doc_id", "grade", "source", "annotator"?}``.
  ``source`` is mandatory and is one of ``hand`` (human judgments; the only
  labels NDCG is ever reported on), ``weak`` (heuristic weak supervision, see
  ``dishcovery.labels.weak``) or ``clicks`` (simulated-click IPW labels, see
  ``dishcovery.labels.clicks``). Loaders filter by source so the three never
  mix by accident.
* **runs** — ``{"qid", "system", "doc_ids": [...], "scores": [...]}``; one file
  may hold several systems.

TREC exports (``qid Q0 doc rank score tag`` / ``qid 0 doc grade``) are provided
for cross-checking with ``trec_eval``.
"""
from __future__ import annotations

import gzip
import io as _io
import json
import os
import warnings
from typing import IO, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

try:  # orjson is ~5x faster on large label/run files; fall back to stdlib json.
    import orjson as _orjson
except ImportError:  # pragma: no cover
    _orjson = None

from .metrics import Qrels, Run

PathLike = Union[str, "os.PathLike[str]"]

LABEL_SOURCES = ("hand", "weak", "clicks")
MAX_GRADE = 3


# ---------------------------------------------------------------------------
# JSONL primitives
# ---------------------------------------------------------------------------

def _open(path: PathLike, mode: str) -> IO:
    p = str(path)
    if "w" in mode or "a" in mode:
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)
    if p.endswith(".gz"):
        return _io.TextIOWrapper(gzip.open(p, mode.replace("t", "") + "b"), encoding="utf-8")
    return open(p, mode, encoding="utf-8")


def dumps(obj) -> str:
    """Compact single-line JSON."""
    if _orjson is not None:
        return _orjson.dumps(obj).decode("utf-8")
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def loads(line: Union[str, bytes]):
    if _orjson is not None:
        return _orjson.loads(line)
    return json.loads(line)


def read_jsonl(path: PathLike) -> Iterator[dict]:
    """Yield one object per non-empty line."""
    with _open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield loads(line)


def write_jsonl(path: PathLike, records: Iterable[dict], append: bool = False) -> int:
    """Write records, one per line. Returns the number written."""
    n = 0
    with _open(path, "at" if append else "wt") as f:
        for r in records:
            f.write(dumps(r))
            f.write("\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Qrels
# ---------------------------------------------------------------------------

def make_qrel(qid: str, doc_id: str, grade: int, source: str,
              annotator: Optional[str] = None) -> dict:
    """Build a validated qrel record."""
    if source not in LABEL_SOURCES:
        raise ValueError(f"source must be one of {LABEL_SOURCES}, got {source!r}")
    g = int(grade)
    if not 0 <= g <= MAX_GRADE:
        raise ValueError(f"grade must be in 0..{MAX_GRADE}, got {grade!r}")
    rec = {"qid": str(qid), "doc_id": str(doc_id), "grade": g, "source": source}
    if annotator:
        rec["annotator"] = annotator
    return rec


def qrels_to_records(qrels: Mapping[str, Mapping[str, int]], source: str,
                     annotator: Optional[str] = None) -> List[dict]:
    """Flatten ``qid -> {doc: grade}`` into records tagged with ``source``."""
    return [make_qrel(q, d, g, source, annotator)
            for q in sorted(qrels) for d, g in sorted(qrels[q].items())]


def qrels_from_records(records: Iterable[dict], sources: Optional[Sequence[str]] = None,
                       aggregate: str = "last") -> Qrels:
    """Collapse qrel records into ``qid -> {doc_id: grade}``.

    Args:
        records: qrel dicts (order matters for ``aggregate="last"``).
        sources: keep only these sources (``None`` = all — avoid for reporting).
        aggregate: how to combine several judgments of one pair:
            ``last`` (later records override — re-labelling / "back" in the
            labelling tool), ``max``, ``min`` or ``median`` (lower median, for
            multiple annotators).
    """
    if aggregate not in ("last", "max", "min", "median"):
        raise ValueError(f"unknown aggregate {aggregate!r}")
    keep = set(sources) if sources is not None else None
    multi: Dict[str, Dict[str, List[int]]] = {}
    for r in records:
        if keep is not None and r.get("source") not in keep:
            continue
        multi.setdefault(str(r["qid"]), {}).setdefault(str(r["doc_id"]), []).append(int(r["grade"]))
    out: Qrels = {}
    for q, docs in multi.items():
        row: Dict[str, int] = {}
        for d, gs in docs.items():
            if aggregate == "last":
                row[d] = gs[-1]
            elif aggregate == "max":
                row[d] = max(gs)
            elif aggregate == "min":
                row[d] = min(gs)
            else:
                row[d] = sorted(gs)[(len(gs) - 1) // 2]
        out[q] = row
    return out


def read_qrels_records(path: PathLike, sources: Optional[Sequence[str]] = None) -> List[dict]:
    """Raw qrel records, optionally filtered by source."""
    keep = set(sources) if sources is not None else None
    return [r for r in read_jsonl(path) if keep is None or r.get("source") in keep]


def load_qrels(path: PathLike, sources: Optional[Sequence[str]] = ("hand",),
               aggregate: str = "last") -> Qrels:
    """Load qrels as ``qid -> {doc_id: grade}``, keeping only ``sources``.

    Defaults to hand labels — the only ones evaluation numbers are reported
    on. Pass ``sources=("weak",)`` or ``("clicks",)`` explicitly for training
    labels. Warns if the filter removes every record of a non-empty file.
    """
    records = list(read_jsonl(path))
    qrels = qrels_from_records(records, sources, aggregate)
    if records and not qrels:
        found = sorted({str(r.get("source")) for r in records})
        warnings.warn(f"{path}: no qrels with source in {list(sources or [])}; file has {found}")
    return qrels


def label_sources(path: PathLike) -> Dict[str, int]:
    """Count records per source in a qrels file."""
    counts: Dict[str, int] = {}
    for r in read_jsonl(path):
        s = str(r.get("source"))
        counts[s] = counts.get(s, 0) + 1
    return counts


def write_qrels(path: PathLike, records: Iterable[dict], append: bool = False) -> int:
    """Write qrel records (validated) to JSONL."""
    checked = (make_qrel(r["qid"], r["doc_id"], r["grade"], r["source"], r.get("annotator"))
               for r in records)
    return write_jsonl(path, checked, append=append)


def append_qrel(path: PathLike, record: dict) -> None:
    """Append one qrel and flush to disk immediately (resume-safe labelling)."""
    rec = make_qrel(record["qid"], record["doc_id"], record["grade"], record["source"],
                    record.get("annotator"))
    with _open(path, "at") as f:
        f.write(dumps(rec))
        f.write("\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, ValueError, _io.UnsupportedOperation):  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def run_records(run: Mapping[str, Sequence[str]], system: str,
                scores: Optional[Mapping[str, Sequence[float]]] = None) -> List[dict]:
    """Convert ``qid -> doc ids`` (+ optional ``qid -> scores``) to run records."""
    out = []
    for qid in sorted(run):
        docs = list(run[qid])
        if scores is not None and qid in scores:
            sc = [float(s) for s in scores[qid]]
            if len(sc) != len(docs):
                raise ValueError(f"{system}/{qid}: {len(docs)} docs but {len(sc)} scores")
        else:  # synthesise strictly decreasing scores so rank order is preserved
            sc = [float(len(docs) - i) for i in range(len(docs))]
        out.append({"qid": str(qid), "system": system, "doc_ids": docs, "scores": sc})
    return out


def write_run(path: PathLike, run: Mapping[str, Sequence[str]], system: str,
              scores: Optional[Mapping[str, Sequence[float]]] = None, append: bool = False) -> int:
    """Write one system's run as JSONL (``append=True`` to add a system to a file)."""
    return write_jsonl(path, run_records(run, system, scores), append=append)


def read_run_records(path: PathLike, systems: Optional[Sequence[str]] = None) -> List[dict]:
    keep = set(systems) if systems is not None else None
    return [r for r in read_jsonl(path) if keep is None or r.get("system") in keep]


def load_runs(path: PathLike, systems: Optional[Sequence[str]] = None) -> Dict[str, Run]:
    """``system -> qid -> doc ids`` from a (multi-system) run file."""
    out: Dict[str, Run] = {}
    for r in read_run_records(path, systems):
        out.setdefault(str(r["system"]), {})[str(r["qid"])] = [str(d) for d in r["doc_ids"]]
    return out


def load_run(path: PathLike, system: Optional[str] = None) -> Run:
    """``qid -> doc ids`` for one system (required if the file holds several)."""
    runs = load_runs(path, [system] if system else None)
    if system is None:
        if len(runs) > 1:
            raise ValueError(f"{path} holds systems {sorted(runs)}; pass system=")
        return next(iter(runs.values()), {})
    return runs.get(system, {})


def load_run_scores(path: PathLike, system: str) -> Dict[str, List[float]]:
    """``qid -> scores`` aligned with ``load_run(path, system)``."""
    return {str(r["qid"]): [float(s) for s in r.get("scores", [])]
            for r in read_run_records(path, [system])}


# ---------------------------------------------------------------------------
# TREC formats
# ---------------------------------------------------------------------------

def export_trec_run(path: PathLike, run: Mapping[str, Sequence[str]], system: str,
                    scores: Optional[Mapping[str, Sequence[float]]] = None) -> int:
    """Write ``qid Q0 doc_id rank score system`` lines (rank is 1-based)."""
    n = 0
    with _open(path, "wt") as f:
        for rec in run_records(run, system, scores):
            for i, (d, s) in enumerate(zip(rec["doc_ids"], rec["scores"])):
                f.write(f"{rec['qid']} Q0 {d} {i + 1} {s:.6f} {system}\n")
                n += 1
    return n


def export_trec_qrels(path: PathLike, qrels: Mapping[str, Mapping[str, int]]) -> int:
    """Write ``qid 0 doc_id grade`` lines."""
    n = 0
    with _open(path, "wt") as f:
        for q in sorted(qrels):
            for d, g in sorted(qrels[q].items()):
                f.write(f"{q} 0 {d} {int(g)}\n")
                n += 1
    return n


def read_trec_run(path: PathLike) -> Dict[str, Run]:
    """``system -> qid -> doc ids`` sorted by score (desc), ties by rank."""
    rows: Dict[str, Dict[str, List[tuple]]] = {}
    with _open(path, "rt") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, doc, rank, score, tag = parts[:6]
            rows.setdefault(tag, {}).setdefault(qid, []).append((-float(score), int(rank), doc))
    return {tag: {q: [d for _, _, d in sorted(v)] for q, v in by_q.items()}
            for tag, by_q in rows.items()}


def read_trec_qrels(path: PathLike) -> Qrels:
    out: Qrels = {}
    with _open(path, "rt") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                out.setdefault(parts[0], {})[parts[2]] = int(parts[3])
    return out
