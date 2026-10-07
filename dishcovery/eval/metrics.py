"""Offline ranking metrics over graded relevance judgments.

Conventions (fixed here so every table in the project means the same thing):

* **Grades** are integers 0..3 (see ``data/eval/LABELING_GUIDELINES.md``).
  Negative grades, if they ever appear, are treated as 0 gain but count as judged.
* **Gain** is ``2**grade - 1`` and the **discount** at 0-based rank ``i`` is
  ``1 / log2(i + 2)`` (Burges et al. / trec_eval ``ndcg_cut``).
* **Ideal DCG** is computed from *all* judged grades of the query, not only the
  ones the run retrieved, so a run is penalised for missing judged-relevant docs.
* **Unjudged** documents in a run get grade 0 (they are *not* removed —
  "condensed list" evaluation would hide pool bias; ``judged@k`` exposes it
  instead).
* **Relevant** (for MRR / recall / precision) means ``grade >= rel_threshold``
  (default 2: "good substitute" or better).
* A per-query metric is **undefined** (``None``) when the query has no judged
  document that could ever score: no positive grade for NDCG, nothing at or
  above the threshold for MRR / recall / precision. Following TREC practice
  such queries are *skipped and counted* (``EvalResult.skipped``) rather than
  averaged in as zeros. Skipping depends only on the qrels, so every system is
  averaged over the same query set and paired tests stay aligned.
* Duplicate doc ids in a ranking are collapsed to their first occurrence.
* Judged queries that are absent from the run are scored as an empty ranking
  (NDCG = 0 etc.) and listed in ``EvalResult.missing_in_run``; run queries
  without any judgment are ignored and listed in ``EvalResult.unjudged_queries``.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Qrels = Dict[str, Dict[str, int]]
Run = Dict[str, List[str]]

DEFAULT_REL_THRESHOLD = 2
DEFAULT_METRICS = ("ndcg@10", "mrr", "recall@100", "judged@10")

# Canonical metric names and accepted aliases.
_ALIASES = {
    "ndcg": "ndcg", "ndcg_cut": "ndcg",
    "dcg": "dcg",
    "mrr": "mrr", "rr": "mrr", "recip_rank": "mrr",
    "recall": "recall", "r": "recall",
    "precision": "precision", "p": "precision", "prec": "precision",
    "judged": "judged",
}
_METRIC_RE = re.compile(r"^([a-z_]+)(?:@(\d+))?$")


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------

def _gain(grade: float) -> float:
    return 2.0 ** max(grade, 0) - 1.0


def _discount(i: int) -> float:
    """Discount for 0-based rank ``i``."""
    return 1.0 / math.log2(i + 2)


def _dedupe(ranked: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for d in ranked:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def dcg_at_k(grades: Sequence[float], k: int) -> float:
    """DCG of ``grades`` (listed in rank order) truncated at ``k``.

    ``k`` larger than ``len(grades)`` simply uses every grade.
    """
    if k <= 0:
        return 0.0
    return sum(_gain(g) * _discount(i) for i, g in enumerate(list(grades)[:k]))


def ideal_dcg_at_k(judgments: Mapping[str, int], k: int) -> float:
    """DCG of the best possible ranking of *all* judged docs of the query."""
    return dcg_at_k(sorted(judgments.values(), reverse=True), k)


def ndcg_at_k(ranked: Sequence[str], judgments: Mapping[str, int], k: int) -> Optional[float]:
    """NDCG@k of one ranking.

    Args:
        ranked: doc ids in rank order (duplicates collapsed to first occurrence).
        judgments: ``doc_id -> grade`` for this query; unjudged docs count as 0.
        k: cutoff.

    Returns:
        A value in [0, 1], or ``None`` if the query has no positively graded
        document (ideal DCG is 0, NDCG undefined).
    """
    idcg = ideal_dcg_at_k(judgments, k)
    if idcg <= 0.0:
        return None
    ranked = _dedupe(ranked)
    dcg = dcg_at_k([judgments.get(d, 0) for d in ranked[:k]], k)
    return dcg / idcg


def mrr(ranked: Sequence[str], judgments: Mapping[str, int],
        rel_threshold: int = DEFAULT_REL_THRESHOLD, k: Optional[int] = None) -> Optional[float]:
    """Reciprocal rank of the first doc with ``grade >= rel_threshold``.

    This is the per-query value; MRR is its mean over queries
    (``evaluate_run`` does the averaging). ``k`` optionally truncates the
    ranking. Returns 0.0 when no relevant doc is retrieved and ``None`` when the
    query has no judged relevant doc at all.
    """
    if not any(g >= rel_threshold for g in judgments.values()):
        return None
    ranked = _dedupe(ranked)
    if k is not None:
        ranked = ranked[:k]
    for i, d in enumerate(ranked):
        if judgments.get(d, 0) >= rel_threshold:
            return 1.0 / (i + 1)
    return 0.0


reciprocal_rank = mrr


def recall_at_k(ranked: Sequence[str], judgments: Mapping[str, int], k: int,
                rel_threshold: int = DEFAULT_REL_THRESHOLD) -> Optional[float]:
    """Fraction of the query's judged-relevant docs found in the top ``k``.

    The denominator is the number of judged docs with ``grade >= rel_threshold``
    (pool-based recall: relevant docs nobody judged are invisible). Returns
    ``None`` if that number is 0.
    """
    relevant = {d for d, g in judgments.items() if g >= rel_threshold}
    if not relevant:
        return None
    top = _dedupe(ranked)[:k]
    return sum(1 for d in top if d in relevant) / len(relevant)


def precision_at_k(ranked: Sequence[str], judgments: Mapping[str, int], k: int,
                   rel_threshold: int = DEFAULT_REL_THRESHOLD) -> Optional[float]:
    """Relevant docs in the top ``k`` divided by ``k`` (trec_eval convention:
    a run shorter than ``k`` is penalised). ``None`` if the query has no judged
    relevant doc."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not any(g >= rel_threshold for g in judgments.values()):
        return None
    top = _dedupe(ranked)[:k]
    return sum(1 for d in top if judgments.get(d, 0) >= rel_threshold) / k


def judged_at_k(ranked: Sequence[str], judgments: Mapping[str, int], k: int) -> Optional[float]:
    """Fraction of the retrieved top ``k`` that carries *any* judgment.

    Denominator is ``min(k, len(ranking))`` — we measure how much of what the
    system actually showed was assessed. Low values mean NDCG for this system
    is unreliable (unjudged docs silently count as 0; classic pool bias).
    ``None`` for an empty ranking.
    """
    top = _dedupe(ranked)[:k]
    if not top:
        return None
    return sum(1 for d in top if d in judgments) / len(top)


# ---------------------------------------------------------------------------
# Run-level evaluation
# ---------------------------------------------------------------------------

def parse_metric(spec: str) -> Tuple[str, Optional[int]]:
    """``"ndcg@10" -> ("ndcg", 10)``, ``"mrr" -> ("mrr", None)``."""
    m = _METRIC_RE.match(spec.strip().lower())
    if not m:
        raise ValueError(f"bad metric spec {spec!r}")
    name, k = m.group(1), m.group(2)
    if name not in _ALIASES:
        raise ValueError(f"unknown metric {name!r} (known: {sorted(set(_ALIASES.values()))})")
    return _ALIASES[name], (int(k) if k is not None else None)


def expand_metrics(metrics: Iterable[str], ks: Sequence[int] = (10,)) -> List[str]:
    """Normalise metric specs; a cutoff metric given without ``@k`` is expanded
    over ``ks`` (``"ndcg"`` with ``ks=(5, 10)`` -> ``ndcg@5, ndcg@10``).
    ``mrr`` without ``@k`` means the full ranking."""
    out: List[str] = []
    for spec in metrics:
        name, k = parse_metric(spec)
        if k is not None:
            names = [f"{name}@{k}"]
        elif name == "mrr":
            names = ["mrr"]
        else:
            names = [f"{name}@{kk}" for kk in ks]
        for n in names:
            if n not in out:
                out.append(n)
    return out


def metric_value(spec: str, ranked: Sequence[str], judgments: Mapping[str, int],
                 rel_threshold: int = DEFAULT_REL_THRESHOLD) -> Optional[float]:
    """Compute one per-query metric given its spec (e.g. ``"recall@100"``)."""
    name, k = parse_metric(spec)
    if name == "mrr":
        return mrr(ranked, judgments, rel_threshold, k)
    if k is None:
        raise ValueError(f"metric {spec!r} needs a cutoff")
    if name == "ndcg":
        return ndcg_at_k(ranked, judgments, k)
    if name == "dcg":
        return dcg_at_k([judgments.get(d, 0) for d in _dedupe(ranked)[:k]], k)
    if name == "recall":
        return recall_at_k(ranked, judgments, k, rel_threshold)
    if name == "precision":
        return precision_at_k(ranked, judgments, k, rel_threshold)
    if name == "judged":
        return judged_at_k(ranked, judgments, k)
    raise ValueError(spec)  # pragma: no cover


@dataclass
class EvalResult:
    """Per-query and mean metric values of one run against one set of qrels.

    ``per_query[metric][qid]`` holds only *defined* values; queries where the
    metric is undefined are listed in ``skipped[metric]``.
    """
    metrics: List[str]
    per_query: Dict[str, Dict[str, float]]
    means: Dict[str, float]
    n_queries: Dict[str, int]
    skipped: Dict[str, List[str]]
    missing_in_run: List[str] = field(default_factory=list)
    unjudged_queries: List[str] = field(default_factory=list)
    rel_threshold: int = DEFAULT_REL_THRESHOLD
    label_source: str = ""   # provenance of the qrels, e.g. "hand"

    def values(self, metric: str) -> Dict[str, float]:
        return self.per_query[metric]

    def mean(self, metric: str) -> float:
        return self.means[metric]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalResult":
        return cls(**d)


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def evaluate_run(run: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]],
                 ks: Sequence[int] = (10,), metrics: Sequence[str] = DEFAULT_METRICS,
                 rel_threshold: int = DEFAULT_REL_THRESHOLD, label_source: str = "") -> EvalResult:
    """Evaluate a run against qrels.

    Args:
        run: ``qid -> ranked doc ids``.
        qrels: ``qid -> {doc_id: grade}``. Only queries with at least one
            judgment are evaluated.
        ks: cutoffs used for metric specs given without ``@k``.
        metrics: metric specs, e.g. ``("ndcg@10", "mrr", "recall@100", "judged@10")``.
        rel_threshold: minimum grade counted as relevant for MRR/recall/precision.
        label_source: provenance tag stored on the result (``"hand"`` for
            reported numbers; anything else is a diagnostic).
    """
    specs = expand_metrics(metrics, ks)
    per_query: Dict[str, Dict[str, float]] = {m: {} for m in specs}
    skipped: Dict[str, List[str]] = {m: [] for m in specs}
    judged_qids = sorted(q for q, j in qrels.items() if j)
    missing = [q for q in judged_qids if q not in run]
    unjudged = sorted(q for q in run if q not in qrels or not qrels[q])
    for qid in judged_qids:
        ranked = _dedupe(run.get(qid, []))
        judgments = qrels[qid]
        for m in specs:
            v = metric_value(m, ranked, judgments, rel_threshold)
            if v is None:
                skipped[m].append(qid)
            else:
                per_query[m][qid] = v
    means = {m: _mean(per_query[m].values()) for m in specs}
    n = {m: len(per_query[m]) for m in specs}
    return EvalResult(metrics=specs, per_query=per_query, means=means, n_queries=n,
                      skipped=skipped, missing_in_run=missing, unjudged_queries=unjudged,
                      rel_threshold=rel_threshold, label_source=label_source)


def per_class(result: EvalResult, qid_to_class: Mapping[str, str],
              unknown: str = "unknown") -> Dict[str, Dict[str, float]]:
    """Mean of every metric per query class.

    Returns ``{class: {metric: mean, ..., "n_queries": int}}`` where
    ``n_queries`` counts the evaluated queries of that class (any metric) and a
    metric's mean is over the queries where it is defined (``nan`` if none).
    Queries missing from ``qid_to_class`` go to ``unknown``.
    """
    classes: Dict[str, Dict[str, List[float]]] = {}
    members: Dict[str, set] = {}
    for m in result.metrics:
        for qid, v in result.per_query[m].items():
            c = qid_to_class.get(qid, unknown)
            classes.setdefault(c, {}).setdefault(m, []).append(v)
            members.setdefault(c, set()).add(qid)
        for qid in result.skipped[m]:
            members.setdefault(qid_to_class.get(qid, unknown), set()).add(qid)
    out: Dict[str, Dict[str, float]] = {}
    for c in sorted(members):
        row: Dict[str, float] = {m: _mean(classes.get(c, {}).get(m, [])) for m in result.metrics}
        row["n_queries"] = len(members[c])
        out[c] = row
    return out
