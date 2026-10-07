"""TREC-style depth-k pooling of system runs for hand labelling.

The pool for a query is the union of every system's top ``depth`` results,
minus pairs that are already judged. Pooling every system that will appear in
a results table keeps ``judged@k`` high for all of them (reported alongside
NDCG to expose residual pool bias). Re-pool whenever a new system is added and
label only the new, unjudged pairs.

Pool file format (JSONL): ``{"qid": ..., "doc_ids": [...]}`` per query.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

from dishcovery.eval.io import read_jsonl, write_jsonl


def build_pool(runs: Mapping[str, Mapping[str, Sequence[str]]], depth: int = 10,
               existing_qrels: Optional[Mapping[str, Mapping[str, int]]] = None,
               qids: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    """Union of each system's top-``depth`` docs per query, unjudged docs only.

    Args:
        runs: ``system -> qid -> ranked doc ids``.
        depth: pool depth per system.
        existing_qrels: already-judged pairs to exclude.
        qids: restrict/extend the query set (default: every qid in any run).

    Returns:
        ``qid -> doc ids`` in a stable order: best (smallest) rank across
        systems first; ties broken by the order systems are iterated in sorted
        name order, then by doc id. Queries with nothing left to judge map to
        an empty list.
    """
    existing_qrels = existing_qrels or {}
    systems = sorted(runs)
    all_qids = list(qids) if qids is not None else sorted({q for s in systems for q in runs[s]})
    pool: Dict[str, List[str]] = {}
    for qid in all_qids:
        judged = existing_qrels.get(qid, {})
        best: Dict[str, tuple] = {}
        for si, s in enumerate(systems):
            seen = set()
            rank = 0
            for d in runs[s].get(qid, []):
                if d in seen:
                    continue
                seen.add(d)
                rank += 1
                if rank > depth:
                    break
                if d in judged:
                    continue
                key = (rank, si, d)
                if d not in best or key < best[d]:
                    best[d] = key
        pool[qid] = [d for d, _ in sorted(best.items(), key=lambda kv: kv[1])]
    return pool


def pool_stats(pool: Mapping[str, Sequence[str]]) -> Dict[str, float]:
    sizes = [len(v) for v in pool.values()]
    return {"queries": len(sizes), "pairs": sum(sizes),
            "mean_per_query": (sum(sizes) / len(sizes)) if sizes else 0.0,
            "max_per_query": max(sizes) if sizes else 0}


def save_pool(path: str, pool: Mapping[str, Sequence[str]]) -> int:
    return write_jsonl(path, ({"qid": q, "doc_ids": list(d)} for q, d in pool.items()))


def load_pool(path: str) -> Dict[str, List[str]]:
    return {str(r["qid"]): [str(d) for d in r["doc_ids"]] for r in read_jsonl(path)}
