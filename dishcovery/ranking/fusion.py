"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, 2009).

score(d) = Σ_paths w_p / (k + rank_p(d)), ranks 1-based; docs missing from a
path contribute nothing. Rank-based, so BM25 scores, cosine similarities and
geo priors never need calibrating against each other.
"""
from __future__ import annotations

from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple


def rrf(ranked_lists: Mapping[str, Sequence[Hashable]], k: float = 60.0,
        weights: Optional[Mapping[str, float]] = None) -> List[Tuple[Hashable, float]]:
    scores: Dict[Hashable, float] = {}
    for path, docs in ranked_lists.items():
        w = (weights or {}).get(path, 1.0)
        if w <= 0:
            continue
        for r, d in enumerate(docs, start=1):
            scores[d] = scores.get(d, 0.0) + w / (k + r)
    return sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
