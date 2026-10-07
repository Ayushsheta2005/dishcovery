"""Result diversification so the top 10 is not ten biryanis from one kitchen.

* ``mmr`` — Maximal Marginal Relevance (Carbonell & Goldstein, 1998):
  greedily pick argmax λ·rel(d) − (1−λ)·max_{s∈S} sim(d, s).
* ``coverage`` — a submodular objective: relevance plus a concave (sqrt)
  reward per restaurant / cuisine "aspect", greedily maximised (the greedy
  solution is within (1 − 1/e) of optimal for monotone submodular functions).

Similarity between two results is defined from catalogue structure, not
embeddings, so it is cheap and interpretable: same restaurant 1.0, same
canonical dish 0.7, same primary cuisine 0.25.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence


def similarity(a: Mapping, b: Mapping) -> float:
    if a.get("restaurant_id") and a.get("restaurant_id") == b.get("restaurant_id"):
        return 1.0
    if a.get("canonical_dish") and a.get("canonical_dish") == b.get("canonical_dish"):
        return 0.7
    ca, cb = (a.get("cuisines") or [None])[0], (b.get("cuisines") or [None])[0]
    if ca and ca == cb:
        return 0.25
    return 0.0


def _normalise(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    return [1.0] * len(scores) if hi - lo < 1e-12 else [(s - lo) / (hi - lo) for s in scores]


def mmr(items: Sequence[Mapping], scores: Sequence[float], k: int, lam: float = 0.75,
        max_per_restaurant: int = 2) -> List[int]:
    """Return indices of the selected items in order."""
    rel = _normalise(list(scores))
    remaining = list(range(len(items)))
    chosen: List[int] = []
    per_rest: Dict[str, int] = defaultdict(int)
    while remaining and len(chosen) < k:
        best, best_val = None, -1e9
        for i in remaining:
            rid = items[i].get("restaurant_id", "")
            if per_rest[rid] >= max_per_restaurant:
                continue
            red = max((similarity(items[i], items[j]) for j in chosen), default=0.0)
            v = lam * rel[i] - (1 - lam) * red
            if v > best_val:
                best, best_val = i, v
        if best is None:  # only capped restaurants left: relax the cap
            best = max(remaining, key=lambda i: rel[i])
        chosen.append(best)
        remaining.remove(best)
        per_rest[items[best].get("restaurant_id", "")] += 1
    return chosen


def coverage(items: Sequence[Mapping], scores: Sequence[float], k: int, alpha: float = 0.5) -> List[int]:
    rel = _normalise(list(scores))
    chosen: List[int] = []
    mass: Dict[str, float] = defaultdict(float)

    def aspects(it: Mapping) -> List[str]:
        return [f"r:{it.get('restaurant_id', '')}", f"c:{(it.get('cuisines') or [''])[0]}",
                f"d:{it.get('canonical_dish', '')}"]

    remaining = set(range(len(items)))
    while remaining and len(chosen) < k:
        def gain(i: int) -> float:
            g = 0.0
            for a in aspects(items[i]):
                g += math.sqrt(mass[a] + rel[i]) - math.sqrt(mass[a])
            return (1 - alpha) * rel[i] + alpha * g
        best = max(remaining, key=gain)
        chosen.append(best)
        remaining.remove(best)
        for a in aspects(items[best]):
            mass[a] += rel[best]
    return chosen


def intra_list_diversity(items: Sequence[Mapping]) -> Dict[str, float]:
    n = max(len(items), 1)
    return {"distinct_restaurants": len({i.get("restaurant_id") for i in items}) / n,
            "distinct_dishes": len({i.get("canonical_dish") or i.get("id") for i in items}) / n,
            "distinct_cuisines": len({(i.get("cuisines") or [""])[0] for i in items}) / n}
