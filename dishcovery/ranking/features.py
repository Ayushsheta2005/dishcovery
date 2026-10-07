"""Ranking feature schema (~40 features in six groups).

Shard-computed signals (need the index: postings, positions, doc values,
vectors, polygons) arrive from /features; query-level and behavioural features
are added here. The feature list is fixed and versioned so a trained model
and the serving path always agree on column order.
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence

from dishcovery.data.schema import MEAL_BUCKETS

FEATURE_VERSION = 1

FEATURES: List[str] = [
    # textual (shard)
    "bm25_name", "bm25_name_exact", "bm25_restaurant_name", "bm25_cuisines", "bm25_categories", "bm25_tags",
    "bm25_description", "bm25_reviews", "bm25_weighted", "proximity_name", "proximity_restaurant_name",
    "proximity_description", "coverage", "name_coverage", "exact_name", "phrase_name", "name_len",
    # semantic / fusion
    "cosine", "rrf_score", "bm25_rank", "vector_rank", "geo_rank", "in_bm25", "in_vector", "in_geo",
    # entity
    "rating", "log_review_count", "price", "price_tier", "popularity", "is_dish",
    # geo
    "distance_km", "inside_zone", "delivery_radius_km", "est_delivery_min",
    # temporal
    "open_now", "prep_time_min", "time_of_day_match",
    # query-document agreement (from query understanding)
    "intent_type_match", "chain_match", "cuisine_match", "diet_match", "query_len",
    # behavioural (click logs)
    "hist_ctr", "hist_cvr", "doc_ctr",
]
FEATURE_INDEX = {f: i for i, f in enumerate(FEATURES)}
MISSING_RANK = 2000.0


def meal_bucket_of_hour(hour: int) -> str:
    if 6 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 15:
        return "lunch"
    if 15 <= hour < 18:
        return "snack"
    if 18 <= hour < 22:
        return "dinner"
    return "late_night"


def assemble(signals: Mapping[str, float], summary: Mapping, ranks: Mapping[str, int], rrf_score: float,
             sq: Optional[object], hour: int, behavior: Optional[Mapping[str, float]] = None) -> List[float]:
    """One feature row. ``signals`` = shard /features row as a name->value dict;
    ``summary`` = stored summary fields; ``ranks`` = 1-based rank per retrieval path."""
    f = [0.0] * len(FEATURES)

    def put(name: str, v: float) -> None:
        f[FEATURE_INDEX[name]] = float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else 0.0

    for name in FEATURES[:17]:
        put(name, signals.get(name, 0.0))
    put("cosine", signals.get("cosine", 0.0))
    put("rrf_score", rrf_score)
    for p in ("bm25", "vector", "geo"):
        r = ranks.get(p)
        put(f"{p}_rank", r if r else MISSING_RANK)
        put(f"in_{p}", 1.0 if r else 0.0)
    put("rating", signals.get("rating", 0.0))
    put("log_review_count", math.log1p(max(0.0, signals.get("review_count", 0.0))))
    put("price", signals.get("price", 0.0))
    put("price_tier", signals.get("price_tier", 0.0))
    put("popularity", signals.get("popularity", 0.0))
    put("is_dish", signals.get("is_dish", 0.0))
    dist = signals.get("distance_km", 0.0)
    put("distance_km", dist)
    put("inside_zone", signals.get("inside_zone", 0.0))
    put("delivery_radius_km", signals.get("delivery_radius_km", 0.0))
    prep = signals.get("prep_time_min", 20.0)
    put("prep_time_min", prep)
    put("est_delivery_min", prep + 8.0 + dist / 0.4)  # pickup overhead + ~24 km/h courier
    put("open_now", signals.get("open_now", 1.0))
    meal = [signals.get(f"meal_{b}", 0.0) for b in MEAL_BUCKETS]
    put("time_of_day_match", meal[MEAL_BUCKETS.index(meal_bucket_of_hour(hour))])

    if sq is not None:
        is_dish = signals.get("is_dish", 0.0) > 0.5
        intent = getattr(sq, "intent", "dish")
        type_match = {"dish": is_dish, "restaurant": not is_dish, "chain": True, "cuisine": True,
                      "attribute": True}.get(intent, True)
        put("intent_type_match", 1.0 if type_match else 0.0)
        flt = getattr(sq, "filters", None)
        chain = getattr(flt, "chain", None) if flt else None
        put("chain_match", 1.0 if chain and summary.get("chain") == chain else 0.0)
        cz = getattr(flt, "cuisine", None) if flt else None
        put("cuisine_match", 1.0 if cz and cz.replace(" ", "_") in (summary.get("cuisines") or []) else 0.0)
        diets = getattr(flt, "diet", []) if flt else []
        doc_diets = summary.get("diet") or []
        put("diet_match", 1.0 if diets and all(d in doc_diets for d in diets) else 0.0)
        put("query_len", len(getattr(sq, "tokens", []) or []))
    if behavior:
        put("hist_ctr", behavior.get("ctr_smoothed", 0.0))
        put("hist_cvr", behavior.get("cvr_smoothed", 0.0))
        put("doc_ctr", behavior.get("doc_ctr", 0.0))
    return f
