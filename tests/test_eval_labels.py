from __future__ import annotations

import math
import random

import pytest

from dishcovery.eval.metrics import evaluate_run, mrr, ndcg_at_k, recall_at_k
from dishcovery.eval.stats import bootstrap_ci, paired_randomization_test
from dishcovery.labels.clicks import ctr_table, ipw_click_labels, simulate
from dishcovery.labels.pooling import build_pool
from dishcovery.labels.queries import QUERY_CLASSES, load_queries
from dishcovery.labels.weak import WeakLabeler


def test_ndcg_hand_computed():
    judg = {"a": 3, "b": 2, "c": 0, "d": 1}
    ranked = ["c", "a", "d"]
    dcg = (2 ** 0 - 1) / math.log2(2) + (2 ** 3 - 1) / math.log2(3) + (2 ** 1 - 1) / math.log2(4)
    idcg = 7 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(ranked, judg, 10) == pytest.approx(dcg / idcg)
    assert mrr(ranked, judg) == pytest.approx(1 / 2)
    assert recall_at_k(ranked, judg, 100) == pytest.approx(0.5)  # a found, b missing (rel >= 2)


def test_evaluate_run_and_stats():
    qrels = {"q1": {"a": 3}, "q2": {"b": 2}}
    res = evaluate_run({"q1": ["a"], "q2": ["x", "b"]}, qrels)
    assert res.means["ndcg@10"] > 0.5
    ci = bootstrap_ci([0.5, 0.6, 0.7, 0.8], n_resamples=2000, seed=0)
    assert ci.lo <= 0.65 <= ci.hi
    p = paired_randomization_test([1.0] * 20, [0.0] * 20, n=2000, seed=0)
    assert p < 0.01


def test_eval_queries_cover_classes():
    qs = load_queries()
    assert len(qs) == 50
    for c in QUERY_CLASSES:
        assert sum(q.query_class == c for q in qs) == 5


def test_pbm_click_model_recovers_bias():
    docs = [f"d{i}" for i in range(10)]
    qs = [{"qid": "q", "text": "pizza"}]
    imps = simulate(qs, ranker=lambda q: docs, grader=lambda q, d: 3, n_sessions=20000, seed=1)
    clicks_by_rank = [0] * 10
    for imp in imps:
        for r, c in enumerate(imp.clicks):
            clicks_by_rank[r] += c
    ctr1, ctr5 = clicks_by_rank[0] / len(imps), clicks_by_rank[4] / len(imps)
    assert ctr1 == pytest.approx(0.95, abs=0.02)
    assert ctr5 == pytest.approx(0.95 / 5, abs=0.02)
    labels = ipw_click_labels(imps, eta=1.0)
    assert labels[("q", "d4")]["ipw"] == pytest.approx(0.95, abs=0.1)
    tbl = ctr_table(imps)
    assert tbl is not None


def test_pooling():
    runs = {"s1": {"q": ["a", "b", "c"]}, "s2": {"q": ["c", "d"]}}
    pool = build_pool(runs, depth=2)
    assert set(pool["q"]) == {"a", "b", "c", "d"}


def test_weak_labeler_examples():
    poly = [[39.90, -75.20], [39.90, -75.10], [40.00, -75.10], [40.00, -75.20]]
    base = dict(restaurant_id="r:1", restaurant_name="Taj", chain="", categories=["Indian"], price_tier=2,
                rating=4.5, review_count=100, lat=39.95, lon=-75.15, city="Philadelphia", state="PA",
                hours="f" * 84, delivery_polygon=poly, delivery_radius_km=4, source="synthetic", tags=["rice"],
                reviews="", description="", popularity=0.5, prep_time_min=20, meal_affinity=[0.2] * 5,
                price=12.0, cuisines=["indian"])
    docs = {
        "d:1:0": dict(base, id="d:1:0", type="dish", name="Chicken Biryani", canonical_dish="chicken biryani",
                      diet=["non_veg"]),
        "d:1:1": dict(base, id="d:1:1", type="dish", name="Veg Biryani", canonical_dish="veg biryani",
                      diet=["veg"]),
        "r:1": dict(base, id="r:1", type="restaurant", name="Taj", canonical_dish="", diet=["veg", "non_veg"]),
    }
    wl = WeakLabeler(docs)
    inside = {"lat": 39.95, "lon": -75.15}
    outside = {"lat": 41.0, "lon": -75.15}
    assert wl.grade("chicken biryani", docs["d:1:0"], ctx=inside) == 3
    assert wl.grade("chicken biryani", docs["d:1:0"], ctx=outside) == 0
    assert wl.grade("vegan biryani", docs["d:1:0"], ctx=inside) <= 1
