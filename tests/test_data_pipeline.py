from __future__ import annotations

import json
from collections import defaultdict

import pytest

from dishcovery.data.build_docs import iter_restaurant_groups, scale_docs, subsample_docs
from dishcovery.data.geo import point_in_polygon
from dishcovery.data.hours import osm_hours_to_hex, yelp_hours_to_hex
from dishcovery.data.prepare import prepare
from dishcovery.data.schema import hours_bit, time_slot, validate_doc
from dishcovery.data.taxonomy import Taxonomy


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    stats = prepare("synthetic", raw_dir=str(d / "raw"), out_dir=str(d / "corpus"), n_synthetic=300, seed=3,
                    workers=1)
    docs = [json.loads(l) for l in open(stats["docs_path"])]
    return stats, docs, d


def test_docs_follow_contract(corpus):
    stats, docs, _ = corpus
    assert stats["build"]["restaurants"] > 200
    assert stats["build"]["dishes"] > 10 * stats["build"]["restaurants"]
    for doc in docs:
        assert validate_doc(doc) == [], doc["id"]
    assert {d["type"] for d in docs} == {"restaurant", "dish"}


def test_polygon_contains_restaurant(corpus):
    _, docs, _ = corpus
    for d in docs:
        if d["type"] == "restaurant":
            assert point_in_polygon(d["lat"], d["lon"], d["delivery_polygon"])


def test_chains_share_menus(corpus):
    _, docs, _ = corpus
    menus = defaultdict(list)
    for group in iter_restaurant_groups(corpus[0]["docs_path"]):
        r = group[0]
        if r["chain"]:
            menus[r["chain"]].append(sorted(d["name"] for d in group[1:]))
    multi = {k: v for k, v in menus.items() if len(v) >= 2}
    assert multi, "expected at least one chain with several locations"
    for v in multi.values():
        assert all(m == v[0] for m in v)


def test_hours_parsing():
    h = yelp_hours_to_hex({"Monday": "11:0-22:0", "Friday": "18:0-2:0", "Sunday": "0:0-0:0"})
    assert hours_bit(h, time_slot(0, 12, 0)) and not hours_bit(h, time_slot(0, 23, 0))
    assert hours_bit(h, time_slot(4, 23, 30)) and hours_bit(h, time_slot(5, 1, 0))  # past midnight
    assert all(hours_bit(h, time_slot(6, hh, 0)) for hh in range(24))           # 24h Sunday
    o = osm_hours_to_hex("Mo-Fr 11:00-22:00; Sa,Su 12:00-23:00")
    assert hours_bit(o, time_slot(2, 13, 0)) and not hours_bit(o, time_slot(5, 11, 0))


def test_scale_and_subsample(corpus, tmp_path):
    stats, docs, _ = corpus
    n = len(docs)
    s = scale_docs(stats["docs_path"], str(tmp_path / "big.jsonl"), int(n * 2.5))
    big = [json.loads(l) for l in open(tmp_path / "big.jsonl")]
    assert abs(len(big) - int(n * 2.5)) <= 0.05 * n
    assert len({d["id"] for d in big}) == len(big)
    subsample_docs(stats["docs_path"], str(tmp_path / "small.jsonl"), n // 2)
    groups = list(iter_restaurant_groups(str(tmp_path / "small.jsonl")))
    for g in groups:
        assert g[0]["type"] == "restaurant" and all(d["restaurant_id"] == g[0]["id"] for d in g)


def test_dish_matcher():
    m = Taxonomy.load().dish_matcher()
    assert m.mentions("Their Chicken Biryani and garlic naan were great") == ["chicken biryani", "garlic naan"]
    assert "chicken biryani" in m.mentions("best chicken biriyani in town")  # listed variant
