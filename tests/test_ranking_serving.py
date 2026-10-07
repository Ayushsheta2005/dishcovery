from __future__ import annotations

import asyncio
import time

import aiohttp
import numpy as np
import pytest
from aiohttp import web

from dishcovery.qu.types import QueryFilters, StructuredQuery
from dishcovery.ranking.diversify import coverage, intra_list_diversity, mmr
from dishcovery.ranking.features import FEATURE_INDEX, FEATURES, assemble
from dishcovery.ranking.fusion import rrf
from dishcovery.ranking.ltr import LTRModel, mean_ndcg
from dishcovery.serving.cache import ResultCache
from dishcovery.serving.shard_client import ClientOptions, ShardError, ShardGroup


def test_rrf_prefers_consensus():
    fused = dict(rrf({"bm25": ["a", "b", "c"], "vector": ["b", "a", "d"]}, k=60))
    assert fused["a"] == pytest.approx(fused["b"])
    assert fused["a"] > fused["c"] > 0 and fused["d"] > 0
    assert rrf({"x": ["a"]}, weights={"x": 0.0}) == []


def test_mmr_spreads_restaurants():
    items = [{"restaurant_id": "r1", "canonical_dish": "biryani", "cuisines": ["indian"]}] * 5 + \
            [{"restaurant_id": "r2", "canonical_dish": "pizza", "cuisines": ["italian"]}]
    scores = [1.0, 0.99, 0.98, 0.97, 0.96, 0.5]
    sel = mmr(items, scores, k=3, lam=0.7, max_per_restaurant=2)
    assert 5 in sel and len(sel) == 3
    assert intra_list_diversity([items[i] for i in sel])["distinct_restaurants"] > 0.5
    assert len(coverage(items, scores, k=3)) == 3


def test_result_cache_invalidates_on_generation():
    c = ResultCache(capacity=2, ttl_s=60)
    c.put("q", (1, 1), {"r": 1})
    assert c.get("q", (1, 1)) == {"r": 1}
    assert c.get("q", (1, 2)) is None        # shard 1 changed -> stale
    c.put("a", (1,), 1)
    c.put("b", (1,), 2)
    c.put("c", (1,), 3)                      # evicts LRU
    assert c.get("a", (1,)) is None and c.stats()["evictions"] == 1


def test_feature_assembly_columns():
    sq = StructuredQuery(raw="vegan pizza", intent="dish", tokens=["pizza"], filters=QueryFilters(diet=["vegan"]))
    sig = {"bm25_name": 3.0, "is_dish": 1.0, "distance_km": 2.0, "prep_time_min": 10.0, "open_now": 1.0,
           "meal_lunch": 0.8, "review_count": 99.0}
    row = assemble(sig, {"diet": ["vegan", "veg"]}, {"bm25": 1}, 0.03, sq, hour=13,
                   behavior={"ctr_smoothed": 0.2, "cvr_smoothed": 0.1, "doc_ctr": 0.05})
    assert len(row) == len(FEATURES)
    assert row[FEATURE_INDEX["bm25_name"]] == 3.0
    assert row[FEATURE_INDEX["vector_rank"]] == 2000.0 and row[FEATURE_INDEX["in_bm25"]] == 1.0
    assert row[FEATURE_INDEX["time_of_day_match"]] == pytest.approx(0.8)
    assert row[FEATURE_INDEX["diet_match"]] == 1.0 and row[FEATURE_INDEX["intent_type_match"]] == 1.0
    assert row[FEATURE_INDEX["est_delivery_min"]] == pytest.approx(10 + 8 + 5)
    assert row[FEATURE_INDEX["hist_ctr"]] == 0.2


def test_numpy_lambdamart_learns_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    n, G = 30, 80
    X = rng.normal(size=(n * G, len(FEATURES)))
    y = np.digitize(X[:, 0] * 2 + X[:, 1], [-1, 1, 2]).astype(float)
    m = LTRModel("numpy", {"n_trees": 30}).fit(X[: 60 * n], y[: 60 * n], [n] * 60, X[60 * n:], y[60 * n:], [n] * 20,
                                               log=lambda *_: None)
    assert mean_ndcg(m.predict(X[60 * n:]), y[60 * n:], [n] * 20) > 0.9
    m.save(str(tmp_path / "m.pkl"))
    m2 = LTRModel.load(str(tmp_path / "m.pkl"))
    assert np.allclose(m.predict(X[:10]), m2.predict(X[:10]))


@pytest.fixture
def fake_replicas(unused_tcp_port_factory=None):
    """Two fake shard replicas: r0 is slow, r1 is fast."""
    async def make(delay_s: float, port: int, fail: bool = False):
        async def search(req):
            await asyncio.sleep(delay_s)
            if fail:
                return web.Response(status=503)
            return web.json_response({"generation": 7, "who": port})
        app = web.Application()
        app.router.add_post("/search", search)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()
        return runner
    return make


def _free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_hedging_and_failover(fake_replicas):
    async def go():
        p_slow, p_fast, p_dead = _free_port(), _free_port(), _free_port()
        r1 = await fake_replicas(0.3, p_slow)
        r2 = await fake_replicas(0.0, p_fast)
        r3 = await fake_replicas(0.0, p_dead, fail=True)
        async with aiohttp.ClientSession() as s:
            g = ShardGroup(0, [f"http://127.0.0.1:{p_slow}", f"http://127.0.0.1:{p_fast}"], s,
                           ClientOptions(timeout_s=2.0, hedge=True, hedge_min_ms=20, hedge_budget=1.0))
            g.recent_ms.extend([5.0] * 50)       # p95 ~ 5 ms -> hedge after max(20, 5) ms
            g.replicas[1].ewma_ms = 1e6          # force the slow replica to be picked first
            t0 = time.perf_counter()
            name, body, _ = await g.request("/search", {})
            took = time.perf_counter() - t0
            assert body["who"] == p_fast and took < 0.25 and g.stats["hedges"] == 1
            assert g.generation == 7
            # Failover: a 503 replica is retried on the healthy one.
            g2 = ShardGroup(1, [f"http://127.0.0.1:{p_dead}", f"http://127.0.0.1:{p_fast}"], s,
                            ClientOptions(timeout_s=1.0, hedge=False))
            g2.replicas[1].ewma_ms = 1e6
            name, body, _ = await g2.request("/search", {})
            assert body["who"] == p_fast and g2.stats["failovers"] == 1
            # All replicas down -> ShardError.
            g3 = ShardGroup(2, [f"http://127.0.0.1:{p_dead}"], s, ClientOptions(timeout_s=1.0))
            with pytest.raises(ShardError):
                await g3.request("/search", {})
        for r in (r1, r2, r3):
            await r.cleanup()
    asyncio.run(go())
