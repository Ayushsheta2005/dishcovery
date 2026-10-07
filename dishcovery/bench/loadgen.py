"""Open-loop load generator.

Arrivals follow a Poisson process at the target QPS and latency is measured
from each request's *intended* start time, so a stalled server is charged for
the queueing it causes (no coordinated omission). In-flight requests are capped;
arrivals beyond the cap are counted as dropped rather than silently delayed.
Per-stage latencies come from the gateway's ``timings_ms``.
"""
from __future__ import annotations

import asyncio
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import aiohttp
import orjson

from dishcovery.serving.metrics import LatencyHistogram


async def run_open_loop(url: str, queries: Sequence[dict], qps: float, duration_s: float, system: str = "full",
                        k: int = 20, max_inflight: int = 512, seed: int = 0, extra: Optional[dict] = None) -> dict:
    rng = random.Random(seed)
    e2e = LatencyHistogram()
    stages: Dict[str, LatencyHistogram] = defaultdict(LatencyHistogram)
    stats = {"sent": 0, "ok": 0, "errors": 0, "dropped": 0, "partial": 0, "cache_hits": 0}
    inflight = 0
    conn = aiohttp.TCPConnector(limit=max_inflight, keepalive_timeout=60)
    async with aiohttp.ClientSession(connector=conn) as session:
        async def one(q: dict, intended: float):
            nonlocal inflight
            params = {"q": q["text"], "k": str(k), "system": system}
            if q.get("lat") is not None:
                params.update(lat=str(q["lat"]), lon=str(q["lon"]))
            if q.get("time"):
                params["time"] = q["time"]
            if extra:
                params.update({kk: str(v) for kk, v in extra.items()})
            try:
                async with session.get(url + "/search", params=params,
                                       timeout=aiohttp.ClientTimeout(total=10)) as r:
                    body = await r.read()
                    done = time.perf_counter()
                    if r.status != 200:
                        stats["errors"] += 1
                        return
                    data = orjson.loads(body)
                stats["ok"] += 1
                e2e.record_ms((done - intended) * 1000.0)
                if data.get("cache_hit"):
                    stats["cache_hits"] += 1
                if (data.get("partial") or {}).get("failed_shards"):
                    stats["partial"] += 1
                for name, v in (data.get("timings_ms") or {}).items():
                    stages[name].record_ms(v)
            except Exception:
                stats["errors"] += 1
            finally:
                inflight -= 1

        tasks: List[asyncio.Task] = []
        t0 = time.perf_counter()
        next_t = t0
        while True:
            next_t += rng.expovariate(qps)
            if next_t - t0 > duration_s:
                break
            delay = next_t - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            stats["sent"] += 1
            if inflight >= max_inflight:
                stats["dropped"] += 1
                continue
            inflight += 1
            tasks.append(asyncio.ensure_future(one(rng.choice(queries), next_t)))
        await asyncio.gather(*tasks, return_exceptions=True)
        wall = time.perf_counter() - t0
    return {"target_qps": qps, "achieved_qps": stats["ok"] / wall, "duration_s": round(wall, 2), **stats,
            "e2e": e2e.summary(), "stages": {k2: h.summary() for k2, h in sorted(stages.items())}}
