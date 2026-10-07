"""Shard replica groups: load balancing, hedged requests, failover.

Tail-at-scale techniques (Dean & Barroso, 2013):

* **Replica selection** — power-of-two-choices over (outstanding requests,
  EWMA latency), so a slow or overloaded replica sheds traffic.
* **Hedged requests** — if the chosen replica has not answered after the
  shard's recent p95 latency, send the same request to a second replica and
  take whichever answers first. A token bucket caps hedges at ``hedge_budget``
  of requests so hedging cannot double the load under overload.
* **Failover** — a failed request is retried once on another replica;
  replicas that keep failing are skipped for a cool-down period.
"""
from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiohttp
import orjson


class ShardError(Exception):
    def __init__(self, msg: str, status: int = 0) -> None:
        super().__init__(msg)
        self.status = status


@dataclass
class ReplicaState:
    url: str
    name: str
    outstanding: int = 0
    ewma_ms: float = 5.0
    failures: int = 0
    down_until: float = 0.0
    requests: int = 0

    def healthy(self, now: float) -> bool:
        return now >= self.down_until

    def load_score(self) -> float:
        return (self.outstanding + 1) * self.ewma_ms


@dataclass
class ClientOptions:
    timeout_s: float = 2.0
    hedge: bool = True
    hedge_min_ms: float = 2.0
    hedge_quantile: float = 0.95
    hedge_budget: float = 0.05        # max fraction of requests that may be hedged
    retry: bool = True
    cooldown_s: float = 5.0


class _HedgeBudget:
    """Token bucket refilled by a fraction of every request."""

    def __init__(self, fraction: float, burst: float = 10.0) -> None:
        self.fraction = fraction
        self.tokens = burst
        self.burst = burst

    def on_request(self) -> None:
        self.tokens = min(self.burst, self.tokens + self.fraction)

    def take(self) -> bool:
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class ShardGroup:
    def __init__(self, shard_id: int, urls: Sequence[str], session: aiohttp.ClientSession,
                 opts: Optional[ClientOptions] = None) -> None:
        self.shard_id = shard_id
        self.replicas = [ReplicaState(u.rstrip("/"), f"s{shard_id}r{i}") for i, u in enumerate(urls)]
        self.session = session
        self.opts = opts or ClientOptions()
        self.recent_ms: deque = deque(maxlen=512)
        self.budget = _HedgeBudget(self.opts.hedge_budget)
        self.stats = {"requests": 0, "hedges": 0, "hedge_wins": 0, "failovers": 0, "errors": 0}
        self.generation = 0

    # ------------------------------------------------------------ selection
    def _pick(self, exclude: Sequence[ReplicaState] = ()) -> Optional[ReplicaState]:
        now = time.monotonic()
        cands = [r for r in self.replicas if r not in exclude and r.healthy(now)]
        if not cands:
            cands = [r for r in self.replicas if r not in exclude]
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        a, b = random.sample(cands, 2)
        return a if a.load_score() <= b.load_score() else b

    def hedge_delay_ms(self) -> float:
        if len(self.recent_ms) < 20:
            return max(self.opts.hedge_min_ms, 50.0)
        xs = sorted(self.recent_ms)
        return max(self.opts.hedge_min_ms, xs[int(self.opts.hedge_quantile * (len(xs) - 1))])

    # ------------------------------------------------------------- requests
    async def _send(self, rep: ReplicaState, method: str, path: str, payload: Any) -> Tuple[ReplicaState, dict, float]:
        rep.outstanding += 1
        rep.requests += 1
        t0 = time.perf_counter()
        try:
            timeout = aiohttp.ClientTimeout(total=self.opts.timeout_s)
            data = orjson.dumps(payload) if payload is not None else None
            async with self.session.request(method, rep.url + path, data=data, timeout=timeout,
                                            headers={"Content-Type": "application/json"}) as resp:
                body = await resp.read()
                if resp.status != 200:
                    raise ShardError(f"{rep.name} {path} -> HTTP {resp.status}: {body[:200]!r}", resp.status)
                out = orjson.loads(body)
            ms = (time.perf_counter() - t0) * 1000.0
            rep.ewma_ms = 0.8 * rep.ewma_ms + 0.2 * ms
            rep.failures = 0
            return rep, out, ms
        except ShardError as e:
            if e.status >= 500:
                self._mark_failure(rep)
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            self._mark_failure(rep)
            raise ShardError(f"{rep.name} {path}: {type(e).__name__} {e}") from e
        finally:
            rep.outstanding -= 1

    def _mark_failure(self, rep: ReplicaState) -> None:
        rep.failures += 1
        rep.ewma_ms = min(rep.ewma_ms * 2 + 10, 5000.0)
        if rep.failures >= 3:
            rep.down_until = time.monotonic() + self.opts.cooldown_s

    async def request(self, path: str, payload: Any = None, method: str = "POST", hedge: Optional[bool] = None,
                      pin: Optional[str] = None) -> Tuple[str, dict, float]:
        """Returns (replica name, json, latency ms). ``pin`` forces a replica (phase-2 contexts)."""
        self.stats["requests"] += 1
        self.budget.on_request()
        if pin is not None:
            rep = next(r for r in self.replicas if r.name == pin)
            rep, out, ms = await self._send(rep, method, path, payload)
            return rep.name, out, ms
        primary = self._pick()
        if primary is None:
            raise ShardError(f"shard {self.shard_id}: no replicas")
        do_hedge = (self.opts.hedge if hedge is None else hedge) and len(self.replicas) > 1
        t0 = time.perf_counter()
        first = asyncio.ensure_future(self._send(primary, method, path, payload))
        tasks = [first]
        used = [primary]
        try:
            if do_hedge:
                done, _ = await asyncio.wait(tasks, timeout=self.hedge_delay_ms() / 1000.0)
                if not done and self.budget.take():
                    backup = self._pick(exclude=[primary])
                    if backup is not None:
                        used.append(backup)
                        self.stats["hedges"] += 1
                        tasks.append(asyncio.ensure_future(self._send(backup, method, path, payload)))
            last_err: Optional[BaseException] = None
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    if t.exception() is None:
                        rep, out, _ = t.result()
                        if t is not first:
                            self.stats["hedge_wins"] += 1
                        for p in pending:
                            p.cancel()
                        ms = (time.perf_counter() - t0) * 1000.0
                        self.recent_ms.append(ms)
                        self.generation = max(self.generation, int(out.get("generation", 0) or 0))
                        return rep.name, out, ms
                    last_err = t.exception()
            # Every attempt failed: one failover try on a different replica.
            if self.opts.retry and len(self.replicas) > 1:
                backup = self._pick(exclude=used)
                if backup is not None:
                    self.stats["failovers"] += 1
                    rep, out, ms = await self._send(backup, method, path, payload)
                    return rep.name, out, (time.perf_counter() - t0) * 1000.0
            self.stats["errors"] += 1
            raise last_err if last_err else ShardError("unknown failure")
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    async def broadcast(self, path: str, payload: Any = None, method: str = "POST") -> List[Tuple[str, Optional[dict]]]:
        """Send to every replica (writes): returns per-replica result or None on failure."""
        async def one(rep):
            try:
                _, out, _ = await self._send(rep, method, path, payload)
                self.generation = max(self.generation, int(out.get("generation", 0) or 0))
                return rep.name, out
            except ShardError:
                return rep.name, None
        return list(await asyncio.gather(*(one(r) for r in self.replicas)))

    def describe(self) -> dict:
        return {"shard": self.shard_id, "generation": self.generation, **self.stats,
                "hedge_delay_ms": round(self.hedge_delay_ms(), 2),
                "replicas": [{"name": r.name, "url": r.url, "ewma_ms": round(r.ewma_ms, 2),
                              "outstanding": r.outstanding, "requests": r.requests,
                              "healthy": r.healthy(time.monotonic())} for r in self.replicas]}
