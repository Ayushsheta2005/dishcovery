"""The broker: query understanding → scatter-gather retrieval → fusion →
feature fetch → LambdaMART → diversification → facets.

    phase 1  POST /search    on one replica of every shard (hedged); each shard
                             returns per-path top-k (segment, doc, score) under a
                             point-in-time context
    fuse     per-path global merge by score, then Reciprocal Rank Fusion
    phase 2  POST /features  for the kept candidates, pinned to the replica
                             that answered phase 1 (its context id)
    rank     LambdaMART over ~45 features; MMR diversification; facets

Partial failures degrade gracefully: a shard that fails or times out is
reported in ``partial`` and the remaining shards' results are served.
"""
from __future__ import annotations

import asyncio
import base64
import datetime as dt
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiohttp
import numpy as np

from dishcovery.config import Config
from dishcovery.data.schema import time_slot
from dishcovery.qu.text import normalize
from dishcovery.qu.types import QueryTerm, StructuredQuery
from dishcovery.ranking.behavior import BehaviorStore
from dishcovery.ranking.diversify import coverage, mmr
from dishcovery.ranking.features import FEATURES, assemble
from dishcovery.ranking.fusion import rrf
from dishcovery.serving.cache import ResultCache
from dishcovery.serving.metrics import Metrics
from dishcovery.serving.shard_client import ClientOptions, ShardError, ShardGroup

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass(frozen=True)
class SystemConfig:
    name: str
    qu: bool = True
    paths: Tuple[str, ...] = ("bm25",)
    rerank: Optional[str] = None        # "ltr"
    diversify: Optional[str] = None     # "mmr" | "coverage"
    qu_overrides: Tuple[Tuple[str, Any], ...] = ()


SYSTEMS: Dict[str, SystemConfig] = {
    "exact": SystemConfig("exact", qu=False, paths=("exact",)),
    "bm25": SystemConfig("bm25", qu=False, paths=("bm25",)),
    "bm25_qu": SystemConfig("bm25_qu", qu=True, paths=("bm25",)),
    "vector": SystemConfig("vector", qu=True, paths=("vector",)),
    "hybrid": SystemConfig("hybrid", qu=True, paths=("bm25", "vector", "geo")),
    "ltr": SystemConfig("ltr", qu=True, paths=("bm25", "vector", "geo"), rerank="ltr"),
    "full": SystemConfig("full", qu=True, paths=("bm25", "vector", "geo"), rerank="ltr", diversify="mmr"),
    # QU stage ablations on top of the hybrid retriever
    "hybrid_nospell": SystemConfig("hybrid_nospell", paths=("bm25", "vector", "geo"),
                                   qu_overrides=(("spell", False), ("segment", False))),
    "hybrid_noexpand": SystemConfig("hybrid_noexpand", paths=("bm25", "vector", "geo"),
                                    qu_overrides=(("expand", False),)),
    "bm25_vector": SystemConfig("bm25_vector", paths=("bm25", "vector")),
}

# Intent-dependent field weights (others keep their schema weight).
INTENT_FIELD_WEIGHTS: Dict[str, Dict[str, float]] = {
    "restaurant": {"restaurant_name": 4.0, "name": 3.0, "cuisines": 1.0},
    "chain": {"restaurant_name": 4.0, "name": 2.0},
    "cuisine": {"cuisines": 4.0, "categories": 2.5, "name": 1.5},
    "dish": {},
    "attribute": {},
}


def parse_when(when: Optional[str]) -> Tuple[int, int, int]:
    """'tue 13:00' -> (1, 13, 0); None -> local now."""
    if not when:
        now = dt.datetime.now()
        return now.weekday(), now.hour, now.minute
    day, hm = when.strip().lower().split()
    h, m = hm.split(":")
    return DAYS.index(day[:3]), int(h), int(m)


class _LRU(OrderedDict):
    def __init__(self, cap: int) -> None:
        super().__init__()
        self.cap = cap

    def get_or(self, key, fn):
        if key in self:
            self.move_to_end(key)
            return self[key]
        v = fn()
        self[key] = v
        if len(self) > self.cap:
            self.popitem(last=False)
        return v


@dataclass
class SearchParams:
    q: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    when: Optional[str] = None
    k: int = 20
    system: str = "full"
    debug: bool = False
    use_cache: bool = True
    filter_mode: Optional[str] = None
    ef: Optional[int] = None
    filters: Dict[str, Any] = field(default_factory=dict)   # explicit facet selections
    return_features: bool = False                           # training: all candidates + feature rows
    depth: Optional[int] = None                             # override candidate depth
    hedge: Optional[bool] = None                            # override hedging (ablation)


class Broker:
    def __init__(self, cfg: Config, qu=None, encoder=None, ltr=None, behavior: Optional[BehaviorStore] = None,
                 shard_urls: Optional[List[List[str]]] = None) -> None:
        self.cfg = cfg
        self.bc = cfg["broker"]
        self.qu = qu
        self.encoder = encoder
        self.ltr = ltr
        self.behavior = behavior or BehaviorStore(None)
        self._shard_urls = shard_urls or cfg.shard_urls()
        self.groups: List[ShardGroup] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = ResultCache(self.bc["cache_capacity"], self.bc["cache_ttl_s"])
        self.metrics = Metrics()
        self._qvec_cache = _LRU(20000)
        self._qu_cache = _LRU(20000)
        self._poller: Optional[asyncio.Task] = None

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> "Broker":
        # cpp-httplib serves one keep-alive connection per thread, so the pool per
        # replica must stay below the shard's thread count (see serving.shard_threads).
        conn = aiohttp.TCPConnector(limit=0, limit_per_host=int(self.bc.get("conns_per_shard", 16)),
                                    keepalive_timeout=30)
        self.session = aiohttp.ClientSession(connector=conn)
        opts = ClientOptions(timeout_s=self.bc["timeout_s"], hedge=self.bc["hedge"],
                             hedge_budget=self.bc["hedge_budget"])
        self.groups = [ShardGroup(i, urls, self.session, opts) for i, urls in enumerate(self._shard_urls)]
        if self.bc.get("poll_generations_s"):
            self._poller = asyncio.ensure_future(self._poll_generations(self.bc["poll_generations_s"]))
        return self

    async def close(self) -> None:
        if self._poller:
            self._poller.cancel()
        if self.session:
            await self.session.close()

    async def __aenter__(self) -> "Broker":
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def _poll_generations(self, every_s: float) -> None:
        while True:
            await asyncio.sleep(every_s)
            await self.refresh_generations()

    async def refresh_generations(self) -> None:
        async def one(g: ShardGroup):
            try:
                await g.request("/health", None, method="GET", hedge=False)
            except ShardError:
                pass
        await asyncio.gather(*(one(g) for g in self.groups))

    def generations(self) -> Tuple[int, ...]:
        return tuple(g.generation for g in self.groups)

    # ------------------------------------------------------------- helpers
    def _understand(self, q: str, sysc: SystemConfig) -> StructuredQuery:
        if sysc.qu and self.qu is not None:
            def run():
                qu = self.qu.with_config(**dict(sysc.qu_overrides)) if sysc.qu_overrides else self.qu
                return qu.parse(q)
            return self._qu_cache.get_or((q, sysc.qu_overrides), run)
        sq = StructuredQuery(raw=q, normalized=normalize(q))
        sq.corrected = sq.normalized
        sq.tokens = sq.normalized.split()
        sq.terms = [QueryTerm(t, 1.0, i, "query") for i, t in enumerate(sq.tokens)]
        sq.intent = "dish"
        return sq

    def _qvec(self, text: str) -> Optional[str]:
        if self.encoder is None or not text:
            return None

        def enc():
            v = np.asarray(self.encoder.encode_query(text), dtype="<f4")
            return base64.b64encode(v.tobytes()).decode()
        return self._qvec_cache.get_or(text, enc)

    def _filters(self, sq: StructuredQuery, p: SearchParams, slot: int) -> Dict[str, Any]:
        f: Dict[str, Any] = {"terms": {}, "range": {}}
        qf = sq.filters
        if qf.diet:
            f["terms"]["diet"] = list(qf.diet)
        if qf.chain:
            f["terms"]["chain"] = [qf.chain]
        if qf.doc_type:
            f["terms"]["type"] = [qf.doc_type]
        if qf.price_tier_max is not None or qf.price_tier_min is not None:
            r = {}
            if qf.price_tier_max is not None:
                r["lte"] = qf.price_tier_max
            if qf.price_tier_min is not None:
                r["gte"] = qf.price_tier_min
            f["range"]["price_tier"] = r
        if qf.max_price is not None:
            f["range"]["price"] = {"lte": qf.max_price}
        if qf.min_rating is not None:
            f["range"]["rating"] = {"gte": qf.min_rating}
        if qf.open_now:
            f["open_slot"] = slot
        for k, v in (p.filters or {}).items():  # explicit facet selections from the UI
            if k in ("price_tier", "rating", "price") and isinstance(v, dict):
                f["range"][k] = v
            else:
                f["terms"][k] = v if isinstance(v, list) else [v]
        if p.lat is not None and p.lon is not None and self.bc.get("geo_filter", True):
            if sq.radius_km:
                f["geo"] = {"mode": "radius", "lat": p.lat, "lon": p.lon, "radius_km": sq.radius_km}
            else:
                f["geo"] = {"mode": "delivery", "lat": p.lat, "lon": p.lon}
        return f

    # --------------------------------------------------------------- search
    async def search(self, p: SearchParams) -> Dict[str, Any]:
        t_start = time.perf_counter()
        timings: Dict[str, float] = {}
        sysc = SYSTEMS[p.system]
        dow, hour, minute = parse_when(p.when)
        slot = time_slot(dow, hour, minute)
        cache_key = None
        if p.use_cache and not p.return_features and not p.debug:
            cell = (round(p.lat, 3), round(p.lon, 3)) if p.lat is not None else None
            cache_key = (p.system, normalize(p.q), cell, slot // 2, p.k, p.filter_mode, p.ef, p.hedge,
                         tuple(sorted((k, str(v)) for k, v in p.filters.items())))
            hit = self.cache.get(cache_key, self.generations())
            if hit is not None:
                self.metrics.inc("cache_hits")
                out = dict(hit, cache_hit=True)
                out["timings_ms"] = {"total": round((time.perf_counter() - t_start) * 1000, 3)}
                self.metrics.observe_ms("e2e", out["timings_ms"]["total"])
                return out

        t = time.perf_counter()
        sq = self._understand(p.q, sysc)
        timings["qu"] = (time.perf_counter() - t) * 1000

        terms = [{"text": x.text, "weight": x.weight, "group": x.group, "primary": x.source != "synonym"}
                 for x in sq.terms]
        field_weights = INTENT_FIELD_WEIGHTS.get(sq.intent, {}) if sysc.qu else {}
        flt = self._filters(sq, p, slot)
        paths: Dict[str, Any] = {}
        if "bm25" in sysc.paths and terms:
            paths["bm25"] = {"k": self.bc["bm25_k"]}
        if "exact" in sysc.paths and terms:
            paths["exact"] = {"k": self.bc["exact_k"]}
        t = time.perf_counter()
        qvec = None
        if "vector" in sysc.paths and sq.corrected:
            qvec = self._qvec(sq.corrected)
            if qvec:
                paths["vector"] = {"k": self.bc["vector_k"], "ef": p.ef or self.bc["ef"], "vector_b64": qvec}
        timings["embed"] = (time.perf_counter() - t) * 1000
        if "geo" in sysc.paths and "geo" in flt:
            paths["geo"] = {"k": self.bc["geo_k"]}
        if not paths:  # attribute-only query on a text-only system: fall back to geo popularity
            if "geo" in flt:
                paths["geo"] = {"k": self.bc["geo_k"]}
            else:
                return self._empty(p, sq, timings, t_start)

        req = {"terms": terms, "field_weights": field_weights, "filter": flt, "paths": paths,
               "filter_mode": p.filter_mode or self.bc["filter_mode"]}

        # ---- phase 1: scatter-gather
        t = time.perf_counter()
        results = await asyncio.gather(*(g.request("/search", req, hedge=p.hedge) for g in self.groups),
                                       return_exceptions=True)
        timings["phase1"] = (time.perf_counter() - t) * 1000
        ok: Dict[int, Tuple[str, dict]] = {}
        failed: List[int] = []
        shard_timings: Dict[str, float] = defaultdict(float)
        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                failed.append(i)
                self.metrics.inc("shard_errors")
                continue
            rep, body, _ = r
            ok[i] = (rep, body)
            for k2, v in (body.get("timings_us") or {}).items():
                shard_timings[k2] = max(shard_timings[k2], v / 1000.0)

        # ---- fuse
        t = time.perf_counter()
        per_path: Dict[str, List[Tuple[float, Tuple[int, int, int]]]] = defaultdict(list)
        for i, (_, body) in ok.items():
            for path, pr in (body.get("paths") or {}).items():
                for seg, doc, score in pr["hits"]:
                    per_path[path].append((score, (i, seg, doc)))
        ranked: Dict[str, List[Tuple[int, int, int]]] = {}
        for path, hits in per_path.items():
            hits.sort(key=lambda x: (-x[0], x[1]))
            ranked[path] = [h[1] for h in hits]
        if len(ranked) == 1:
            path = next(iter(ranked))
            fused = [(key, 1.0 / (self.bc["rrf_k"] + r)) for r, key in enumerate(ranked[path], start=1)]
        else:
            fused = rrf(ranked, k=self.bc["rrf_k"])
        rank_of: Dict[str, Dict[Tuple[int, int, int], int]] = {
            path: {key: r for r, key in enumerate(keys, start=1)} for path, keys in ranked.items()}
        depth = p.depth or max(p.k, self.bc["facet_depth"],
                               self.bc["rerank_depth"] if (sysc.rerank or p.return_features) else 0)
        cands = fused[:depth]
        timings["fuse"] = (time.perf_counter() - t) * 1000

        # ---- phase 2: features + stored summaries, pinned to the phase-1 replica
        t = time.perf_counter()
        by_shard: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for (i, seg, doc), _ in cands:
            by_shard[i].append((seg, doc))
        feat_req_base = {"terms": terms, "field_weights": field_weights,
                         "user": {"lat": p.lat, "lon": p.lon, "slot": slot}}
        if qvec:
            feat_req_base["vector_b64"] = qvec

        async def fetch(i: int):
            rep, body = ok[i]
            payload = dict(feat_req_base, ctx=body["ctx"], docs=by_shard[i])
            return i, await self.groups[i].request("/features", payload, pin=rep)

        feats = await asyncio.gather(*(fetch(i) for i in by_shard), return_exceptions=True)
        timings["phase2"] = (time.perf_counter() - t) * 1000
        info: Dict[Tuple[int, int, int], Tuple[Dict[str, float], dict, str]] = {}
        for r in feats:
            if isinstance(r, BaseException):
                self.metrics.inc("feature_errors")
                continue
            i, (_, body, _) = r
            names = body["names"]
            for (seg, doc), row, did, summary in zip(by_shard[i], body["rows"], body["ids"], body.get("docs") or []):
                info[(i, seg, doc)] = (dict(zip(names, row)), summary or {}, did)

        # ---- rank
        t = time.perf_counter()
        cand_keys = [key for key, _ in cands if key in info]
        fused_score = dict(cands)
        X = None
        if sysc.rerank == "ltr" or p.return_features:
            rows = []
            for key in cand_keys:
                sig, summary, did = info[key]
                ranks = {path: rank_of[path].get(key) for path in rank_of}
                beh = self.behavior.lookup(p.q, did) if self.behavior.table else None
                rows.append(assemble(sig, summary, ranks, fused_score[key], sq, hour, beh))
            X = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(FEATURES)))
        if sysc.rerank == "ltr" and self.ltr is not None and len(cand_keys):
            # Model inference is CPU-bound: keep it off the event loop so other
            # in-flight requests keep making progress (LightGBM releases the GIL).
            scores = await asyncio.get_event_loop().run_in_executor(None, self.ltr.predict, X)
            order = np.argsort(-scores, kind="stable")
            final = [(cand_keys[j], float(scores[j])) for j in order]
        else:
            final = [(key, fused_score[key]) for key in cand_keys]
        timings["rank"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        if sysc.diversify and final:
            items = [info[key][1] for key, _ in final]
            sc = [s for _, s in final]
            if sysc.diversify == "mmr":
                idx = mmr(items, sc, p.k, lam=self.bc["mmr_lambda"], max_per_restaurant=self.bc["max_per_restaurant"])
            else:
                idx = coverage(items, sc, p.k)
            final = [final[j] for j in idx] + [x for j, x in enumerate(final) if j not in set(idx)]
        timings["diversify"] = (time.perf_counter() - t) * 1000

        # ---- facets over the candidate set
        facets: Dict[str, Dict[str, int]] = {"cuisine": defaultdict(int), "price_tier": defaultdict(int),
                                             "diet": defaultdict(int), "type": defaultdict(int)}
        for key in cand_keys:
            s = info[key][1]
            for c in (s.get("cuisines") or [])[:1]:
                facets["cuisine"][c] += 1
            facets["price_tier"][str(s.get("price_tier", ""))] += 1
            for d in s.get("diet") or []:
                facets["diet"][d] += 1
            facets["type"][s.get("type", "")] += 1

        results_out = []
        for key, score in final[: p.k]:
            sig, s, did = info[key]
            item = {"id": did, "score": round(score, 6), "type": s.get("type"), "name": s.get("name"),
                    "restaurant_name": s.get("restaurant_name"), "restaurant_id": s.get("restaurant_id"),
                    "chain": s.get("chain"), "cuisines": s.get("cuisines"), "canonical_dish": s.get("canonical_dish"),
                    "diet": s.get("diet"), "price": s.get("price"), "price_tier": s.get("price_tier"),
                    "rating": s.get("rating"), "distance_km": round(sig.get("distance_km", 0.0), 3),
                    "open_now": bool(sig.get("open_now", 1.0)),
                    "paths": {path: rank_of[path].get(key) for path in rank_of if rank_of[path].get(key)}}
            if p.debug:
                item["signals"] = sig
            results_out.append(item)

        timings.update({f"shard_{k}": v for k, v in shard_timings.items()})
        timings["total"] = (time.perf_counter() - t_start) * 1000
        out: Dict[str, Any] = {
            "query": p.q, "system": p.system, "results": results_out,
            "facets": {k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in facets.items()},
            "understanding": {"corrected": sq.corrected, "tokens": sq.tokens, "intent": sq.intent,
                              "filters": sq.filters.__dict__, "corrections": sq.corrections,
                              "terms": [t2.__dict__ for t2 in sq.terms], "geo": sq.geo, "meal_time": sq.meal_time},
            "partial": {"failed_shards": failed, "shards": len(self.groups)},
            "candidates": len(cand_keys), "generations": list(self.generations()),
            "timings_ms": {k: round(v, 3) for k, v in timings.items()}, "cache_hit": False,
        }
        if p.debug:
            out["shard_request"] = {k: v for k, v in req.items() if k != "paths"}
            out["path_sizes"] = {path: len(v) for path, v in ranked.items()}
        if p.return_features:
            out["training"] = {"features": FEATURES, "rows": X.tolist() if X is not None else [],
                               "ids": [info[key][2] for key in cand_keys],
                               "summaries": [info[key][1] for key in cand_keys], "structured": sq.to_dict()}
        self.metrics.observe_timings(out["timings_ms"])
        self.metrics.observe_ms("e2e", timings["total"])
        self.metrics.inc("searches")
        if failed:
            self.metrics.inc("partial_responses")
        if cache_key is not None and not failed:
            self.cache.put(cache_key, self.generations(), out)
        return out

    def _empty(self, p: SearchParams, sq: StructuredQuery, timings: Dict[str, float], t_start: float) -> Dict[str, Any]:
        timings["total"] = (time.perf_counter() - t_start) * 1000
        return {"query": p.q, "system": p.system, "results": [], "facets": {}, "candidates": 0,
                "understanding": {"corrected": sq.corrected, "tokens": sq.tokens, "intent": sq.intent},
                "partial": {"failed_shards": [], "shards": len(self.groups)},
                "timings_ms": {k: round(v, 3) for k, v in timings.items()}, "cache_hit": False}

    # --------------------------------------------------------------- writes
    async def index_docs(self, docs: Sequence[dict], refresh: bool = False) -> Dict[str, Any]:
        """Route documents to their shard (by restaurant) and write to every replica."""
        from dishcovery.embed.encoders import doc_text
        from dishcovery.hashing import shard_of

        n = len(self.groups)
        by_shard: Dict[int, List[dict]] = defaultdict(list)
        need = [d for d in docs if "embedding_b64" not in d and "embedding" not in d]
        if need and self.encoder is not None:
            embs = self.encoder.encode([doc_text(d) for d in need])
            for d, e in zip(need, embs):
                d["embedding_b64"] = base64.b64encode(np.asarray(e, dtype="<f4").tobytes()).decode()
        for d in docs:
            by_shard[shard_of(d.get("restaurant_id", d["id"]), n)].append(d)
        path = "/index?refresh=true" if refresh else "/index"
        acks: Dict[int, Any] = {}
        for i, ds in by_shard.items():
            acks[i] = await self.groups[i].broadcast(path, ds)
        if refresh:
            self.cache.clear()  # explicit invalidation; generation vectors moved as well
        return {"routed": {str(i): len(ds) for i, ds in by_shard.items()},
                "acks": {str(i): [name for name, r in a if r is not None] for i, a in acks.items()},
                "generations": list(self.generations())}

    async def delete_docs(self, ids: Sequence[str]) -> Dict[str, Any]:
        out = await asyncio.gather(*(g.broadcast("/delete", {"ids": list(ids)}) for g in self.groups))
        deleted = sum(max((r or {}).get("deleted", 0) for _, r in acks) for acks in out)
        return {"deleted": deleted, "generations": list(self.generations())}

    def describe(self) -> Dict[str, Any]:
        return {"shards": [g.describe() for g in self.groups], "cache": self.cache.stats(),
                "metrics": self.metrics.summary(), "ltr": getattr(self.ltr, "backend", None),
                "encoder": getattr(self.encoder, "name", None)}
