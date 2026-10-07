"""Benchmarks. Each ``bench_<name>(cfg, args)`` writes work_dir/results/<name>.json.

Cluster-backed benches (load, nrt, hnsw, filter, hedging, cache) need
``dishcovery cluster up`` first; codecs/merge/partition work on files.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import statistics
import subprocess
import time
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np

from dishcovery.config import Config
from dishcovery.serving.cluster import binary


def _save(cfg: Config, name: str, data: Any) -> Any:
    os.makedirs(cfg.path("results"), exist_ok=True)
    with open(cfg.path("results", f"{name}.json"), "w") as f:
        json.dump(data, f, indent=2, default=str)
    return data


def _queries(cfg: Config, n: int = 2000) -> List[dict]:
    """Load-test queries: the training query log (Zipf, realistic repeats) or the eval set."""
    p = cfg.path("labels", "train_queries.jsonl")
    if os.path.exists(p):
        qs = [json.loads(l) for l in open(p)][:n]
        if qs:
            return qs
    from dishcovery.labels.queries import load_queries
    return [{"text": q.text, "lat": q.lat, "lon": q.lon, "time": q.time} for q in load_queries(cfg["eval"]["queries"])]


def _http(method: str, url: str, body: Optional[dict] = None, timeout: float = 30) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ------------------------------------------------------------------ latency

def bench_load(cfg: Config, args=None) -> dict:
    from dishcovery.bench.loadgen import run_open_loop

    lt = cfg["loadtest"]
    qps_list = [float(x) for x in args.qps.split(",")] if getattr(args, "qps", None) else lt["qps"]
    duration = getattr(args, "duration", None) or lt["duration_s"]
    system = getattr(args, "system", None) or lt["system"]
    qs = _queries(cfg)
    out = []
    for qps in qps_list:
        # Result cache off: this measures the engine; `bench cache` measures the cache.
        r = asyncio.run(run_open_loop(cfg.gateway_url, qs, qps, duration, system=system,
                                      max_inflight=lt["concurrency_cap"], extra={"nocache": "1"}))
        e = r["e2e"]
        print(f"qps {qps:>6}: achieved {r['achieved_qps']:.1f}, p50 {e['p50_ms']:.2f} ms, p95 {e['p95_ms']:.2f}, "
              f"p99 {e['p99_ms']:.2f}, errors {r['errors']}, dropped {r['dropped']}", flush=True)
        out.append(r)
    stats = _http("GET", cfg.gateway_url + "/stats")
    return _save(cfg, f"load_{system}", {"system": system, "runs": out, "docs": _doc_count(cfg),
                                          "shards": cfg["index"]["shards"], "replicas": cfg["index"]["replicas"],
                                          "gateway_stats": stats})


def _doc_count(cfg: Config) -> int:
    try:
        return sum(1 for _ in open(cfg.docs_path))
    except OSError:
        return 0


def bench_cache(cfg: Config, args=None) -> dict:
    from dishcovery.bench.loadgen import run_open_loop

    qs = _queries(cfg)
    qps = float(args.qps.split(",")[0]) if getattr(args, "qps", None) else cfg["loadtest"]["qps"][0]
    dur = getattr(args, "duration", None) or cfg["loadtest"]["duration_s"]
    off = asyncio.run(run_open_loop(cfg.gateway_url, qs, qps, dur, extra={"nocache": "1"}))
    on = asyncio.run(run_open_loop(cfg.gateway_url, qs, qps, dur))
    return _save(cfg, "cache", {"qps": qps, "cache_off": off, "cache_on": on})


def bench_hedging(cfg: Config, args=None) -> dict:
    """Inject a slow replica (r0 of every shard: +delay on 10% of requests) and compare p99 with/without hedging."""
    from dishcovery.bench.loadgen import run_open_loop

    if cfg["index"]["replicas"] < 2:
        raise SystemExit("hedging needs >= 2 replicas per shard")
    qs = _queries(cfg)
    qps = float(args.qps.split(",")[0]) if getattr(args, "qps", None) else cfg["loadtest"]["qps"][0]
    dur = getattr(args, "duration", None) or cfg["loadtest"]["duration_s"]
    slow = [urls[0] for urls in cfg.shard_urls()]
    fault = {"delay_ms": 60, "delay_prob": 0.10}
    try:
        for u in slow:
            _http("POST", u + "/admin/fault", fault)
        res = {}
        for hedge in ("0", "1"):
            res["hedge_on" if hedge == "1" else "hedge_off"] = asyncio.run(
                run_open_loop(cfg.gateway_url, qs, qps, dur, extra={"hedge": hedge, "nocache": "1"}))
    finally:
        for u in slow:
            _http("POST", u + "/admin/fault", {})
    stats = _http("GET", cfg.gateway_url + "/stats")
    return _save(cfg, "hedging", {"qps": qps, "fault": fault, **res,
                                  "hedges": [s["hedges"] for s in stats["shards"]],
                                  "hedge_wins": [s["hedge_wins"] for s in stats["shards"]]})


def bench_nrt(cfg: Config, args=None, n: int = 50) -> dict:
    """Time from POST /index (no forced refresh) until the doc is visible on every replica of its shard."""
    from dishcovery.hashing import shard_of

    urls = cfg.shard_urls()
    docs = []
    with open(cfg.docs_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            docs.append(json.loads(line))
    lat = []
    for i, d in enumerate(docs):
        d = dict(d)
        d["id"] = f"{d['id']}~nrt{int(time.time() * 1000)}_{i}"
        t0 = time.perf_counter()
        _http("POST", cfg.gateway_url + "/index", [d])
        s = shard_of(d["restaurant_id"], len(urls))
        pending = set(urls[s])
        while pending and time.perf_counter() - t0 < 10:
            for u in list(pending):
                try:
                    if _http("GET", f"{u}/doc?id={urllib.request.quote(d['id'])}", timeout=2).get("found"):
                        pending.discard(u)
                except Exception:
                    pass
            if pending:
                time.sleep(0.005)
        lat.append((time.perf_counter() - t0) * 1000)
        _http("POST", cfg.gateway_url + "/delete", {"ids": [d["id"]]})
    lat.sort()
    res = {"docs": len(lat), "refresh_ms": cfg["serving"]["refresh_ms"],
           "visible_ms": {"p50": lat[len(lat) // 2], "p95": lat[int(0.95 * (len(lat) - 1))], "max": lat[-1],
                          "mean": statistics.mean(lat)},
           "under_1s": sum(1 for x in lat if x < 1000) / len(lat)}
    print(res)
    return _save(cfg, "nrt", res)


# ------------------------------------------------------------------ engine-level

def bench_codecs(cfg: Config, args=None) -> dict:
    r = subprocess.run([binary("bench_codecs"), "--index", cfg.shard_dir(0), "--schema", cfg.schema_path],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    rows = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    return _save(cfg, "codecs", rows)


def bench_merge(cfg: Config, args=None, limit: int = 0) -> dict:
    limit = limit or min(_doc_count(cfg), 300000)
    out = {}
    policies = {"tiered": {"type": "tiered", "floor_segment_bytes": 1 << 20, "segments_per_tier": 8},
                "log": {"type": "log", "merge_factor": 8, "min_merge_bytes": 1 << 20},
                "none": {"type": "none"}}
    for name, pol in policies.items():
        r = subprocess.run([binary("bench_merge"), "--docs", cfg.docs_path, "--schema", cfg.schema_path,
                            "--policy", json.dumps(pol), "--limit", str(limit), "--batch", "2000",
                            "--flush-every", "3", "--work", cfg.path("tmp", f"merge_{name}")],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-2000:])
        rows = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
        out[name] = {"timeline": [x for x in rows if x["bench"] == "merge_timeline"],
                     "summary": next(x for x in rows if x["bench"] == "merge_summary")}
        s = out[name]["summary"]
        print(f"{name}: write amp {s['write_amplification']:.2f}, segments {s['segments']}, "
              f"query p99 {s['query_p99_us']} us")
    return _save(cfg, "merge", out)


def bench_partition(cfg: Config, args=None) -> dict:
    from dishcovery.bench.partition_sim import simulate, term_df

    df = term_df(cfg.docs_path, limit=2_000_000)
    qs = [q["text"] for q in _queries(cfg, 5000)]
    return _save(cfg, "partition", {str(n): simulate(df, qs, nodes=n) for n in (4, 8, 16)})


def _shard_search(url: str, body: dict) -> dict:
    return _http("POST", url + "/search", body)


def bench_filter(cfg: Config, args=None, n: int = 200) -> dict:
    """Post-filter vs filter-aware retrieval on shard 0 (BM25 path, k=100)."""
    from dishcovery.qu.pipeline import QueryUnderstanding

    qu = QueryUnderstanding.load(cfg.qu_dir)
    url = cfg.shard_urls()[0][0]
    rng = random.Random(0)
    rows = {"filter_aware": [], "post_filter": [], "post_filter_topk": []}
    recall_topk = []
    for q in _queries(cfg, n * 3)[:n]:
        sq = qu.parse(q["text"])
        if not sq.terms:
            continue
        terms = [{"text": t.text, "weight": t.weight, "group": t.group} for t in sq.terms]
        flt: Dict[str, Any] = {"geo": {"mode": "delivery", "lat": q["lat"], "lon": q["lon"]}}
        if rng.random() < 0.5:
            flt["terms"] = {"diet": [rng.choice(["veg", "vegan", "non_veg"])]}
        body = {"terms": terms, "filter": flt, "paths": {"bm25": {"k": 100}}, "filter_cache": False,
                "postings_cache": False}
        res = {}
        for mode in rows:
            r = _shard_search(url, dict(body, filter_mode=mode))
            res[mode] = r
            rows[mode].append({"bm25_us": r["timings_us"]["bm25"], "filter_us": r["timings_us"]["filter"],
                               "scored": r["paths"]["bm25"]["scored"], "admissible": r["filter_cardinality"]})
        exact = {(h[0], h[1]) for h in res["post_filter"]["paths"]["bm25"]["hits"]}
        got = {(h[0], h[1]) for h in res["post_filter_topk"]["paths"]["bm25"]["hits"]}
        if exact:
            recall_topk.append(len(exact & got) / len(exact))

    def summ(v: List[dict]) -> dict:
        b = sorted(x["bm25_us"] for x in v)
        return {"queries": len(v), "bm25_p50_us": b[len(b) // 2], "bm25_p99_us": b[int(0.99 * (len(b) - 1))],
                "mean_scored_docs": statistics.mean(x["scored"] for x in v),
                "mean_admissible_docs": statistics.mean(x["admissible"] for x in v)}
    out = {m: summ(v) for m, v in rows.items() if v}
    out["post_filter_topk_recall_at_100"] = statistics.mean(recall_topk) if recall_topk else None
    print(json.dumps(out, indent=2))
    return _save(cfg, "filter", out)


def bench_hnsw(cfg: Config, args=None, n: int = 200) -> dict:
    """HNSW recall/latency vs exact scan over the shard's real embeddings (vector path only, no filter)."""
    from dishcovery.embed.encoders import load_encoder

    enc = load_encoder(cfg["embedding"]["model"], cfg["embedding"]["dim"], cfg["embedding"].get("device"))
    url = cfg.shard_urls()[0][0]
    qs = [q["text"] for q in _queries(cfg, n * 3)][:n]
    vecs = [base64.b64encode(np.asarray(enc.encode_query(t), dtype="<f4").tobytes()).decode() for t in qs]
    k = 100

    def run(ef: int, brute: int):
        lat, res = [], []
        for v in vecs:
            r = _shard_search(url, {"paths": {"vector": {"k": k, "ef": ef, "vector_b64": v,
                                                         "brute_force_max": brute}}})
            lat.append(r["timings_us"]["vector"])
            res.append({(h[0], h[1]) for h in r["paths"]["vector"]["hits"]})
        return lat, res
    exact_lat, exact = run(k, 1 << 30)
    out = {"k": k, "queries": len(qs), "exact_p50_us": sorted(exact_lat)[len(exact_lat) // 2], "ef": []}
    for ef in (100, 128, 200, 400, 800):
        lat, res = run(ef, 0)
        rec = statistics.mean(len(a & b) / max(len(b), 1) for a, b in zip(res, exact))
        out["ef"].append({"ef": ef, "recall_at_100": rec, "p50_us": sorted(lat)[len(lat) // 2],
                          "p99_us": sorted(lat)[int(0.99 * (len(lat) - 1))]})
        print(out["ef"][-1])
    return _save(cfg, "hnsw", out)


def bench_es_latency(cfg: Config, args=None, n: int = 500) -> dict:
    from dishcovery.baselines.es import ElasticBaseline

    es = ElasticBaseline(cfg["es"]["url"], cfg["es"]["index"])
    took, wall = [], []
    for q in _queries(cfg, n)[:n]:
        t0 = time.perf_counter()
        took.append(es.search_timed(q["text"], q.get("lat"), q.get("lon")))
        wall.append((time.perf_counter() - t0) * 1000)
    took.sort()
    wall.sort()
    out = {"queries": len(took), "took_p50_ms": took[len(took) // 2], "took_p99_ms": took[int(0.99 * (len(took) - 1))],
           "wall_p50_ms": wall[len(wall) // 2], "wall_p99_ms": wall[int(0.99 * (len(wall) - 1))],
           "note": "sequential single-client requests; compare with dishcovery `bench load` at low QPS"}
    return _save(cfg, "es_latency", out)
