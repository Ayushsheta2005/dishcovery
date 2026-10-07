"""Compile work_dir/results/*.json (+ eval reports) into one RESULTS.md, and plots."""
from __future__ import annotations

import json
import os
from typing import Any, List, Optional

from dishcovery.config import Config


def _load(cfg: Config, name: str) -> Optional[Any]:
    p = cfg.path("results", f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    def fmt(x):
        if isinstance(x, float):
            return f"{x:.3f}" if abs(x) < 100 else f"{x:.1f}"
        return str(x)
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def write_results_md(cfg: Config) -> str:
    parts = [f"# Results — `{cfg['name']}`\n",
             f"Docs: {sum(1 for _ in open(cfg.docs_path)) if os.path.exists(cfg.docs_path) else 'n/a'} · "
             f"shards {cfg['index']['shards']} × replicas {cfg['index']['replicas']} · "
             f"embeddings `{cfg['embedding']['model']}`\n"]
    if cfg["embedding"]["model"] == "hashing":
        parts.append("> **Note:** hashing embeddings (no semantics) — vector/hybrid quality numbers from this run "
                     "are not meaningful; latency numbers are.\n")
    for src in ("hand", "weak"):
        p = cfg.path("eval", f"report_{src}.md")
        if os.path.exists(p):
            parts.append(f"## Retrieval quality ({src} labels)\n")
            if src == "weak":
                parts.append("> Weak labels are heuristic (review/category/constraint matching). Final claims must "
                             "use hand labels.\n")
            parts.append(open(p).read())
    ltr = cfg.path("ltr", "metrics.json")
    if os.path.exists(ltr):
        m = json.load(open(ltr))
        imp = list(m.get("importance", {}).items())[:12]
        parts.append("## LambdaMART\n")
        parts.append(f"backend `{m['backend']}`, labels `{m['label_source']}`, {m['train_queries']} train / "
                     f"{m['valid_queries']} valid queries; validation NDCG@10 {m['valid_ndcg10_ltr']:.4f} vs "
                     f"RRF order {m['valid_ndcg10_rrf']:.4f} ({m['note']}).\n")
        parts.append(_table(["feature", "gain share"], [[k, v] for k, v in imp]) + "\n")
    # Index size vs corpus size, build throughput (from the indexer's build reports).
    builds = []
    for s_ in range(cfg["index"]["shards"]):
        bp = os.path.join(cfg.shard_dir(s_, 0), "build_report.json")
        if os.path.exists(bp):
            builds.append(json.load(open(bp)))
    if builds:
        corpus_bytes = os.path.getsize(cfg.docs_path) if os.path.exists(cfg.docs_path) else 0
        vec_bytes = os.path.getsize(cfg.vectors_path) if os.path.exists(cfg.vectors_path) else 0
        total_idx = sum(b["index_bytes"] for b in builds)
        parts.append("## Index build\n")
        parts.append(_table(["shard", "docs", "index MiB", "build s", "merge s", "docs/s", "segments", "codec"],
                            [[b["shard"], b["indexed"], b["index_bytes"] / 2**20, b["build_seconds"],
                              b["merge_seconds"], b["docs_per_second"], b["segments"], b["codec"]] for b in builds]))
        parts.append(f"\nTotal index {total_idx / 2**20:.1f} MiB for a {corpus_bytes / 2**20:.1f} MiB docs.jsonl "
                     f"(+ {vec_bytes / 2**20:.1f} MiB of float32 embeddings; vectors + HNSW dominate the index).\n")
    f = _load(cfg, "filter")
    if f:
        rows = [[m, v["bm25_p50_us"], v["bm25_p99_us"], v["mean_scored_docs"], v["mean_admissible_docs"]]
                for m, v in f.items() if isinstance(v, dict)]
        parts.append("## Post-filter vs filter-aware retrieval (BM25 path, one shard)\n")
        parts.append(_table(["mode", "p50 µs", "p99 µs", "docs scored", "docs admissible"], rows))
        parts.append(f"\npost_filter_topk (k×10 over-fetch) recall@100 vs exact: "
                     f"{f.get('post_filter_topk_recall_at_100')}\n")
    c = _load(cfg, "codecs")
    if c:
        rows = [[r["codec"], r["bits_per_posting"], r.get("decode_mints_s", 0)] for r in c if r["bench"] == "codec"]
        parts.append("## Posting compression (real postings of shard 0)\n")
        parts.append(_table(["codec", "bits/posting", "decode M ints/s"], rows))
        inter = [r for r in c if r["bench"] == "intersection"]
        if inter:
            i = inter[0]
            parts.append("\n" + _table(["AND strategy", "µs / pair"],
                                       [["decode all + linear merge", i["decode_all_linear_us"]],
                                        ["skip-list advance()", i["skiplist_advance_us"]],
                                        ["Roaring AND", i["roaring_and_us"]]]) + "\n")
    mg = _load(cfg, "merge")
    if mg:
        rows = [[k, v["summary"]["write_amplification"], v["summary"]["segments"], v["summary"]["merges"],
                 v["summary"]["query_p50_us"], v["summary"]["query_p99_us"]] for k, v in mg.items()]
        parts.append("## Merge policy\n")
        parts.append(_table(["policy", "write amp", "final segments", "merges", "query p50 µs", "query p99 µs"], rows) + "\n")
    h = _load(cfg, "hnsw")
    if h:
        parts.append(f"## HNSW (k={h['k']}, exact scan p50 {h['exact_p50_us']} µs)\n")
        parts.append(_table(["ef", "recall@100", "p50 µs", "p99 µs"],
                            [[r["ef"], r["recall_at_100"], r["p50_us"], r["p99_us"]] for r in h["ef"]]) + "\n")
    pt = _load(cfg, "partition")
    if pt:
        rows = []
        for n, v in pt.items():
            for kind in ("document_partitioned", "term_partitioned"):
                rows.append([n, kind, v[kind]["nodes_per_query"], v[kind]["bytes_per_query"],
                             v[kind]["load_imbalance_max_over_mean"]])
        parts.append("## Document- vs term-partitioning (model)\n")
        parts.append(_table(["nodes", "scheme", "nodes/query", "bytes/query", "load max/mean"], rows) + "\n")
    nrt = _load(cfg, "nrt")
    if nrt:
        parts.append(f"## Near-real-time indexing\nvisible on all replicas: p50 {nrt['visible_ms']['p50']:.0f} ms, "
                     f"p95 {nrt['visible_ms']['p95']:.0f} ms, max {nrt['visible_ms']['max']:.0f} ms; "
                     f"{100 * nrt['under_1s']:.0f}% under 1 s (refresh {nrt['refresh_ms']} ms)\n")
    for name in sorted(os.listdir(cfg.path("results"))) if os.path.exists(cfg.path("results")) else []:
        if name.startswith("load_") and name.endswith(".json"):
            L = json.load(open(cfg.path("results", name)))
            stage_names = ["qu", "embed", "phase1", "phase2", "rank", "shard_bm25", "shard_vector", "shard_filter"]
            rows = []
            for r in L["runs"]:
                st = r["stages"]
                rows.append([r["target_qps"], round(r["achieved_qps"], 1), r["e2e"]["p50_ms"], r["e2e"]["p95_ms"],
                             r["e2e"]["p99_ms"], r["errors"] + r["dropped"]] +
                            [st.get(s, {}).get("p99_ms", 0.0) for s in stage_names])
            parts.append(f"## Latency under load — system `{L['system']}`, {L['docs']} docs\n")
            parts.append(_table(["QPS", "achieved", "p50 ms", "p95 ms", "p99 ms", "err+drop"] +
                                [f"{s} p99" for s in stage_names], rows) + "\n")
    hd = _load(cfg, "hedging")
    if hd:
        parts.append("## Hedged requests (slow replica injected: +60 ms on 10% of r0 requests)\n")
        parts.append(_table(["", "p50 ms", "p95 ms", "p99 ms"],
                            [[k, hd[k]["e2e"]["p50_ms"], hd[k]["e2e"]["p95_ms"], hd[k]["e2e"]["p99_ms"]]
                             for k in ("hedge_off", "hedge_on")]) + "\n")
    ca = _load(cfg, "cache")
    if ca:
        parts.append("## Result cache\n")
        parts.append(_table(["", "p50 ms", "p99 ms", "hit rate"],
                            [[k, ca[k]["e2e"]["p50_ms"], ca[k]["e2e"]["p99_ms"],
                              ca[k]["cache_hits"] / max(ca[k]["ok"], 1)] for k in ("cache_off", "cache_on")]) + "\n")
    es = _load(cfg, "es_latency")
    if es:
        parts.append(f"## Elasticsearch latency (single client)\nES took p50 {es['took_p50_ms']} ms / p99 "
                     f"{es['took_p99_ms']} ms; wall p50 {es['wall_p50_ms']:.1f} ms / p99 {es['wall_p99_ms']:.1f} ms\n")
    geo = _load(cfg, "geostat")
    if geo:
        parts.append("## H3 resolution choice\n")
        parts.append(_table(["res", "edge km", "docs/cell p50", "docs/cell p99", "cells in 3 km disk",
                             "cover cells (compacted) mean"],
                            [[g["res"], g["edge_km"], g["docs_per_point_cell"].get("p50"),
                              g["docs_per_point_cell"].get("p99"), g["cells_in_3km_disk"],
                              g["cover_cells_compacted"].get("mean", 0)] for g in geo]) + "\n")
    md = "\n".join(parts)
    os.makedirs(cfg.path("results"), exist_ok=True)
    out = cfg.path("results", "RESULTS.md")
    with open(out, "w") as fh:
        fh.write(md)
    try:
        from dishcovery.bench.plots import make_plots
        make_plots(cfg)
    except Exception as e:  # plotting is optional
        print(f"plots skipped: {e}")
    return out
