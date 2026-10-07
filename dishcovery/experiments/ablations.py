"""Run the ablation suite. Every step is independent and writes results/<name>.json.

    quality    systems x query classes (hand labels if present, else weak — and labelled as such)
                 BM25 -> +QU -> +hybrid (RRF) -> +LambdaMART -> +MMR, Elasticsearch baselines,
                 QU stage ablations (no spell/segment, no synonym expansion)
    filter     post-filter vs filter-aware retrieval: latency and docs scored
    codecs     raw / vbyte / pfor / roaring: size vs decode speed; skip lists vs linear AND
    merge      tiered vs leveled vs no merging: write amplification vs query latency
    hnsw       ef sweep: recall@100 vs latency against exact scan
    partition  document- vs term-partitioning model on this corpus
    nrt        indexing-to-searchable latency
    load       p50/p95/p99 per stage across QPS
    cache      result cache on/off
    hedging    hedged requests on/off with an injected slow replica
"""
from __future__ import annotations

import json
import os
import traceback
import urllib.request
from typing import Callable, Dict, List, Optional

from dishcovery.config import Config

QUALITY_SYSTEMS = ["exact", "bm25", "bm25_qu", "vector", "bm25_vector", "hybrid", "hybrid_nospell",
                   "hybrid_noexpand", "ltr", "full"]


def _es_up(cfg: Config) -> bool:
    try:
        with urllib.request.urlopen(cfg["es"]["url"], timeout=2):
            return True
    except Exception:
        return False


def _quality(cfg: Config) -> dict:
    from dishcovery.eval.runner import evaluate_systems

    systems = list(QUALITY_SYSTEMS)
    if _es_up(cfg):
        systems += ["es_oob", "es_qu"]
    labels = "hand" if os.path.exists(cfg["eval"]["qrels"]) else "weak"
    return evaluate_systems(cfg, systems, labels=labels)


def run_all(cfg: Config, only: Optional[List[str]] = None) -> Dict[str, str]:
    from dishcovery.bench import runners as B
    from dishcovery.pipeline import step_geostat

    steps: Dict[str, Callable[[], object]] = {
        "quality": lambda: _quality(cfg),
        "filter": lambda: B.bench_filter(cfg),
        "codecs": lambda: B.bench_codecs(cfg),
        "merge": lambda: B.bench_merge(cfg),
        "hnsw": lambda: B.bench_hnsw(cfg),
        "partition": lambda: B.bench_partition(cfg),
        "geostat": lambda: step_geostat(cfg),
        "nrt": lambda: B.bench_nrt(cfg),
        "load": lambda: B.bench_load(cfg),
        "cache": lambda: B.bench_cache(cfg),
        "hedging": lambda: B.bench_hedging(cfg) if cfg["index"]["replicas"] >= 2 else None,
    }
    if _es_up(cfg):
        steps["es_latency"] = lambda: B.bench_es_latency(cfg)
    status: Dict[str, str] = {}
    for name, fn in steps.items():
        if only and name not in only:
            continue
        print(f"\n=== ablation: {name}", flush=True)
        try:
            fn()
            status[name] = "ok"
        except Exception as e:  # keep going: one missing prerequisite must not kill the suite
            traceback.print_exc()
            status[name] = f"failed: {e}"
    os.makedirs(cfg.path("results"), exist_ok=True)
    with open(cfg.path("results", "ablations_status.json"), "w") as f:
        json.dump(status, f, indent=2)
    from dishcovery.experiments.report import write_results_md
    write_results_md(cfg)
    print(json.dumps(status, indent=2))
    return status
