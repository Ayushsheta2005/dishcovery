"""Experiment configuration (YAML). See configs/*.yaml.

Paths are resolved relative to the repository root unless absolute. A config
may ``extends:`` another file; nested dicts are merged.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS: Dict[str, Any] = {
    "name": "default",
    "seed": 0,
    "work_dir": "work/default",
    "data": {"source": "synthetic", "n_synthetic": 2000, "raw_dir": "data/raw/yelp", "osm_path": None,
             "max_businesses": None, "target_docs": None, "workers": None},
    "embedding": {"model": "hashing", "dim": 384, "batch_size": 512, "device": None},
    "index": {"schema": "configs/schema.json", "shards": 2, "replicas": 1, "segment_docs": 250000,
              "max_segments": 4, "codec": "pfor", "threads": None, "hnsw_m": 16, "hnsw_efc": 200,
              "merge_policy": {"type": "tiered"}},
    "serving": {"host": "127.0.0.1", "shard_base_port": 7101, "suggest_port": 7099, "gateway_port": 8080,
                "shard_threads": 64, "filter_cache_mb": 256, "postings_cache_mb": 256, "read_only": False,
                "refresh_ms": 200, "flush_ms": 5000, "gateway_workers": 1},
    "broker": {"bm25_k": 500, "vector_k": 200, "geo_k": 200, "exact_k": 100, "ef": 128, "rrf_k": 60.0,
               "rerank_depth": 100, "facet_depth": 100, "timeout_s": 2.0, "hedge": True, "hedge_budget": 0.05,
               "cache_capacity": 10000, "cache_ttl_s": 300, "filter_mode": "filter_aware",
               "poll_generations_s": 1.0, "conns_per_shard": 16, "geo_filter": True, "mmr_lambda": 0.75, "max_per_restaurant": 2},
    "ranking": {"backend": "auto", "train_queries": 2000, "label_source": "weak", "params": {}},
    "labels": {"history_sessions": 30000, "label_sessions": 60000, "eta": 1.0, "noise_shuffle": 0.1},
    "eval": {"queries": "data/eval/queries.jsonl", "qrels": "data/eval/hand_labels.jsonl",
             "systems": ["exact", "bm25", "bm25_qu", "hybrid", "ltr", "full"], "baseline": "bm25", "depth": 100},
    "es": {"url": "http://localhost:9200", "index": "dishcovery"},
    "loadtest": {"qps": [50, 100, 200, 400], "duration_s": 60, "concurrency_cap": 512, "system": "full"},
}


def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve(path: Optional[str]) -> Optional[str]:
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


class Config(dict):
    """dict with attribute-style section access and derived paths."""

    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    # derived locations under work_dir
    def path(self, *parts: str) -> str:
        return os.path.join(resolve(self["work_dir"]), *parts)

    @property
    def corpus_dir(self) -> str:
        return self.path("corpus")

    @property
    def docs_path(self) -> str:
        return self.path("corpus", "docs.jsonl")

    @property
    def vectors_path(self) -> str:
        return self.path("corpus", "embeddings.f32")

    @property
    def index_dir(self) -> str:
        return self.path("index")

    def shard_dir(self, shard: int, replica: int = 0) -> str:
        # Replicas of a built (read-only) index share files; writable replicas get their own copy.
        if self["serving"].get("read_only") or replica == 0:
            return os.path.join(self.index_dir, f"shard_{shard}")
        return os.path.join(self.index_dir, f"shard_{shard}_r{replica}")

    @property
    def qu_dir(self) -> str:
        return self.path("qu")

    @property
    def schema_path(self) -> str:
        return resolve(self["index"]["schema"])

    def shard_urls(self) -> List[List[str]]:
        s = self["serving"]
        out = []
        for shard in range(self["index"]["shards"]):
            out.append([f"http://{s['host']}:{s['shard_base_port'] + shard * 10 + r}"
                        for r in range(self["index"]["replicas"])])
        return out

    @property
    def suggest_url(self) -> str:
        s = self["serving"]
        return f"http://{s['host']}:{s['suggest_port']}"

    @property
    def gateway_url(self) -> str:
        s = self["serving"]
        return f"http://{s['host']}:{s['gateway_port']}"


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    cfg: Dict[str, Any] = copy.deepcopy(DEFAULTS)
    if path:
        p = resolve(path) if not os.path.exists(path) else path
        raw = yaml.safe_load(open(p)) or {}
        if "extends" in raw:
            base = load_config(os.path.join(os.path.dirname(p), raw.pop("extends")))
            cfg = _merge(dict(base), raw)
        else:
            cfg = _merge(cfg, raw)
    if overrides:
        cfg = _merge(cfg, overrides)
    cfg["_config_path"] = os.path.abspath(path) if path else ""
    return Config(cfg)
