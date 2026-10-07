"""Offline pipeline steps (each idempotent; outputs land under cfg.work_dir)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections import Counter
from typing import List, Optional

from dishcovery.config import Config, resolve
from dishcovery.serving.cluster import binary


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def step_prepare(cfg: Config, force: bool = False) -> dict:
    from dishcovery.data.prepare import prepare

    if os.path.exists(cfg.docs_path) and not force:
        log(f"corpus exists: {cfg.docs_path}")
        return {}
    d = cfg["data"]
    base = resolve(d.get("base_docs"))
    if base and os.path.exists(base):
        # Reuse an already-built corpus (e.g. work/yelp) and only rescale it.
        from dishcovery.data.build_docs import scale_docs, subsample_docs
        os.makedirs(cfg.corpus_dir, exist_ok=True)
        n = sum(1 for _ in open(base))
        target = d.get("target_docs") or n
        fn = scale_docs if target > n else subsample_docs
        stats = fn(base, cfg.docs_path, target, seed=cfg["seed"])
        log(f"corpus rescaled from {base}: {stats}")
        return stats
    stats = prepare(d["source"], raw_dir=resolve(d["raw_dir"]), out_dir=cfg.corpus_dir, n_synthetic=d["n_synthetic"],
                    osm_path=resolve(d.get("osm_path")), max_businesses=d.get("max_businesses"),
                    target_docs=d.get("target_docs"), seed=cfg["seed"], workers=d.get("workers"))
    log(f"corpus: {json.dumps({k: v for k, v in stats.items() if k != 'build'})}")
    return stats


def step_embed(cfg: Config, force: bool = False) -> dict:
    from dishcovery.embed.encoders import embed_docs, load_encoder

    if os.path.exists(cfg.vectors_path + ".json") and not force:
        meta = json.load(open(cfg.vectors_path + ".json"))
        n = sum(1 for _ in open(cfg.docs_path))
        if meta.get("rows") == n:
            log(f"embeddings exist: {cfg.vectors_path}")
            return meta
    e = cfg["embedding"]
    enc = load_encoder(e["model"], e["dim"], e.get("device"))
    schema = json.load(open(cfg.schema_path))
    if schema.get("vector_dim") != enc.dim:
        raise ValueError(f"schema vector_dim {schema.get('vector_dim')} != encoder dim {enc.dim}")
    if force and os.path.exists(cfg.vectors_path):
        os.remove(cfg.vectors_path)
    meta = embed_docs(cfg.docs_path, cfg.vectors_path, enc, batch_size=e["batch_size"])
    log(f"embeddings: {meta}")
    return meta


def step_index(cfg: Config, shards: Optional[List[int]] = None, force: bool = False, codec: Optional[str] = None,
               out_dir: Optional[str] = None) -> List[dict]:
    ic = cfg["index"]
    reports = []
    for s in (shards if shards is not None else range(ic["shards"])):
        out = os.path.join(out_dir, f"shard_{s}") if out_dir else cfg.shard_dir(s, 0)
        if os.path.exists(os.path.join(out, "COMMIT")):
            if not force:
                log(f"shard {s} index exists: {out}")
                reports.append(json.load(open(os.path.join(out, "build_report.json"))))
                continue
            shutil.rmtree(out)
        args = [binary("dishcovery-indexer"), "--docs", cfg.docs_path, "--schema", cfg.schema_path, "--out", out,
                "--shard", str(s), "--num-shards", str(ic["shards"]), "--segment-docs", str(ic["segment_docs"]),
                "--max-segments", str(ic["max_segments"]), "--codec", codec or ic["codec"],
                "--hnsw-m", str(ic["hnsw_m"]), "--hnsw-efc", str(ic["hnsw_efc"]),
                "--merge-policy", json.dumps(ic["merge_policy"])]
        if ic.get("threads"):
            args += ["--threads", str(ic["threads"])]
        if os.path.exists(cfg.vectors_path):
            args += ["--vectors", cfg.vectors_path]
        log(f"indexing shard {s}: {' '.join(args)}")
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"indexer failed for shard {s}:\n{r.stderr[-4000:]}")
        reports.append(json.loads(r.stdout.strip().splitlines()[-1]))
        log(f"shard {s}: {reports[-1]}")
    # Replica copies are created lazily by `cluster up` for writable replicas.
    return reports


def step_qu(cfg: Config, force: bool = False) -> None:
    from dishcovery.qu.pipeline import QueryUnderstanding

    if os.path.exists(os.path.join(cfg.qu_dir, "intent.pkl")) and not force:
        log(f"QU artifacts exist: {cfg.qu_dir}")
        return
    t0 = time.time()
    QueryUnderstanding.build(cfg.docs_path, cfg.qu_dir, seed=cfg["seed"])
    log(f"QU artifacts built in {time.time() - t0:.1f}s -> {cfg.qu_dir}")


def step_suggest(cfg: Config, force: bool = False) -> dict:
    """Autocomplete keys: dish names, canonical dishes, restaurant names, cuisines — weighted by popularity."""
    out = cfg.path("suggest", "suggest.fst")
    if os.path.exists(out) and not force:
        log(f"suggest FST exists: {out}")
        return {}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    w: Counter = Counter()
    with open(cfg.docs_path) as f:
        for line in f:
            d = json.loads(line)
            pop = 1 + int(100 * d.get("popularity", 0.0))
            w[d["name"]] += pop
            if d.get("canonical_dish"):
                w[d["canonical_dish"]] += pop
            if d["type"] == "restaurant":
                for c in d.get("cuisines", []):
                    w[c.replace("_", " ")] += 1
    tsv = cfg.path("suggest", "suggestions.tsv")
    with open(tsv, "w") as f:
        for k, v in w.items():
            if k and "\t" not in k:
                f.write(f"{k}\t{v}\n")
    r = subprocess.run([binary("dishcovery-suggest"), "build", "--input", tsv, "--out", out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    log(f"suggest FST: {rep}")
    return rep


def step_geostat(cfg: Config) -> List[dict]:
    r = subprocess.run([binary("dishcovery-geostat"), "--docs", cfg.docs_path, "--limit", "2000000"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    rows = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
    os.makedirs(cfg.path("results"), exist_ok=True)
    with open(cfg.path("results", "geostat.json"), "w") as f:
        json.dump(rows, f, indent=2)
    for row in rows:
        log(f"res {row['res']}: edge {row['edge_km']:.2f} km, docs/cell p99 {row['docs_per_point_cell'].get('p99')}, "
            f"cover cells (compacted) mean {row['cover_cells_compacted'].get('mean', 0):.1f}")
    return rows


def step_build_all(cfg: Config, force: bool = False) -> None:
    step_prepare(cfg, force)
    step_embed(cfg, force)
    step_index(cfg, force=force)
    step_qu(cfg, force)
    step_suggest(cfg, force)
