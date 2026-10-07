"""One-call data preparation used by the CLI and scripts."""
from __future__ import annotations

import os
import shutil
import time
from typing import Optional

from dishcovery.data.build_docs import BUSINESS_FILE, build_docs, scale_docs, subsample_docs
from dishcovery.data.synth_yelp import generate_synthetic_yelp


def prepare(source: str = "synthetic", raw_dir: str = "data/raw/yelp", out_dir: str = "data/corpus",
            n_synthetic: int = 2000, osm_path: Optional[str] = None, max_businesses: Optional[int] = None,
            target_docs: Optional[int] = None, seed: int = 0, workers: Optional[int] = None) -> dict:
    """Write ``out_dir/docs.jsonl`` (+ catalog_stats.json) and return stats.

    source="synthetic" generates a Yelp-format dataset into raw_dir first;
    source="yelp" expects the real Yelp Open Dataset files in raw_dir.
    target_docs scales the corpus up (clones) or down (whole-restaurant subsample).
    """
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    stats = {"source": source}
    if source == "synthetic":
        if not os.path.exists(os.path.join(raw_dir, BUSINESS_FILE)):
            stats["synthetic"] = generate_synthetic_yelp(raw_dir, n_synthetic, seed=seed)
    elif source == "yelp":
        if not os.path.exists(os.path.join(raw_dir, BUSINESS_FILE)):
            raise FileNotFoundError(
                f"{raw_dir}/{BUSINESS_FILE} not found. Download the Yelp Open Dataset "
                "(https://www.yelp.com/dataset) and extract the JSON files into that directory.")
    else:
        raise ValueError(f"unknown source {source!r}")
    base = os.path.join(out_dir, "docs.base.jsonl")
    stats["build"] = build_docs(raw_dir, base, osm_path=osm_path, max_businesses=max_businesses, seed=seed,
                                workers=workers)
    final = os.path.join(out_dir, "docs.jsonl")
    n = stats["build"]["docs"]
    if target_docs and target_docs > n:
        stats["scale"] = scale_docs(base, final, target_docs, seed=seed)
    elif target_docs and target_docs < n:
        stats["subsample"] = subsample_docs(base, final, target_docs, seed=seed)
    else:
        shutil.copyfile(base, final)
    stats["seconds"] = round(time.time() - t0, 2)
    stats["docs_path"] = final
    stats["bytes"] = os.path.getsize(final)
    return stats
