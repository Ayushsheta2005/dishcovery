"""PNG plots for the results (matplotlib, Agg backend)."""
from __future__ import annotations

import json
import os

from dishcovery.config import Config


def make_plots(cfg: Config) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = cfg.path("results")
    out = []

    def save(fig, name):
        p = os.path.join(res, name)
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        out.append(p)

    for name in sorted(os.listdir(res)):
        if name.startswith("load_") and name.endswith(".json"):
            L = json.load(open(os.path.join(res, name)))
            x = [r["achieved_qps"] for r in L["runs"]]
            fig, ax = plt.subplots(figsize=(6, 4))
            for q, lab in (("p50_ms", "p50"), ("p95_ms", "p95"), ("p99_ms", "p99")):
                ax.plot(x, [r["e2e"][q] for r in L["runs"]], marker="o", label=lab)
            ax.set_xlabel("achieved QPS")
            ax.set_ylabel("end-to-end latency (ms)")
            ax.set_title(f"Latency vs load ({L['system']}, {L['docs']} docs)")
            ax.grid(alpha=0.3)
            ax.legend()
            save(fig, name.replace(".json", ".png"))
    p = os.path.join(res, "codecs.json")
    if os.path.exists(p):
        rows = [r for r in json.load(open(p)) if r["bench"] == "codec"]
        fig, ax = plt.subplots(figsize=(5, 4))
        for r in rows:
            ax.scatter(r["bits_per_posting"], r["decode_mints_s"], s=60)
            ax.annotate(r["codec"], (r["bits_per_posting"], r["decode_mints_s"]), textcoords="offset points",
                        xytext=(6, 4))
        ax.set_xlabel("bits per posting")
        ax.set_ylabel("decode throughput (M ints/s)")
        ax.set_title("Compression: size vs decode speed")
        ax.grid(alpha=0.3)
        save(fig, "codecs.png")
    p = os.path.join(res, "hnsw.json")
    if os.path.exists(p):
        h = json.load(open(p))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([r["p50_us"] for r in h["ef"]], [r["recall_at_100"] for r in h["ef"]], marker="o")
        for r in h["ef"]:
            ax.annotate(f"ef={r['ef']}", (r["p50_us"], r["recall_at_100"]), textcoords="offset points", xytext=(4, -10))
        ax.axvline(h["exact_p50_us"], ls="--", c="gray", label="exact scan p50")
        ax.set_xlabel("p50 latency (µs)")
        ax.set_ylabel("recall@100")
        ax.legend()
        ax.grid(alpha=0.3)
        save(fig, "hnsw.png")
    p = os.path.join(res, "merge.json")
    if os.path.exists(p):
        m = json.load(open(p))
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, v in m.items():
            ax.plot([t["docs"] for t in v["timeline"]], [t["segments"] for t in v["timeline"]], label=name)
        ax.set_xlabel("documents ingested")
        ax.set_ylabel("segments")
        ax.set_title("Segment count under ingest")
        ax.legend()
        ax.grid(alpha=0.3)
        save(fig, "merge_segments.png")
    return out
