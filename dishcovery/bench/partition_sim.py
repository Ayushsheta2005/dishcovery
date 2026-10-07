"""Document- vs term-partitioning, quantified on this corpus and query log.

Model (per query, S nodes, top-k results):
  document-partitioned  every node evaluates every query term over its 1/S of
                        the corpus; the broker receives S top-k lists.
                        work/node = Σ_t df_t / S; network = S · k · 12 B.
  term-partitioned      each term's full postings live on one node (hash of
                        term); the coordinator must pull all but the rarest
                        list to intersect/score; nodes owning hot terms are hot.
                        network = Σ_{t≠rarest} df_t · bytes_per_posting.

Also reports per-node load imbalance (max/mean total work over the query log),
which is the term-partitioned killer for skewed (Zipf) food queries.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

from dishcovery.hashing import fnv1a64
from dishcovery.qu.text import tokenize

_STOP = {"and", "the", "of", "with", "a", "in", "near", "me", "open", "now", "cheap", "best"}


def term_df(docs_path: str, fields: Sequence[str] = ("name", "restaurant_name", "cuisines", "tags", "description"),
            limit: int = 0) -> Counter:
    df: Counter = Counter()
    with open(docs_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            d = json.loads(line)
            toks = set()
            for fld in fields:
                v = d.get(fld)
                toks.update(tokenize(" ".join(v) if isinstance(v, list) else str(v or "")))
            df.update(toks)
    return df


def simulate(df: Counter, queries: Sequence[str], nodes: int = 8, k: int = 100, bytes_per_posting: float = 1.5) -> dict:
    out = {}
    q_terms = [[t for t in tokenize(q) if t not in _STOP and df.get(t)] for q in queries]
    q_terms = [t for t in q_terms if t]
    # Document partitioning.
    load_doc = [0.0] * nodes
    net_doc = 0.0
    for ts in q_terms:
        w = sum(df[t] for t in ts) / nodes
        for n in range(nodes):
            load_doc[n] += w
        net_doc += nodes * k * 12
    # Term partitioning.
    load_term = [0.0] * nodes
    net_term = 0.0
    contacted = 0
    for ts in q_terms:
        owners = defaultdict(float)
        for t in ts:
            owners[fnv1a64(t) % nodes] += df[t]
        for n, w in owners.items():
            load_term[n] += w
        contacted += len(owners)
        dfs = sorted(df[t] for t in ts)
        net_term += sum(dfs[1:]) * bytes_per_posting + k * 12
    nq = max(len(q_terms), 1)

    def imbalance(load):
        m = sum(load) / len(load)
        return max(load) / m if m else 0.0
    out["queries"] = nq
    out["nodes"] = nodes
    out["document_partitioned"] = {"nodes_per_query": nodes, "bytes_per_query": net_doc / nq,
                                   "load_imbalance_max_over_mean": imbalance(load_doc)}
    out["term_partitioned"] = {"nodes_per_query": contacted / nq, "bytes_per_query": net_term / nq,
                               "load_imbalance_max_over_mean": imbalance(load_term)}
    hot = sorted(((fnv1a64(t) % nodes, t, c) for t, c in df.most_common(10)), key=lambda x: -x[2])
    out["hottest_terms"] = [{"term": t, "df": c, "node": n} for n, t, c in hot]
    return out
