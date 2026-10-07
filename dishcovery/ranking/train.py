"""LambdaMART training pipeline.

1. Training queries: a synthetic query log over the corpus (eval queries excluded).
2. Candidates: the hybrid retriever's top-``depth`` per query, with features,
   collected through the real broker (so training and serving features match).
3. Labels (``ranking.label_source``):
     weak    heuristic grades from review text / category / constraint matching
     clicks  inverse-propensity-weighted clicks from PBM-simulated sessions
             (latent relevance = the weak grade; position bias eta)
4. Behavioural features come from a *separate* simulated history window.
5. Query-level 80/20 split; early stopping on validation NDCG@10.

None of this touches the hand-labelled evaluation set.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dishcovery.config import Config
from dishcovery.data.docstore import DocStore
from dishcovery.labels.clicks import ipw_click_labels, save_impressions, simulate
from dishcovery.labels.querylog import TrainQuery, generate_query_log, load_query_log, save_query_log
from dishcovery.labels.weak import WeakLabeler
from dishcovery.ranking.behavior import BehaviorStore
from dishcovery.ranking.features import FEATURE_INDEX, FEATURES
from dishcovery.ranking.ltr import LTRModel, mean_ndcg


async def collect_candidates(broker, queries: Sequence[TrainQuery], depth: int, concurrency: int = 16,
                             log=print) -> List[dict]:
    from dishcovery.serving.broker import SearchParams

    sem = asyncio.Semaphore(concurrency)
    out: List[Optional[dict]] = [None] * len(queries)
    t0 = time.time()

    async def one(i: int, q: TrainQuery):
        async with sem:
            r = await broker.search(SearchParams(q=q.text, lat=q.lat, lon=q.lon, when=q.time, k=depth,
                                                 system="hybrid", return_features=True, depth=depth,
                                                 use_cache=False))
            tr = r.get("training") or {}
            out[i] = {"qid": q.qid, "text": q.text, "rows": tr.get("rows", []), "ids": tr.get("ids", [])}
            if (i + 1) % 200 == 0:
                log(f"collected {i + 1}/{len(queries)} queries ({(i + 1) / (time.time() - t0):.1f} q/s)")

    await asyncio.gather(*(one(i, q) for i, q in enumerate(queries)))
    return [c for c in out if c and c["ids"]]


def weak_grades(cands: Sequence[dict], queries: Dict[str, TrainQuery], docs: DocStore) -> Dict[Tuple[str, str], int]:
    wl = WeakLabeler(docs)
    grades = {}
    for c in cands:
        q = queries[c["qid"]]
        ctx = {"lat": q.lat, "lon": q.lon, "time": q.time}
        for did in c["ids"]:
            doc = docs.get(did)
            grades[(c["qid"], did)] = wl.grade(q.text, doc, ctx=ctx, target_type=q.target_type,
                                               target=q.target) if doc else 0
    return grades


def simulate_sessions(cands: Sequence[dict], grades: Dict[Tuple[str, str], int], n_sessions: int, eta: float,
                      noise_shuffle: float, seed: int):
    order = {c["qid"]: c["ids"] for c in cands}
    qs = [{"qid": c["qid"], "text": c["text"]} for c in cands]
    return simulate(qs, ranker=lambda q: order[q["qid"]], grader=lambda q, d: grades.get((q["qid"], d), 0),
                    n_sessions=n_sessions, eta=eta, noise_shuffle=noise_shuffle, seed=seed)


def build_matrix(cands: Sequence[dict], labels: Dict[Tuple[str, str], int], behavior: Optional[BehaviorStore]
                 ) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    X, y, groups, qids = [], [], [], []
    bcols = [FEATURE_INDEX[f] for f in ("hist_ctr", "hist_cvr", "doc_ctr")]
    for c in cands:
        rows = np.asarray(c["rows"], dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(FEATURES):
            continue
        if behavior is not None and behavior.table is not None:
            for j, did in enumerate(c["ids"]):
                b = behavior.lookup(c["text"], did)
                rows[j, bcols] = [b.get("ctr_smoothed", 0.0), b.get("cvr_smoothed", 0.0), b.get("doc_ctr", 0.0)]
        lab = [labels.get((c["qid"], d), 0) for d in c["ids"]]
        if max(lab) == 0:
            continue  # no signal for lambdarank
        X.append(rows)
        y.extend(lab)
        groups.append(len(lab))
        qids.append(c["qid"])
    if not X:
        return np.zeros((0, len(FEATURES))), np.zeros(0), [], []
    return np.vstack(X), np.asarray(y, dtype=np.float64), groups, qids


def train_pipeline(cfg: Config, log=print) -> dict:
    from dishcovery.serving.broker import Broker
    from dishcovery.serving.gateway import load_components

    rc, lc = cfg["ranking"], cfg["labels"]
    os.makedirs(cfg.path("ltr"), exist_ok=True)
    os.makedirs(cfg.path("clicks"), exist_ok=True)
    qlog_path = cfg.path("labels", "train_queries.jsonl")
    os.makedirs(os.path.dirname(qlog_path), exist_ok=True)
    if not os.path.exists(qlog_path):
        qs = generate_query_log(cfg.docs_path, rc["train_queries"], seed=cfg["seed"],
                                eval_queries_path=cfg["eval"]["queries"])
        save_query_log(qlog_path, qs)
    queries = load_query_log(qlog_path)
    qmap = {q.qid: q for q in queries}
    depth = cfg["broker"]["rerank_depth"]

    cand_path = cfg.path("ltr", "candidates.jsonl")
    if os.path.exists(cand_path):
        cands = [json.loads(l) for l in open(cand_path)]
    else:
        qu, enc, _, _ = load_components(cfg)

        async def go():
            async with Broker(cfg, qu, enc, None, None) as b:
                return await collect_candidates(b, queries, depth, log=log)
        cands = asyncio.run(go())
        with open(cand_path, "w") as f:
            for c in cands:
                f.write(json.dumps(c) + "\n")
    log(f"{len(cands)} training queries with candidates")

    docs = DocStore(cfg.docs_path)
    grades = weak_grades(cands, qmap, docs)
    # Behavioural history (feature source) and label sessions are disjoint simulations.
    hist = simulate_sessions(cands, grades, lc["history_sessions"], lc["eta"], lc["noise_shuffle"], cfg["seed"] + 1)
    save_impressions(cfg.path("clicks", "history.jsonl"), hist)
    behavior = BehaviorStore.from_impressions_file(cfg.path("clicks", "history.jsonl"))
    if rc["label_source"] == "clicks":
        sess = simulate_sessions(cands, grades, lc["label_sessions"], lc["eta"], lc["noise_shuffle"], cfg["seed"] + 2)
        save_impressions(cfg.path("clicks", "label_sessions.jsonl"), sess)
        labels = {k: int(v["grade"]) for k, v in ipw_click_labels(sess, eta=lc["eta"]).items()}
    else:
        labels = grades

    rng = random.Random(cfg["seed"])
    qids = sorted({c["qid"] for c in cands})
    rng.shuffle(qids)
    val_q = set(qids[: max(1, len(qids) // 5)])
    tr = [c for c in cands if c["qid"] not in val_q]
    va = [c for c in cands if c["qid"] in val_q]
    Xtr, ytr, gtr, _ = build_matrix(tr, labels, behavior)
    Xva, yva, gva, _ = build_matrix(va, labels, behavior)
    log(f"train: {len(gtr)} queries / {len(ytr)} rows; valid: {len(gva)} queries / {len(yva)} rows")
    model = LTRModel(rc["backend"], rc.get("params") or {})
    t0 = time.time()
    model.fit(Xtr, ytr, gtr, Xva, yva, gva, log=log)
    secs = time.time() - t0
    # Baseline on the same validation set: the hybrid (RRF) order.
    rrf_col = FEATURE_INDEX["rrf_score"]
    metrics = {"backend": model.backend, "label_source": rc["label_source"], "train_queries": len(gtr),
               "valid_queries": len(gva), "train_seconds": round(secs, 2),
               "valid_ndcg10_ltr": mean_ndcg(model.predict(Xva), yva, gva) if len(gva) else None,
               "valid_ndcg10_rrf": mean_ndcg(Xva[:, rrf_col], yva, gva) if len(gva) else None,
               "importance": model.importance(),
               "note": "validation uses training-distribution labels (weak/click), not hand labels"}
    model.meta = metrics
    model.save(cfg.path("ltr", "model.pkl"))
    with open(cfg.path("ltr", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"LTR valid NDCG@10 {metrics['valid_ndcg10_ltr']:.4f} vs RRF {metrics['valid_ndcg10_rrf']:.4f}"
        if len(gva) else "no validation queries")
    return metrics
