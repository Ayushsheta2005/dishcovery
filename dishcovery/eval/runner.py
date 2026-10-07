"""Run every system over the evaluation queries and score the runs.

Systems are broker configurations (see serving/broker.py::SYSTEMS) plus the
Elasticsearch baselines (``es_oob``: raw query text, default analyzers;
``es_qu``: the same QU output our engine gets). All systems apply the same
delivery-zone filter so the comparison is about ranking, not coverage.

NDCG is computed on hand labels (data/eval/hand_labels.jsonl). If they do not
exist yet, ``--labels weak`` evaluates against weak labels over the pooled
results and the report says so in its provenance line.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Dict, List, Optional, Sequence

from dishcovery.config import Config
from dishcovery.eval.io import load_qrels, write_run
from dishcovery.eval.report import evaluate_and_report
from dishcovery.labels.queries import EvalQuery, load_queries, qid_to_class

Run = Dict[str, List[str]]


async def run_broker_systems(cfg: Config, systems: Sequence[str], queries: Sequence[EvalQuery], depth: int,
                             concurrency: int = 8) -> Dict[str, Run]:
    from dishcovery.serving.broker import Broker, SearchParams
    from dishcovery.serving.gateway import load_components

    qu, enc, ltr, beh = load_components(cfg)
    runs: Dict[str, Run] = {}
    sem = asyncio.Semaphore(concurrency)
    async with Broker(cfg, qu, enc, ltr, beh) as b:
        for sysname in systems:
            if sysname in ("ltr", "full") and ltr is None:
                print(f"skipping {sysname}: no trained model at {cfg.path('ltr', 'model.pkl')}")
                continue
            run: Run = {}

            async def one(q: EvalQuery):
                async with sem:
                    r = await b.search(SearchParams(q=q.text, lat=q.lat, lon=q.lon, when=q.time, k=depth,
                                                    system=sysname, use_cache=False))
                    run[q.qid] = [x["id"] for x in r["results"]]
            await asyncio.gather(*(one(q) for q in queries))
            runs[sysname] = run
    return runs


def run_es_systems(cfg: Config, systems: Sequence[str], queries: Sequence[EvalQuery], depth: int) -> Dict[str, Run]:
    from dishcovery.baselines.es import ElasticBaseline

    es = ElasticBaseline(cfg["es"]["url"], cfg["es"]["index"])
    qu = None
    if "es_qu" in systems:
        from dishcovery.qu.pipeline import QueryUnderstanding
        qu = QueryUnderstanding.load(cfg.qu_dir)
    runs: Dict[str, Run] = {}
    for s in systems:
        runs[s] = {}
        for q in queries:
            sq = qu.parse(q.text) if (s == "es_qu" and qu) else None
            runs[s][q.qid] = es.search(q.text, q.lat, q.lon, k=depth, structured=sq, when=q.time)
    return runs


def weak_qrels_for(cfg: Config, queries: Sequence[EvalQuery], runs: Dict[str, Run], depth: int = 20):
    from dishcovery.data.docstore import DocStore
    from dishcovery.labels.weak import WeakLabeler

    docs = DocStore(cfg.docs_path)
    wl = WeakLabeler(docs)
    qrels: Dict[str, Dict[str, int]] = {}
    for q in queries:
        pool = {d for run in runs.values() for d in run.get(q.qid, [])[:depth]}
        qrels[q.qid] = {d: wl.grade(q.text, docs[d], ctx=q) for d in pool if d in docs}
    return qrels


def evaluate_systems(cfg: Config, systems: Optional[Sequence[str]] = None, labels: str = "hand",
                     out_dir: Optional[str] = None) -> dict:
    systems = list(systems or cfg["eval"]["systems"])
    queries = load_queries(cfg["eval"]["queries"])
    depth = cfg["eval"]["depth"]
    broker_systems = [s for s in systems if not s.startswith("es_")]
    es_systems = [s for s in systems if s.startswith("es_")]
    runs = asyncio.run(run_broker_systems(cfg, broker_systems, queries, depth)) if broker_systems else {}
    if es_systems:
        runs.update(run_es_systems(cfg, es_systems, queries, depth))
    out_dir = out_dir or cfg.path("eval")
    os.makedirs(out_dir, exist_ok=True)
    runs_path = os.path.join(out_dir, "runs.jsonl")
    if os.path.exists(runs_path):
        os.remove(runs_path)
    for s, run in runs.items():
        write_run(runs_path, run, s, append=True)
    if labels == "hand":
        qrels_path = cfg["eval"]["qrels"]
        if not os.path.exists(qrels_path):
            raise FileNotFoundError(
                f"{qrels_path} not found. Label the pool first (dishcovery labels pool / hand), "
                "or run with --labels weak for a clearly-marked weak-label evaluation.")
        qrels = load_qrels(qrels_path, sources=("hand",))
        source = "hand"
    else:
        qrels = weak_qrels_for(cfg, queries, runs)
        source = "weak"
    baseline = cfg["eval"]["baseline"] if cfg["eval"]["baseline"] in runs else next(iter(runs))
    report = evaluate_and_report(runs, qrels, baseline=baseline,
                                 metrics=("ndcg@10", "mrr", "recall@100", "judged@10"),
                                 qid_to_class=qid_to_class(queries), label_source=source,
                                 md_path=os.path.join(out_dir, f"report_{source}.md"),
                                 json_path=os.path.join(out_dir, f"report_{source}.json"),
                                 title=f"Dishcovery retrieval quality ({source} labels)")
    with open(os.path.join(out_dir, "runs_summary.json"), "w") as f:
        json.dump({s: {"queries": len(r)} for s, r in runs.items()}, f, indent=2)
    return report
