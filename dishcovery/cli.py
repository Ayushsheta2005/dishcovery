"""dishcovery command line.

    dishcovery --config configs/smoke.yaml build            # data -> embeddings -> index -> QU -> suggest
    dishcovery --config CFG cluster up|down
    dishcovery --config CFG search "cheap vegan near me open now" --lat 39.95 --lon -75.16
    dishcovery --config CFG train                            # LambdaMART on weak/click labels
    dishcovery --config CFG eval --labels hand|weak
    dishcovery --config CFG bench load|nrt|partition|codecs|merge|hnsw|filter|hedging|cache
    dishcovery --config CFG ablations
    dishcovery --config CFG report

Run ``dishcovery <command> -h`` for options.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from dishcovery.config import load_config


def _cfg(args):
    return load_config(args.config)


def cmd_build(args):
    from dishcovery import pipeline as P

    cfg = _cfg(args)
    steps = args.steps.split(",") if args.steps else ["prepare", "embed", "index", "qu", "suggest"]
    for s in steps:
        {"prepare": lambda: P.step_prepare(cfg, args.force), "embed": lambda: P.step_embed(cfg, args.force),
         "index": lambda: P.step_index(cfg, force=args.force), "qu": lambda: P.step_qu(cfg, args.force),
         "suggest": lambda: P.step_suggest(cfg, args.force), "geostat": lambda: P.step_geostat(cfg)}[s]()


def cmd_data(args):
    cfg = _cfg(args)
    if args.action == "osm":
        from dishcovery.data.osm import fetch_osm_pois
        out = args.out or "data/raw/osm/pois.jsonl"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        print(fetch_osm_pois(out), "POIs ->", out)
    elif args.action == "stats":
        print(open(os.path.join(cfg.corpus_dir, "catalog_stats.json")).read())


def cmd_cluster(args):
    from dishcovery.serving import cluster

    cfg = _cfg(args)
    if args.action == "up":
        print(json.dumps(cluster.up(cfg, gateway=not args.no_gateway)))
        print("gateway:", cfg.gateway_url)
    elif args.action == "down":
        print("stopped", cluster.down(cfg, force=args.force), "processes")
    else:
        p = cfg.path("cluster.json")
        print(open(p).read() if os.path.exists(p) else "not running")


def cmd_serve(args):
    from dishcovery.serving.gateway import serve

    serve(args.config, args.port)


def cmd_search(args):
    from dishcovery.serving.broker import Broker, SearchParams
    from dishcovery.serving.gateway import load_components

    cfg = _cfg(args)

    async def go():
        qu, enc, ltr, beh = load_components(cfg)
        async with Broker(cfg, qu, enc, ltr, beh) as b:
            return await b.search(SearchParams(q=args.query, lat=args.lat, lon=args.lon, when=args.time, k=args.k,
                                               system=args.system, debug=args.debug, use_cache=False))
    r = asyncio.run(go())
    if args.json:
        print(json.dumps(r, indent=2, default=str))
        return
    u = r["understanding"]
    print(f"intent={u['intent']} corrected={u['corrected']!r} filters={ {k: v for k, v in (u.get('filters') or {}).items() if v} }")
    for i, x in enumerate(r["results"], 1):
        print(f"{i:2d}. [{x['type'][0]}] {x['name']} — {x['restaurant_name']} · {', '.join(x['cuisines'] or [])} · "
              f"★{x['rating']} · {x['distance_km']} km · {'open' if x['open_now'] else 'closed'} · {x['paths']}")
    print("timings (ms):", r["timings_ms"])


def cmd_qu(args):
    from dishcovery.qu.pipeline import QueryUnderstanding

    cfg = _cfg(args)
    qu = QueryUnderstanding.load(cfg.qu_dir)
    print(json.dumps(qu.parse(args.query).to_dict(), indent=2))


def cmd_labels(args):
    cfg = _cfg(args)
    from dishcovery.data.docstore import DocStore

    if args.action == "querylog":
        from dishcovery.labels.querylog import generate_query_log, save_query_log
        qs = generate_query_log(cfg.docs_path, args.n, seed=cfg["seed"], eval_queries_path=cfg["eval"]["queries"])
        out = cfg.path("labels", "train_queries.jsonl")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        print(save_query_log(out, qs), "queries ->", out)
    elif args.action == "pool":
        from dishcovery.eval.io import load_runs
        from dishcovery.labels.pooling import build_pool, pool_stats, save_pool
        runs = load_runs(cfg.path("eval", "runs.jsonl"))
        if args.systems:
            runs = {s: r for s, r in runs.items() if s in args.systems.split(",")}
        existing = None
        if os.path.exists(cfg["eval"]["qrels"]):
            from dishcovery.eval.io import load_qrels
            existing = load_qrels(cfg["eval"]["qrels"], sources=("hand",))
        pool = build_pool(runs, depth=args.depth, existing_qrels=existing)
        out = args.out or "data/eval/pool.jsonl"
        save_pool(out, pool)
        print(json.dumps(pool_stats(pool)), "->", out)
    elif args.action == "hand":
        from dishcovery.labels.hand_label import run_labeler
        run_labeler(args.pool or "data/eval/pool.jsonl", cfg["eval"]["queries"], DocStore(cfg.docs_path),
                    cfg["eval"]["qrels"])
    elif args.action == "export-csv":
        from dishcovery.labels.hand_label import export_csv
        from dishcovery.labels.pooling import load_pool
        export_csv(load_pool(args.pool or "data/eval/pool.jsonl"), cfg["eval"]["queries"], DocStore(cfg.docs_path),
                   args.out or "data/eval/pool.csv")
    elif args.action == "import-csv":
        from dishcovery.eval.io import write_qrels
        from dishcovery.labels.hand_label import import_csv
        recs = import_csv(args.csv)
        print(write_qrels(cfg["eval"]["qrels"], recs, append=True), "judgements appended")


def cmd_train(args):
    from dishcovery.ranking.train import train_pipeline

    cfg = _cfg(args)
    if args.label_source:
        cfg["ranking"]["label_source"] = args.label_source
    print(json.dumps(train_pipeline(cfg), indent=2))


def cmd_eval(args):
    from dishcovery.eval.runner import evaluate_systems

    cfg = _cfg(args)
    systems = args.systems.split(",") if args.systems else None
    rep = evaluate_systems(cfg, systems, labels=args.labels)
    print(open(cfg.path("eval", f"report_{args.labels}.md")).read())


def cmd_es(args):
    from dishcovery.baselines.es import ElasticBaseline

    cfg = _cfg(args)
    es = ElasticBaseline(cfg["es"]["url"], cfg["es"]["index"])
    es.wait()
    if args.action == "index":
        es.create()

        def docs():
            with open(cfg.docs_path) as f:
                for line in f:
                    yield json.loads(line)
        st = es.bulk_index(docs())
        os.makedirs(cfg.path("results"), exist_ok=True)
        json.dump(st, open(cfg.path("results", "es_index.json"), "w"), indent=2)
        print(st)


def cmd_bench(args):
    cfg = _cfg(args)
    from dishcovery.bench import runners as B

    fn = getattr(B, f"bench_{args.which}")
    print(json.dumps(fn(cfg, args), indent=2, default=str))


def cmd_ablations(args):
    from dishcovery.experiments.ablations import run_all

    cfg = _cfg(args)
    run_all(cfg, only=args.only.split(",") if args.only else None)


def cmd_report(args):
    from dishcovery.experiments.report import write_results_md

    cfg = _cfg(args)
    print(write_results_md(cfg))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dishcovery", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.environ.get("DISHCOVERY_CONFIG", "configs/smoke.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="offline pipeline: prepare, embed, index, qu, suggest[, geostat]")
    p.add_argument("--steps", help="comma list, e.g. index,qu")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("data")
    p.add_argument("action", choices=["osm", "stats"])
    p.add_argument("--out")
    p.set_defaults(fn=cmd_data)

    p = sub.add_parser("cluster")
    p.add_argument("action", choices=["up", "down", "status"])
    p.add_argument("--no-gateway", action="store_true")
    p.add_argument("--force", action="store_true", help="down: also kill strays matched by work dir")
    p.set_defaults(fn=cmd_cluster)

    p = sub.add_parser("serve")
    p.add_argument("what", choices=["gateway"])
    p.add_argument("--port", type=int)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--lat", type=float, default=39.9526)
    p.add_argument("--lon", type=float, default=-75.1652)
    p.add_argument("--time", default=None, help='e.g. "tue 13:00" (default: now)')
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--system", default="full")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("qu", help="parse a query with query understanding")
    p.add_argument("query")
    p.set_defaults(fn=cmd_qu)

    p = sub.add_parser("labels")
    p.add_argument("action", choices=["querylog", "pool", "hand", "export-csv", "import-csv"])
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--systems")
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--pool")
    p.add_argument("--out")
    p.add_argument("--csv")
    p.set_defaults(fn=cmd_labels)

    p = sub.add_parser("train")
    p.add_argument("--label-source", choices=["weak", "clicks"])
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("eval")
    p.add_argument("--systems")
    p.add_argument("--labels", choices=["hand", "weak"], default="hand")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("es")
    p.add_argument("action", choices=["index"])
    p.set_defaults(fn=cmd_es)

    p = sub.add_parser("bench")
    p.add_argument("which", choices=["load", "nrt", "partition", "codecs", "merge", "hnsw", "filter", "hedging",
                                     "cache", "es_latency"])
    p.add_argument("--qps", help="comma list overriding loadtest.qps")
    p.add_argument("--duration", type=float)
    p.add_argument("--system")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("ablations")
    p.add_argument("--only")
    p.set_defaults(fn=cmd_ablations)

    p = sub.add_parser("report")
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
