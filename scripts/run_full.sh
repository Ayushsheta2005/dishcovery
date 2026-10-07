#!/usr/bin/env bash
# Full experiment on external compute. See docs/RUNBOOK.md for hardware and the
# manual hand-labelling step (the only part that cannot be automated).
#
#   bash scripts/run_full.sh configs/yelp.yaml          # quality + latency at Yelp scale (~2M docs)
#   bash scripts/run_full.sh configs/1m.yaml latency    # latency-only at 1M docs
#   bash scripts/run_full.sh configs/10m.yaml latency   # latency-only at 10M docs
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-configs/yelp.yaml}
MODE=${2:-all}            # all | latency
PY=${PY:-.venv/bin/python}
DC="$PY -m dishcovery.cli --config $CFG"
ES_URL=${ES_URL:-http://localhost:9200}

echo "== build engine"
[ -x build/core/dishcovery-shard ] || { cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"$(nproc)"; }
./build/core/dishcovery_tests > /dev/null && echo "C++ unit tests: OK"

echo "== offline pipeline (data, embeddings, index, QU, suggest, geo resolution study)"
$DC build
$DC build --steps geostat

trap '$DC cluster down || true' EXIT
$DC cluster down --force || true
$DC cluster up

if [ "$MODE" = "all" ]; then
  echo "== learning to rank"
  $DC train
  echo "== Elasticsearch baseline (skipped if ES is not reachable)"
  if curl -s "$ES_URL" > /dev/null; then $DC es index; else echo "ES not reachable at $ES_URL (docker compose -f docker/docker-compose.yml up -d)"; fi
  echo "== evaluation"
  if [ -f data/eval/hand_labels.jsonl ]; then
    $DC eval --labels hand
  else
    $DC eval --labels weak
    $DC labels pool --depth 10
    echo
    echo "Hand labels missing. Label the pool (data/eval/pool.jsonl), then rerun:"
    echo "  $DC labels hand          # interactive, resumable"
    echo "  $DC eval --labels hand"
  fi
  $DC ablations
else
  $DC ablations --only codecs,merge,hnsw,partition,filter,nrt,load,cache,hedging
fi
$DC report   # prints the path of RESULTS.md
