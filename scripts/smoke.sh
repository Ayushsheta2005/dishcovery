#!/usr/bin/env bash
# Laptop end-to-end check (~3-5 minutes): synthetic Yelp-format data, hashing
# embeddings, 2 shards x 2 replicas, LambdaMART on weak labels, weak-label eval,
# every benchmark at toy scale. Validates the plumbing — the numbers are NOT results.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CFG=${CFG:-configs/smoke.yaml}
DC="$PY -m dishcovery.cli --config $CFG"

[ -x build/core/dishcovery-shard ] || { cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j; }
./build/core/dishcovery_tests > /dev/null && echo "C++ unit tests: OK"
$PY -m pytest -q tests

trap '$DC cluster down || true' EXIT
$DC cluster down --force || true
$DC build
$DC cluster up
$DC search "chickenbiryani" --time "fri 19:30" --k 5 --system hybrid
$DC search "cheap vegan near me open now" --time "sat 13:00" --k 5 --system hybrid
$DC train
$DC eval --labels weak --systems exact,bm25,bm25_qu,hybrid,ltr,full
$DC ablations --only filter,codecs,merge,hnsw,partition,geostat,nrt,load,cache,hedging
echo
echo "Report: work/smoke/results/RESULTS.md"
