#!/usr/bin/env bash
# One-time setup on a fresh Linux box (Ubuntu 22.04+): toolchain, venv, engine build.
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v apt-get > /dev/null; then
  sudo apt-get update
  sudo apt-get install -y build-essential cmake git python3 python3-venv python3-pip curl
fi
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements-full.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
./build/core/dishcovery_tests
.venv/bin/python -m pytest -q
echo "ready: see docs/RUNBOOK.md"
