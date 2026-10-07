PY      ?= .venv/bin/python
JOBS    ?= $(shell (nproc 2>/dev/null || sysctl -n hw.ncpu) )
CONFIG  ?= configs/smoke.yaml
DC       = $(PY) -m dishcovery.cli --config $(CONFIG)

.PHONY: all venv build build-asan test test-cpp test-py smoke smoke-down full lint clean

all: build

venv:
	python3 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

venv-full: venv
	$(PY) -m pip install -r requirements-full.txt

build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j $(JOBS)

build-asan:
	cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DDC_SANITIZE=ON -DDC_BUILD_BENCH=OFF
	cmake --build build-asan -j $(JOBS) --target dishcovery_tests
	./build-asan/core/dishcovery_tests

test: test-cpp test-py

test-cpp: build
	./build/core/dishcovery_tests

test-py:
	$(PY) -m pytest -q

# End-to-end on a laptop: synthetic data, 2 shards x 2 replicas, train, eval (weak labels), all benches.
smoke: build
	bash scripts/smoke.sh

smoke-down:
	$(DC) cluster down

# Full experiment on external compute (see docs/RUNBOOK.md).
full: build
	bash scripts/run_full.sh configs/yelp.yaml

clean:
	rm -rf build build-asan
