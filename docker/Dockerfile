# Dishcovery: C++ engine + Python broker in one image.
#   docker build -f docker/Dockerfile -t dishcovery .
#   docker run --rm -it -v $PWD/work:/app/work -v $PWD/data/raw:/app/data/raw -p 8080:8080 dishcovery bash
FROM ubuntu:22.04 AS build
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates python3 python3-venv python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY CMakeLists.txt ./
COPY core ./core
# Portable binaries (no -march=native) so the image runs on any x86-64 host.
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDC_NATIVE=OFF && cmake --build build -j"$(nproc)" \
    && ./build/core/dishcovery_tests

FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip libgomp1 ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-full.txt pyproject.toml README.md ./
RUN python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt lightgbm uvloop
COPY dishcovery ./dishcovery
COPY configs ./configs
COPY data/eval ./data/eval
COPY scripts ./scripts
COPY tests ./tests
COPY --from=build /app/build/core /app/build/core
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080 7101 7111
CMD ["python", "-m", "dishcovery.cli", "--help"]
