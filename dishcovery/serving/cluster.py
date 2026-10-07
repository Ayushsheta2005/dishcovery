"""Launch / stop a local cluster: S shards x R replicas, the suggest service and the gateway.

Process ids are recorded in <work_dir>/cluster.json so ``down`` can stop them.
For multi-machine deployments run ``dishcovery-shard`` on each host and point
the broker at them with the ``serving`` section of the config.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List

from dishcovery.config import Config, REPO_ROOT


def binary(name: str) -> str:
    for d in ("build/core", "build-release/core", "build"):
        p = os.path.join(REPO_ROOT, d, name)
        if os.path.exists(p):
            return p
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"{name} not found; build the C++ engine first (make build)")


def _wait_http(url: str, timeout_s: float = 60.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.2)
    return False


def _prepare_replica_dirs(cfg: Config) -> None:
    """Writable replicas need their own index copy (each keeps its own WAL/segments)."""
    if cfg["serving"].get("read_only"):
        return
    for s in range(cfg["index"]["shards"]):
        src = cfg.shard_dir(s, 0)
        for r in range(1, cfg["index"]["replicas"]):
            dst = cfg.shard_dir(s, r)
            if not os.path.exists(dst):
                shutil.copytree(src, dst)


def _port_in_use(host: str, port: int) -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def up(cfg: Config, gateway: bool = True, suggest: bool = True, wait: bool = True) -> Dict[str, List[int]]:
    sv0 = cfg["serving"]
    ports = [int(u.rsplit(":", 1)[1]) for urls in cfg.shard_urls() for u in urls]
    ports += [sv0["suggest_port"]] + ([sv0["gateway_port"]] if gateway else [])
    busy = [p for p in ports if _port_in_use(sv0["host"], p)]
    if busy:
        raise RuntimeError(f"ports already in use: {busy}. A previous cluster may still be running: "
                           f"`dishcovery --config {cfg['_config_path']} cluster down --force`")
    logs = cfg.path("logs")
    os.makedirs(logs, exist_ok=True)
    _prepare_replica_dirs(cfg)
    sv = cfg["serving"]
    procs: Dict[str, List[int]] = {"shards": [], "suggest": [], "gateway": []}
    urls = cfg.shard_urls()
    for s, rep_urls in enumerate(urls):
        for r, url in enumerate(rep_urls):
            port = int(url.rsplit(":", 1)[1])
            args = [binary("dishcovery-shard"), "--index", cfg.shard_dir(s, r), "--schema", cfg.schema_path,
                    "--port", str(port), "--shard-id", str(s), "--replica", f"r{r}", "--host", sv["host"],
                    "--threads", str(sv["shard_threads"]), "--filter-cache-mb", str(sv["filter_cache_mb"]),
                    "--postings-cache-mb", str(sv["postings_cache_mb"]), "--refresh-ms", str(sv["refresh_ms"]),
                    "--flush-ms", str(sv["flush_ms"]),
                    "--merge-policy", json.dumps(cfg["index"]["merge_policy"])]
            if sv.get("read_only"):
                args.append("--read-only")
            log = open(os.path.join(logs, f"shard{s}_r{r}.log"), "a")
            procs["shards"].append(subprocess.Popen(args, stdout=log, stderr=log).pid)
    fst = cfg.path("suggest", "suggest.fst")
    if suggest and os.path.exists(fst):
        log = open(os.path.join(logs, "suggest.log"), "a")
        procs["suggest"].append(subprocess.Popen(
            [binary("dishcovery-suggest"), "serve", "--fst", fst, "--port", str(sv["suggest_port"])],
            stdout=log, stderr=log).pid)
    if wait:
        for rep_urls in urls:
            for url in rep_urls:
                if not _wait_http(url + "/health", 120):
                    raise RuntimeError(f"shard at {url} did not become healthy; see {logs}")
    if gateway:
        log = open(os.path.join(logs, "gateway.log"), "a")
        env = dict(os.environ, PYTHONPATH=REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        procs["gateway"].append(subprocess.Popen(
            [sys.executable, "-m", "dishcovery.cli", "--config", cfg["_config_path"], "serve", "gateway"],
            stdout=log, stderr=log, env=env, cwd=REPO_ROOT).pid)
        if wait and not _wait_http(cfg.gateway_url + "/health", 180):
            raise RuntimeError(f"gateway did not become healthy; see {logs}/gateway.log")
    with open(cfg.path("cluster.json"), "w") as f:
        json.dump(procs, f)
    return procs


def down(cfg: Config, force: bool = False) -> int:
    """Stop the processes recorded in cluster.json; with ``force``, also any process whose
    command line references this config's work dir or config file (strays from a lost pid file)."""
    path = cfg.path("cluster.json")
    procs = json.load(open(path)) if os.path.exists(path) else {}
    if force:
        needles = [os.path.abspath(cfg.path("index")), os.path.abspath(cfg.path("suggest"))]
        if cfg.get("_config_path"):
            needles.append(f"--config {cfg['_config_path']}")
        for needle in needles:
            r = subprocess.run(["pgrep", "-f", needle], capture_output=True, text=True)
            procs.setdefault("strays", []).extend(int(x) for x in r.stdout.split() if int(x) != os.getpid())
    if not procs:
        return 0
    n = 0
    for pids in procs.values():
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                n += 1
            except ProcessLookupError:
                pass
    time.sleep(1.0)
    if os.path.exists(path):
        os.remove(path)
    return n
