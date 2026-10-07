"""Random access to docs.jsonl by id without loading the corpus.

An index of (hash64(id), byte offset) pairs is built once and stored next to
the file as ``<docs>.idx.npy`` (16 bytes/doc: 160 MB for 10M docs). Lookups
seek + parse one line. Implements the Mapping protocol so it can be handed to
the weak labeller and the labelling tool.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Dict, Iterator, Optional

import numpy as np

from dishcovery.hashing import fnv1a64


class DocStore(Mapping):
    def __init__(self, path: str, cache_size: int = 50000) -> None:
        self.path = path
        idx_path = path + ".idx.npy"
        if not os.path.exists(idx_path) or os.path.getmtime(idx_path) < os.path.getmtime(path):
            self._build(idx_path)
        arr = np.load(idx_path)
        self._hashes = arr[:, 0].copy()
        self._offsets = arr[:, 1].copy()
        self._f = open(path, "rb")
        self._cache: Dict[str, dict] = {}
        self._cache_size = cache_size

    def _build(self, idx_path: str) -> None:
        rows = []
        with open(self.path, "rb") as f:
            off = 0
            for line in f:
                i = line.find(b'"id"')
                j = line.find(b'"', line.find(b":", i) + 1)
                k = line.find(b'"', j + 1)
                rows.append((fnv1a64(line[j + 1:k].decode()), off))
                off += len(line)
        arr = np.array(rows, dtype=np.uint64).reshape(-1, 2)
        arr = arr[np.argsort(arr[:, 0], kind="stable")]
        np.save(idx_path, arr)

    def _read(self, off: int) -> dict:
        self._f.seek(int(off))
        return json.loads(self._f.readline())

    def __getitem__(self, doc_id: str) -> dict:
        d = self._cache.get(doc_id)
        if d is not None:
            return d
        h = np.uint64(fnv1a64(doc_id))
        lo = int(np.searchsorted(self._hashes, h, side="left"))
        hi = int(np.searchsorted(self._hashes, h, side="right"))
        for i in range(lo, hi):
            d = self._read(self._offsets[i])
            if d.get("id") == doc_id:
                if len(self._cache) >= self._cache_size:
                    self._cache.clear()
                self._cache[doc_id] = d
                return d
        raise KeyError(doc_id)

    def get(self, doc_id: str, default: Any = None) -> Any:
        try:
            return self[doc_id]
        except KeyError:
            return default

    def __iter__(self) -> Iterator[str]:
        with open(self.path) as f:
            for line in f:
                yield json.loads(line)["id"]

    def values(self):  # streaming, used by the weak labeller's corpus index
        with open(self.path) as f:
            for line in f:
                yield json.loads(line)

    def __len__(self) -> int:
        return len(self._offsets)
