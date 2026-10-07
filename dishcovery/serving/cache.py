"""Query-result cache with generation-based invalidation.

Every shard reports a monotonically increasing index *generation* (bumped on
refresh, delete, flush and merge). A cached entry records the generation
vector it was computed under; it is served only while every shard is still at
that generation. Index updates routed through the gateway bump the vector
immediately (explicit invalidation); updates that bypass the gateway are picked
up by the broker's generation poller within its interval.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Hashable, Optional, Sequence, Tuple


class ResultCache:
    def __init__(self, capacity: int = 10000, ttl_s: float = 300.0) -> None:
        self.capacity = capacity
        self.ttl_s = ttl_s
        self._data: "OrderedDict[Hashable, Tuple[float, Tuple[int, ...], Any]]" = OrderedDict()
        self.hits = self.misses = self.stale = self.evictions = 0

    def get(self, key: Hashable, generations: Sequence[int]) -> Optional[Any]:
        if self.capacity <= 0:
            return None
        item = self._data.get(key)
        if item is None:
            self.misses += 1
            return None
        ts, gens, value = item
        if tuple(generations) != gens or time.monotonic() - ts > self.ttl_s:
            del self._data[key]
            self.stale += 1
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: Hashable, generations: Sequence[int], value: Any) -> None:
        if self.capacity <= 0:
            return
        self._data[key] = (time.monotonic(), tuple(generations), value)
        self._data.move_to_end(key)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)
            self.evictions += 1

    def clear(self) -> None:
        self._data.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"entries": len(self._data), "capacity": self.capacity, "hits": self.hits, "misses": self.misses,
                "stale": self.stale, "evictions": self.evictions, "hit_rate": self.hits / total if total else 0.0}
