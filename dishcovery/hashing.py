"""Shard routing hash. MUST match core/src/util/hash.h (FNV-1a 64-bit)."""
from __future__ import annotations

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK = 0xFFFFFFFFFFFFFFFF


def fnv1a64(s: str) -> int:
    h = _FNV_OFFSET
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * _FNV_PRIME) & _MASK
    return h


def shard_of(restaurant_id: str, num_shards: int) -> int:
    """Documents are partitioned by restaurant so a restaurant and its dishes co-locate."""
    return fnv1a64(restaurant_id) % num_shards
