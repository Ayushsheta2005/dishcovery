"""Chain (brand) detection from business names."""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, Set

from dishcovery.qu.text import normalize

_LOCATION_SUFFIX = re.compile(
    r"(\s*[-–—|@]\s*.*$)|(\s+#\s*\d+.*$)|(\s+no\.?\s*\d+$)|(\s+(at|in)\s+the\s+.*$)|(\s+store\s*\d+$)",
    re.IGNORECASE)


def brand_key(name: str) -> str:
    """'Domino's Pizza #1234 - Downtown' -> 'dominos pizza'."""
    base = _LOCATION_SUFFIX.sub("", name or "")
    return normalize(base)


def detect_chains(names: Iterable[str], min_count: int = 5) -> Set[str]:
    counts = Counter(brand_key(n) for n in names)
    return {k for k, c in counts.items() if k and c >= min_count}


def chain_counts(names: Iterable[str]) -> Dict[str, int]:
    return dict(Counter(brand_key(n) for n in names))
