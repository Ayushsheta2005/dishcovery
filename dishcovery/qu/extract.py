"""Structured attribute extraction.

Turns the attribute words of a query into filters and hints, and returns the
remaining *content* tokens::

    "cheap vegan near me open now"
      -> filters: diet=[vegan], price=low (price_tier_max=2), open_now=True
         geo="user", content tokens: []

Patterns are matched longest-first over the token list, so "fine dining"
wins over "dining" and "late night" over "night".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from dishcovery.qu.types import QueryFilters

_MILES_TO_KM = 1.609344

# (phrase tokens) -> action. Actions are applied by Extractor._apply.
_DIET = {
    ("vegan",): ["vegan"],
    ("plant", "based"): ["vegan"],
    ("vegetarian",): ["veg"],
    ("veg",): ["veg"],
    ("pure", "veg"): ["veg"],
    ("eggless",): ["veg"],
    ("jain",): ["jain"],
    ("gluten", "free"): ["gluten_free"],
    ("glutenfree",): ["gluten_free"],
    ("halal",): ["halal"],
    ("non", "veg"): ["non_veg"],
    ("nonveg",): ["non_veg"],
    ("non", "vegetarian"): ["non_veg"],
}
_PRICE = {
    ("cheap",): "low", ("budget",): "low", ("affordable",): "low", ("inexpensive",): "low",
    ("low", "cost"): "low", ("cheap", "eats"): "low", ("value",): "low", ("economical",): "low",
    ("expensive",): "high", ("upscale",): "high", ("fine", "dining"): "high", ("fancy",): "high",
    ("pricey",): "high", ("luxury",): "high", ("high", "end"): "high", ("splurge",): "high",
    ("mid", "range"): "mid", ("midrange",): "mid", ("moderately", "priced"): "mid", ("moderate",): "mid",
}
_GEO = {
    ("near", "me"), ("nearby",), ("near", "by"), ("around", "me"), ("close", "by"), ("close", "to", "me"),
    ("around", "here"), ("near", "here"), ("in", "my", "area"), ("closest",), ("nearest",),
}
_OPEN = {("open", "now"): None, ("open",): None, ("open", "late"): "late_night", ("late", "night"): "late_night",
         ("24", "hours"): None, ("24", "7"): None, ("24", "hour"): None, ("midnight",): "late_night"}
_RATING = {("top", "rated"), ("highly", "rated"), ("best", "rated"), ("best",), ("top",), ("five", "star"),
           ("5", "star"), ("well", "reviewed")}
_FAST = {("fast",), ("quick",), ("quickly",), ("fast", "delivery"), ("quick", "delivery"), ("express",),
         ("quick", "bite")}
_MEAL = {"breakfast": "breakfast", "brunch": "breakfast", "lunch": "lunch", "dinner": "dinner",
         "supper": "dinner", "snack": "snack", "snacks": "snack"}
_RESTAURANT_WORDS = {"restaurant", "restaurants", "place", "places", "spot", "spots", "joint", "joints",
                     "eatery", "eateries", "outlet", "outlets"}
_NOISE = {"delivery", "deliver", "delivered", "order", "ordering", "takeout", "take", "away", "takeaway",
          "online", "please", "want", "get", "some", "something", "i", "me", "for", "to", "eat", "the", "a",
          "an", "food", "foods", "with", "options", "option", "menu", "dish", "dishes", "item", "items",
          "good", "great", "nice", "tasty", "yummy", "delicious", "craving", "now", "today", "tonight"}
# Noise words that are dropped only when other content remains ("food" alone is a query).
_SOFT_NOISE = {"food", "foods", "something", "dish", "dishes", "good", "great", "tasty", "delicious"}

_NUM_RE = re.compile(r"^\d+(\.\d+)?$")
_PHRASES: List[Tuple[Tuple[str, ...], str]] = []
for _p in _DIET:
    _PHRASES.append((_p, "diet"))
for _p in _PRICE:
    _PHRASES.append((_p, "price"))
for _p in _GEO:
    _PHRASES.append((_p, "geo"))
for _p in _OPEN:
    _PHRASES.append((_p, "open"))
for _p in _RATING:
    _PHRASES.append((_p, "rating"))
for _p in _FAST:
    _PHRASES.append((_p, "fast"))
_PHRASES.sort(key=lambda x: -len(x[0]))
_MAX_PHRASE = max(len(p) for p, _ in _PHRASES)
_PHRASE_INDEX: Dict[Tuple[str, ...], str] = {}
for _p, _k in _PHRASES:
    _PHRASE_INDEX.setdefault(_p, _k)


@dataclass
class Extraction:
    filters: QueryFilters = field(default_factory=QueryFilters)
    geo: str = "user"
    radius_km: Optional[float] = None
    sort_hint: Optional[str] = None
    meal_time: Optional[str] = None
    content: List[str] = field(default_factory=list)       # remaining content tokens
    attributes: List[str] = field(default_factory=list)    # tokens consumed as attributes
    spans: List[Tuple[int, int, str]] = field(default_factory=list)  # (start, end, kind)


def _price_bounds(level: str, f: QueryFilters) -> None:
    f.price = level
    if level == "low":
        f.price_tier_max = 2
    elif level == "high":
        f.price_tier_min = 3
    elif level == "mid":
        f.price_tier_min, f.price_tier_max = 2, 3


def extract(tokens: Sequence[str]) -> Extraction:
    """Extract attributes from normalised tokens."""
    ex = Extraction()
    f = ex.filters
    toks = list(tokens)
    n = len(toks)
    i = 0
    content_idx: List[int] = []
    while i < n:
        t = toks[i]
        # "under 15 [dollars|bucks|usd]" / "below $10" / "less than 20"
        if t in ("under", "below", "upto", "within", "less") and i + 1 < n:
            j = i + 1
            if t == "less" and j < n and toks[j] == "than":
                j += 1
            if j < n and _NUM_RE.match(toks[j]):
                val = float(toks[j])
                k = j + 1
                unit = toks[k] if k < n else ""
                if t == "within" and unit in ("km", "kms", "kilometer", "kilometers", "kilometres"):
                    ex.radius_km = val
                    ex.spans.append((i, k + 1, "geo"))
                    i = k + 1
                    continue
                if t == "within" and unit in ("mile", "miles", "mi"):
                    ex.radius_km = round(val * _MILES_TO_KM, 3)
                    ex.spans.append((i, k + 1, "geo"))
                    i = k + 1
                    continue
                if t != "within":
                    f.max_price = val
                    end = k + 1 if unit in ("dollars", "dollar", "bucks", "usd", "rs", "rupees", "inr") else k
                    ex.spans.append((i, end, "price"))
                    i = end
                    continue
        # "4 stars", "4 star", "4 plus stars", "4.5 stars"
        if _NUM_RE.match(t) and i + 1 < n:
            j = i + 1
            if toks[j] in ("plus", "and", "or") and j + 1 < n and toks[j + 1] == "above":
                j += 2
            elif toks[j] == "plus":
                j += 1
            if j < n and toks[j] in ("star", "stars"):
                v = float(t)
                if 0 < v <= 5:
                    f.min_rating = v
                    ex.sort_hint = ex.sort_hint or "rating"
                    end = j + 1
                    if end < n and toks[end] in ("and", "plus") and end + 1 < n and toks[end + 1] == "above":
                        end += 2
                    ex.spans.append((i, end, "rating"))
                    i = end
                    continue
        matched = False
        for L in range(min(_MAX_PHRASE, n - i), 0, -1):
            key = tuple(toks[i:i + L])
            kind = _PHRASE_INDEX.get(key)
            if kind is None:
                continue
            if kind == "diet":
                for d in _DIET[key]:
                    if d not in f.diet:
                        f.diet.append(d)
            elif kind == "price":
                _price_bounds(_PRICE[key], f)
            elif kind == "geo":
                ex.geo = "user"
            elif kind == "open":
                f.open_now = True
                if _OPEN[key]:
                    ex.meal_time = _OPEN[key]
            elif kind == "rating":
                ex.sort_hint = "rating"
            elif kind == "fast":
                ex.sort_hint = ex.sort_hint or "fast"
            ex.spans.append((i, i + L, kind))
            i += L
            matched = True
            break
        if matched:
            continue
        if t in _MEAL:
            ex.meal_time = ex.meal_time or _MEAL[t]
            content_idx.append(i)  # meal words are also content ("breakfast" matches categories)
            i += 1
            continue
        if t in _RESTAURANT_WORDS:
            f.doc_type = "restaurant"
            ex.spans.append((i, i + 1, "doc_type"))
            i += 1
            continue
        content_idx.append(i)
        i += 1

    # Drop pure noise; soft noise only when something else remains.
    hard = [k for k in content_idx if toks[k] not in _NOISE or toks[k] in _SOFT_NOISE]
    non_soft = [k for k in hard if toks[k] not in _SOFT_NOISE]
    keep = non_soft if non_soft else hard
    if not non_soft and keep and all(toks[k] in _SOFT_NOISE for k in keep) and (ex.spans or f.diet):
        keep = []  # "cheap vegan food near me" -> no content
    ex.content = [toks[k] for k in keep]
    consumed = set(range(n)) - set(keep)
    ex.attributes = [toks[k] for k in sorted(consumed)]
    return ex
