"""Dish taxonomy loader and fast dish-mention matching."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from dishcovery.data.schema import MEAL_BUCKETS
from dishcovery.qu.text import normalize

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taxonomy.yaml")


@dataclass
class Dish:
    name: str
    cuisine: str
    variants: List[str]
    diet: str
    meal: List[float]          # MEAL_BUCKETS order
    price: Tuple[float, float]
    prep: float
    tags: List[str]
    gluten_free: bool = False

    @property
    def diets(self) -> List[str]:
        """Diet labels in the docs schema (vegan implies veg)."""
        out = [self.diet]
        if self.diet == "vegan":
            out.append("veg")
        if self.gluten_free:
            out.append("gluten_free")
        return out


@dataclass
class Cuisine:
    key: str
    aliases: List[str]
    yelp_categories: List[str]
    dishes: List[Dish] = field(default_factory=list)


class Taxonomy:
    def __init__(self, cuisines: Dict[str, Cuisine], generic: List[Dish]) -> None:
        self.cuisines = cuisines
        self.generic = generic
        self.dishes: Dict[str, Dish] = {}
        for c in cuisines.values():
            for d in c.dishes:
                self.dishes.setdefault(d.name, d)
        self._cat_map: Dict[str, str] = {}
        for c in cuisines.values():
            for cat in c.yelp_categories:
                self._cat_map.setdefault(cat.lower(), c.key)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Taxonomy":
        raw = yaml.safe_load(open(path or DEFAULT_PATH, encoding="utf-8"))
        profiles = raw.get("meal_profiles", {})

        def meal_vec(m) -> List[float]:
            d = profiles.get(m, m) if isinstance(m, str) else (m or {})
            v = [float(d.get(b, 0.0)) for b in MEAL_BUCKETS]
            s = sum(v) or 1.0
            return [x / s for x in v]

        def dish(d: dict, cuisine: str) -> Dish:
            lo, hi = d.get("price", [10, 15])
            return Dish(name=normalize(d["name"]), cuisine=cuisine,
                        variants=[normalize(v) for v in d.get("variants", [])],
                        diet=d.get("diet", "non_veg"), meal=meal_vec(d.get("meal", "main")),
                        price=(float(lo), float(hi)), prep=float(d.get("prep", 15)),
                        tags=[str(t) for t in d.get("tags", [])], gluten_free=bool(d.get("gf", False)))

        cuisines = {}
        for key, c in raw["cuisines"].items():
            cu = Cuisine(key=key, aliases=c.get("aliases", []), yelp_categories=c.get("yelp_categories", []))
            cu.dishes = [dish(d, key) for d in c.get("dishes", [])]
            cuisines[key] = cu
        generic = [dish(d, "generic") for d in raw.get("generic_dishes", [])]
        return cls(cuisines, generic)

    def cuisines_for_categories(self, categories: Sequence[str]) -> List[str]:
        out: List[str] = []
        for cat in categories:
            k = self._cat_map.get(cat.strip().lower())
            if k and k not in out:
                out.append(k)
        return out

    def dish_matcher(self) -> "DishMatcher":
        return DishMatcher(self)


class DishMatcher:
    """Find taxonomy dish mentions (names + variants) in free text.

    Token n-gram lookup restricted to positions whose token can start a dish
    phrase, so the cost is ~one set lookup per token.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, taxonomy: Taxonomy) -> None:
        self.phrases: Dict[Tuple[str, ...], str] = {}
        for d in taxonomy.dishes.values():
            for s in [d.name] + d.variants:
                toks = tuple(s.split())
                if toks and len(toks) <= 6 and not (len(toks) == 1 and len(toks[0]) < 4):
                    self.phrases.setdefault(toks, d.name)
        self.first = {p[0] for p in self.phrases}
        self.max_len = max((len(p) for p in self.phrases), default=1)

    def tokens(self, text: str) -> List[str]:
        return self._TOKEN_RE.findall(text.lower().replace("'", ""))

    def find(self, text: str) -> List[Tuple[str, int, int]]:
        """Return (canonical dish, start token, end token) for non-overlapping mentions."""
        toks = self.tokens(text)
        out = []
        i, n = 0, len(toks)
        while i < n:
            if toks[i] in self.first:
                for L in range(min(self.max_len, n - i), 0, -1):
                    d = self.phrases.get(tuple(toks[i:i + L]))
                    if d:
                        out.append((d, i, i + L))
                        i += L
                        break
                else:
                    i += 1
            else:
                i += 1
        return out

    def mentions(self, text: str) -> List[str]:
        return [d for d, _, _ in self.find(text)]
