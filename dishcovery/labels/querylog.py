"""Synthetic training query log.

Generates queries the way users would issue them against *this* corpus:
Zipf-distributed popularity over canonical dishes, cuisines, chains and
restaurant names, plus attribute and dietary templates; 10% get a random typo
and 3% are compound-joined ("chickenbiryani"). Each query is issued from a
location near (<= 3 km) a restaurant of the target's metro, at a time drawn
from the target's meal affinity.

Queries whose normalised text equals an evaluation query are excluded, so the
learning-to-rank model is never trained on the hand-labelled test queries.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from dishcovery.data.geo import destination
from dishcovery.data.schema import MEAL_BUCKETS
from dishcovery.labels.queries import DAYS, load_queries
from dishcovery.qu.text import normalize

_MEAL_HOURS = {"breakfast": (7, 10), "lunch": (11, 14), "snack": (15, 17), "dinner": (18, 21),
               "late_night": (22, 24)}
_ATTRIBUTE_TEMPLATES = ["open now", "cheap eats near me", "late night food", "top rated near me", "fast delivery",
                        "food near me", "breakfast open now", "best restaurants near me"]
_DIET_TEMPLATES = ["vegan {x}", "vegetarian {x}", "cheap {x}", "{x} near me", "spicy {x}", "best {x}",
                   "gluten free {x}", "{x} open now", "halal {x}"]


@dataclass
class TrainQuery:
    qid: str
    text: str
    query_class: str
    target_type: str       # dish | cuisine | chain | restaurant | attribute
    target: str            # canonical key ("" for attributes)
    lat: float
    lon: float
    time: str              # "tue 13:00"


def _typo(rng: random.Random, s: str) -> str:
    idx = [i for i, c in enumerate(s) if c.isalpha()]
    if len(idx) < 4:
        return s
    i = rng.choice(idx[1:-1])
    op = rng.randrange(4)
    if op == 0:
        return s[:i] + s[i + 1:]                              # deletion
    if op == 1 and i + 1 < len(s) and s[i + 1].isalpha():
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]            # transposition
    if op == 2:
        return s[:i] + s[i] + s[i:]                           # duplication
    return s[:i] + rng.choice("aeiou") + s[i + 1:]           # substitution


def _zipf_weights(n: int, s: float = 1.05) -> List[float]:
    return [1.0 / (r ** s) for r in range(1, n + 1)]


def _time_for(rng: random.Random, meal: Sequence[float]) -> str:
    bucket = rng.choices(MEAL_BUCKETS, weights=[m + 1e-3 for m in meal])[0]
    lo, hi = _MEAL_HOURS[bucket]
    h = rng.randrange(lo, hi)
    return f"{rng.choice(DAYS)} {h % 24:02d}:{rng.choice(['00', '30'])}"


def generate_query_log(docs_path: str, n_queries: int, seed: int = 0, eval_queries_path: Optional[str] = None,
                       typo_rate: float = 0.10, compound_rate: float = 0.03) -> List[TrainQuery]:
    rng = random.Random(seed)
    excluded: Set[str] = set()
    try:
        excluded = {normalize(q.text) for q in (load_queries(eval_queries_path) if eval_queries_path
                                                else load_queries())}
    except FileNotFoundError:
        pass

    # Compact per-entity index: popularity mass, locations, meal affinity.
    pop: Dict[Tuple[str, str], float] = Counter()
    locs: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
    meal: Dict[Tuple[str, str], List[float]] = {}
    with open(docs_path) as f:
        for line in f:
            d = json.loads(line)
            keys = []
            if d["type"] == "dish" and d.get("canonical_dish"):
                keys.append(("dish", d["canonical_dish"]))
            if d["type"] == "restaurant":
                for c in d.get("cuisines", [])[:1]:
                    keys.append(("cuisine", c.replace("_", " ")))
                if d.get("chain"):
                    keys.append(("chain", d["chain"]))
                elif d.get("review_count", 0) >= 20:
                    keys.append(("restaurant", normalize(d["name"])))
            for k in keys:
                pop[k] += 0.1 + d.get("popularity", 0.0)
                if len(locs[k]) < 50:
                    locs[k].append((d["lat"], d["lon"]))
                meal.setdefault(k, d.get("meal_affinity") or [0.2] * 5)

    by_type: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for k, _ in sorted(pop.items(), key=lambda kv: -kv[1]):
        by_type[k[0]].append(k)
    type_mix = [("dish", 0.55), ("cuisine", 0.12), ("chain", 0.10), ("restaurant", 0.08), ("attribute", 0.05),
                ("dietary", 0.10)]
    zipf = {t: _zipf_weights(len(v)) for t, v in by_type.items()}
    all_locs = [l for v in locs.values() for l in v]
    out: List[TrainQuery] = []
    attempts = 0
    while len(out) < n_queries and attempts < n_queries * 20:
        attempts += 1
        qtype = rng.choices([t for t, _ in type_mix], [w for _, w in type_mix])[0]
        if qtype == "attribute":
            text, cls, ttype, target = rng.choice(_ATTRIBUTE_TEMPLATES), "attribute", "attribute", ""
            lat, lon = rng.choice(all_locs) if all_locs else (0.0, 0.0)
            m = [0.2] * 5
        else:
            src = "dish" if qtype == "dietary" else qtype
            if not by_type.get(src):
                continue
            key = rng.choices(by_type[src], zipf[src])[0]
            ttype, target = key
            text = target
            cls = {"dish": "exact_dish", "cuisine": "cuisine", "chain": "chain", "restaurant": "restaurant_name"}[src]
            if qtype == "dietary":
                text = rng.choice(_DIET_TEMPLATES).format(x=target)
                cls = "dietary_structured"
            elif src == "cuisine" and rng.random() < 0.5:
                text = rng.choice(["{x} food", "best {x}", "{x} near me", "{x} restaurants"]).format(x=target)
            r = rng.random()
            if r < typo_rate:
                text, cls = _typo(rng, text), "misspelled"
            elif r < typo_rate + compound_rate and " " in target:
                text, cls = text.replace(target, target.replace(" ", "")), "compound"
            lat, lon = rng.choice(locs[key])
            m = meal.get(key, [0.2] * 5)
        if normalize(text) in excluded:
            continue
        lat, lon = destination(lat, lon, rng.uniform(0, 2 * math.pi), rng.uniform(0, 3.0))
        out.append(TrainQuery(qid=f"t{len(out):07d}", text=text, query_class=cls, target_type=ttype, target=target,
                              lat=round(lat, 6), lon=round(lon, 6), time=_time_for(rng, m)))
    return out


def save_query_log(path: str, queries: Iterable[TrainQuery]) -> int:
    n = 0
    with open(path, "w") as f:
        for q in queries:
            f.write(json.dumps(asdict(q)) + "\n")
            n += 1
    return n


def load_query_log(path: str) -> List[TrainQuery]:
    with open(path) as f:
        return [TrainQuery(**json.loads(line)) for line in f if line.strip()]
