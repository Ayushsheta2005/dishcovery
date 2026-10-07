"""Synthetic menus: dish-level documents for every restaurant.

Dishes come from the taxonomy cuisines matched by the restaurant's categories,
weighted towards dishes its reviews actually mention ("signature" dishes), with
realistic name variation and prices scaled by price tier. Chains get the same
menu at every location (seeded by the brand), which is what makes chain queries
behave like they do in production.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from dishcovery.data.taxonomy import Dish, Taxonomy

TIER_MULT = {1: 0.75, 2: 1.0, 3: 1.4, 4: 2.0}
PREFIXES = ["Classic", "House Special", "Spicy", "Homestyle", "Signature", "Chef's", "Traditional", "Grandma's"]
SUFFIXES = ["Platter", "Combo", "Bowl", "(Large)", "(Regular)", "Plate", "Special"]
REGIONAL = {"indian": ["Hyderabadi", "Punjabi", "Mumbai", "Kerala", "Chettinad"],
            "chinese": ["Sichuan", "Cantonese", "Hunan"], "mexican": ["Oaxacan", "Jalisco"],
            "italian": ["Roman", "Neapolitan", "Sicilian"], "thai": ["Bangkok", "Chiang Mai"]}
DESC_OPENERS = ["A {adj} take on {name}", "Our {adj} {name}", "{Name}, made {adj}", "House-made {name}",
                "{Name} the way locals love it"]
ADJ = ["fresh", "made-to-order", "hearty", "classic", "slow-cooked", "authentic", "bold", "comforting"]


def _seed(*parts: str) -> int:
    return int(hashlib.md5("|".join(parts).encode()).hexdigest()[:12], 16)


def _vary_name(d: Dish, rng: random.Random) -> str:
    base = d.name
    r = rng.random()
    if r < 0.25 and d.variants:
        base = rng.choice(d.variants)
    elif r < 0.35 and d.cuisine in REGIONAL:
        base = f"{rng.choice(REGIONAL[d.cuisine])} {base}"
    elif r < 0.45:
        base = f"{rng.choice(PREFIXES)} {base}"
    if rng.random() < 0.12:
        base = f"{base} {rng.choice(SUFFIXES)}"
    return " ".join(w if w.startswith("(") else w.capitalize() for w in base.split())


def describe(d: Dish, rng: random.Random) -> str:
    opener = rng.choice(DESC_OPENERS).format(adj=rng.choice(ADJ), name=d.name, Name=d.name.capitalize())
    tags = [t for t in d.tags if t not in d.name][:4]
    detail = (" — " + ", ".join(tags) + ".") if tags else "."
    return opener + detail


def generate_menu(business: dict, taxonomy: Taxonomy, cuisines: Sequence[str], mentions: Dict[str, int],
                  price_tier: int, min_dishes: int = 12, max_dishes: int = 45,
                  chain_key: str = "") -> List[dict]:
    """Return dish records (name, canonical, diet list, price, ...) for one restaurant."""
    rng = random.Random(_seed(chain_key) if chain_key else _seed(business["business_id"]))
    pool: List[Dish] = []
    for c in cuisines:
        if c in taxonomy.cuisines:
            pool.extend(taxonomy.cuisines[c].dishes)
    if not pool:
        pool = list(taxonomy.cuisines["american"].dishes)
    # Chains serve one brand-wide menu: per-location reviews must not change it.
    pick_mentions = {} if chain_key else mentions
    # Mentioned dishes from other cuisines (fusion places) are still served.
    for dname in pick_mentions:
        d = taxonomy.dishes.get(dname)
        if d is not None and d not in pool:
            pool.append(d)
    n = rng.randint(min_dishes, max_dishes)
    weights = [1.0 + 4.0 * math.log1p(pick_mentions.get(d.name, 0)) for d in pool]
    chosen: List[Dish] = []
    candidates = list(range(len(pool)))
    while candidates and len(chosen) < min(n, len(pool)):
        i = rng.choices(candidates, [weights[k] for k in candidates])[0]
        chosen.append(pool[i])
        candidates.remove(i)
    if rng.random() < 0.6:
        chosen += rng.sample(taxonomy.generic, min(len(taxonomy.generic), rng.randint(1, 3)))
    mult = TIER_MULT.get(price_tier, 1.0)
    items = []
    seen = set()
    for d in chosen:
        name = _vary_name(d, rng)
        if name.lower() in seen:
            name = d.name.title()
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        lo, hi = d.price
        price = round(max(0.99, rng.uniform(lo, hi) * mult * rng.uniform(0.9, 1.1)), 2)
        items.append({
            "name": name, "canonical_dish": d.name, "cuisine": d.cuisine, "diet": d.diets, "price": price,
            "prep_time_min": round(d.prep * rng.uniform(0.8, 1.3), 1), "meal_affinity": d.meal,
            "tags": list(d.tags), "description": describe(d, rng), "mentions": mentions.get(d.name, 0),
        })
    return items
