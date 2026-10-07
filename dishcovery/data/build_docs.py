"""Yelp (+ OSM) businesses + reviews + synthetic menus -> docs.jsonl.

Each restaurant document is immediately followed by its dish documents, so a
restaurant's docs are contiguous in the file (``scale_docs`` and
``subsample_docs`` rely on that to stream without holding the corpus).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter
from typing import Dict, Iterator, List, Optional, Tuple

from dishcovery.data.chains import brand_key, detect_chains
from dishcovery.data.geo import DensityGrid, delivery_polygon, delivery_radius_km
from dishcovery.data.hours import default_hours, yelp_hours_to_hex
from dishcovery.data.menus import generate_menu
from dishcovery.data.osm import dedupe_against, osm_to_businesses
from dishcovery.data.schema import MEAL_BUCKETS, validate_doc
from dishcovery.data.taxonomy import Taxonomy
from dishcovery.data.yelp import aggregate_reviews, load_businesses

BUSINESS_FILE = "yelp_academic_dataset_business.json"
REVIEW_FILE = "yelp_academic_dataset_review.json"


def _price_tier(b: dict) -> int:
    attrs = b.get("attributes") or {}
    v = attrs.get("RestaurantsPriceRange2")
    try:
        t = int(str(v).strip("'\" "))
        if 1 <= t <= 4:
            return t
    except (TypeError, ValueError):
        pass
    return 2


def _popularity(review_count: int, mentions: int = 0, is_dish: bool = False) -> float:
    """0..1 prior: log review volume, dishes boosted by how often reviews mention them."""
    r = min(1.0, math.log1p(max(review_count, 0)) / 8.0)
    if not is_dish:
        return round(r, 4)
    m = min(1.0, math.log1p(mentions) / 3.0)
    return round(0.55 * r + 0.45 * m, 4)


def _restaurant_diets(menu: List[dict]) -> List[str]:
    """Diet labels a restaurant can honestly claim: >= 3 real (non-generic) dishes and
    >= 20% of them. Fries and soda do not make a fried-chicken shop "vegan"."""
    real = [m for m in menu if m["cuisine"] != "generic"]
    c = Counter(d for item in real for d in item["diet"])
    return sorted(d for d, n in c.items() if n >= 3 and n >= 0.2 * len(real))


def build_docs(yelp_dir: str, out_path: str, taxonomy_path: Optional[str] = None, osm_path: Optional[str] = None,
               max_businesses: Optional[int] = None, seed: int = 0, min_dishes: int = 12, max_dishes: int = 45,
               chain_min_count: Optional[int] = None, review_chars: int = 2000, workers: Optional[int] = None,
               validate_every: int = 1000) -> dict:
    tax = Taxonomy.load(taxonomy_path)
    businesses = load_businesses(os.path.join(yelp_dir, BUSINESS_FILE), max_businesses=max_businesses)
    n_yelp = len(businesses)
    if osm_path:
        osm = dedupe_against(businesses, osm_to_businesses(osm_path))
        businesses.extend(osm)
    businesses.sort(key=lambda b: b["business_id"])
    # A brand is a chain once it has >= 5 locations (Yelp scale); small
    # synthetic corpora use a proportionally lower bar.
    if chain_min_count is None:
        chain_min_count = max(2, min(5, len(businesses) // 400))
    chains = detect_chains((b["name"] for b in businesses), min_count=chain_min_count)
    # Brand-level profile so every location of a chain serves the same menu.
    brand_cuisines: Dict[str, Counter] = {}
    brand_tiers: Dict[str, Counter] = {}
    for b in businesses:
        ck = brand_key(b["name"])
        if ck in chains:
            cs = tuple(tax.cuisines_for_categories(b.get("categories_list") or []))
            brand_cuisines.setdefault(ck, Counter())[cs] += 1
            brand_tiers.setdefault(ck, Counter())[_price_tier(b)] += 1
    ids = {b["business_id"] for b in businesses if b.get("source") != "osm"}
    review_path = os.path.join(yelp_dir, REVIEW_FILE)
    aggs = aggregate_reviews(review_path, ids, taxonomy_path, workers=workers) if os.path.exists(review_path) else {}
    density = DensityGrid([(b["latitude"], b["longitude"]) for b in businesses])

    stats = Counter()
    per_cuisine = Counter()
    per_city = Counter()
    chain_docs = Counter()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as out:
        for b in businesses:
            bid = b["business_id"]
            rng = random.Random(int(hashlib.md5(f"{seed}|{bid}".encode()).hexdigest()[:12], 16))
            cats = b.get("categories_list") or []
            cuisines = tax.cuisines_for_categories(cats) or (["cafe"] if "Coffee & Tea" in cats else ["american"])
            tier = _price_tier(b)
            agg = aggs.get(bid)
            mentions = dict(agg.dish_mentions) if agg else {}
            ck = brand_key(b["name"])
            chain = ck if ck in chains else ""
            if chain:
                mode_cuisines = brand_cuisines[chain].most_common(1)[0][0]
                cuisines = list(mode_cuisines)[:1] or cuisines  # brands have one core menu
                tier = brand_tiers[chain].most_common(1)[0][0]
            menu = generate_menu(b, tax, cuisines, mentions, tier, min_dishes, max_dishes, chain_key=chain)
            if not menu:
                continue
            lat, lon = float(b["latitude"]), float(b["longitude"])
            radius = delivery_radius_km(density.density(lat, lon), tier, rng)
            polygon = delivery_polygon(lat, lon, radius, rng)
            hours = b.get("hours_hex") or yelp_hours_to_hex(b.get("hours")) or default_hours(cats)
            rating = float(b["stars"]) if b.get("stars") is not None else 3.5
            reviews = int(b.get("review_count") or 0)
            rid = f"r:{bid}"
            prices = [m["price"] for m in menu]
            meal = [sum(m["meal_affinity"][i] for m in menu) / len(menu) for i in range(len(MEAL_BUCKETS))]
            top = sorted(menu, key=lambda m: -m["mentions"])[:3]
            tag_counts = Counter(t for m in menu for t in m["tags"])
            common = {
                "restaurant_id": rid, "restaurant_name": b["name"], "chain": chain, "categories": cats,
                "price_tier": tier, "rating": rating, "review_count": reviews, "lat": lat, "lon": lon,
                "city": b.get("city", ""), "state": b.get("state", ""), "hours": hours,
                "delivery_polygon": polygon, "delivery_radius_km": radius, "source": b.get("source", "yelp"),
            }
            rdoc = dict(common, id=rid, type="restaurant", name=b["name"], canonical_dish="", cuisines=cuisines,
                        description=f"{b['name']} serves {', '.join(c.replace('_', ' ') for c in cuisines)} food. "
                                    f"Popular: {', '.join(m['name'] for m in top)}.",
                        tags=[t for t, _ in tag_counts.most_common(6)],
                        reviews=agg.snippet_text(review_chars) if agg else "", diet=_restaurant_diets(menu),
                        price=round(statistics.median(prices), 2), popularity=_popularity(reviews),
                        prep_time_min=round(statistics.median(m["prep_time_min"] for m in menu), 1),
                        meal_affinity=[round(x, 4) for x in meal])
            out.write(json.dumps(rdoc) + "\n")
            stats["restaurants"] += 1
            per_cuisine[cuisines[0]] += 1
            per_city[common["city"]] += 1
            if chain:
                chain_docs[chain] += 1
            for k, m in enumerate(menu):
                sentences = (agg.dish_sentences.get(m["canonical_dish"], []) if agg else [])
                ddoc = dict(common, id=f"d:{bid}:{k}", type="dish", name=m["name"],
                            canonical_dish=m["canonical_dish"],
                            cuisines=[m["cuisine"]] if m["cuisine"] != "generic" else cuisines[:1],
                            description=m["description"], tags=m["tags"], reviews=" ".join(sentences),
                            diet=m["diet"], price=m["price"],
                            popularity=_popularity(reviews, m["mentions"], is_dish=True),
                            prep_time_min=m["prep_time_min"], meal_affinity=[round(x, 4) for x in m["meal_affinity"]])
                if validate_every and stats["dishes"] % validate_every == 0:
                    problems = validate_doc(ddoc)
                    assert not problems, (ddoc["id"], problems)
                out.write(json.dumps(ddoc) + "\n")
                stats["dishes"] += 1
    stats_out = {
        "restaurants": stats["restaurants"], "dishes": stats["dishes"], "docs": stats["restaurants"] + stats["dishes"],
        "yelp_businesses": n_yelp, "osm_businesses": len(businesses) - n_yelp, "chains": len(chains),
        "per_cuisine": dict(per_cuisine.most_common()), "per_city": dict(per_city.most_common()),
        "chain_restaurants": dict(chain_docs.most_common(50)), "businesses_with_reviews": len(aggs),
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(out_path)), "catalog_stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2)
    return stats_out


# ---------------------------------------------------------------------- scale

def iter_restaurant_groups(path: str) -> Iterator[List[dict]]:
    """Yield each restaurant's docs (restaurant + dishes), relying on contiguity."""
    group: List[dict] = []
    cur = None
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d["restaurant_id"] != cur and group:
                yield group
                group = []
            cur = d["restaurant_id"]
            group.append(d)
    if group:
        yield group


def _count_docs(path: str) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def _clone_id(i: str, k: int) -> str:
    # "r:<bid>" -> "r:<bid>~k", "d:<bid>:<n>" -> "d:<bid>~k:<n>"
    parts = i.split(":")
    parts[1] = f"{parts[1]}~{k}"
    return ":".join(parts)


def scale_docs(in_path: str, out_path: str, target_docs: int, seed: int = 0) -> dict:
    """Grow a corpus by cloning whole restaurants into shifted synthetic metros."""
    n = _count_docs(in_path)
    copies = max(1, math.ceil(target_docs / max(n, 1)))
    written = 0
    rng = random.Random(seed)
    with open(out_path, "w") as out:
        for k in range(copies):
            dlat = 0.0 if k == 0 else 1.5 * k * (-1 if k % 2 else 1)
            dlon = 0.0 if k == 0 else 2.0 * k
            for group in iter_restaurant_groups(in_path):
                if written >= target_docs:
                    break
                rating_jitter = 0.0 if k == 0 else rng.uniform(-0.3, 0.3)
                pop_mult = 1.0 if k == 0 else rng.uniform(0.8, 1.2)
                price_mult = 1.0 if k == 0 else rng.uniform(0.9, 1.1)
                for d in group:
                    if k > 0:
                        d = dict(d)
                        d["id"] = _clone_id(d["id"], k)
                        d["restaurant_id"] = _clone_id(d["restaurant_id"], k)
                        d["lat"] = round(d["lat"] + dlat, 6)
                        d["lon"] = round(d["lon"] + dlon, 6)
                        d["delivery_polygon"] = [[round(p[0] + dlat, 6), round(p[1] + dlon, 6)]
                                                 for p in d["delivery_polygon"]]
                        d["city"] = f"{d['city']} Clone {k}"
                        d["rating"] = round(min(5.0, max(1.0, d["rating"] + rating_jitter)), 1)
                        d["popularity"] = round(min(1.0, d["popularity"] * pop_mult), 4)
                        d["price"] = round(d["price"] * price_mult, 2)
                        d["source"] = "synthetic"
                    out.write(json.dumps(d) + "\n")
                    written += 1
    return {"input_docs": n, "copies": copies, "written": written}


def subsample_docs(in_path: str, out_path: str, target_docs: int, seed: int = 0) -> dict:
    """Keep whole restaurants with probability target/n (deterministic per restaurant)."""
    n = _count_docs(in_path)
    p = min(1.0, target_docs / max(n, 1))
    written = 0
    with open(out_path, "w") as out:
        for group in iter_restaurant_groups(in_path):
            h = int(hashlib.md5(f"{seed}|{group[0]['restaurant_id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            if h < p:
                for d in group:
                    out.write(json.dumps(d) + "\n")
                    written += 1
    return {"input_docs": n, "keep_prob": p, "written": written}
