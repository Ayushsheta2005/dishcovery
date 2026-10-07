"""OpenStreetMap POIs to densify the Yelp catalogue.

``fetch_osm_pois`` queries the public Overpass API per metro (needs network;
run it on the external machine), ``osm_to_businesses`` converts elements to
Yelp-like business records, and ``dedupe_against`` drops OSM places that
duplicate a Yelp business (same name within ~60 m), using a grid so it is not
O(n^2).
"""
from __future__ import annotations

import difflib
import json
import math
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

from dishcovery.data.cities import METROS, Metro
from dishcovery.data.geo import haversine_km
from dishcovery.data.hours import osm_hours_to_hex
from dishcovery.qu.text import normalize

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
AMENITIES = "restaurant|fast_food|cafe|food_court|ice_cream"

# OSM cuisine tag -> Yelp-style category (consumed by Taxonomy.cuisines_for_categories).
CUISINE_TO_CATEGORY = {
    "indian": "Indian", "pakistani": "Pakistani", "nepalese": "Himalayan/Nepalese", "chinese": "Chinese",
    "japanese": "Japanese", "sushi": "Sushi Bars", "ramen": "Ramen", "korean": "Korean", "thai": "Thai",
    "vietnamese": "Vietnamese", "italian": "Italian", "pizza": "Pizza", "burger": "Burgers",
    "american": "American (Traditional)", "barbecue": "Barbeque", "bbq": "Barbeque", "mexican": "Mexican",
    "tex-mex": "Tex-Mex", "greek": "Greek", "mediterranean": "Mediterranean", "lebanese": "Lebanese",
    "turkish": "Turkish", "kebab": "Kebab", "middle_eastern": "Middle Eastern", "french": "French",
    "spanish": "Spanish", "tapas": "Tapas Bars", "caribbean": "Caribbean", "cuban": "Cuban",
    "jamaican": "Jamaican", "cajun": "Cajun/Creole", "ethiopian": "Ethiopian", "breakfast": "Breakfast & Brunch",
    "coffee_shop": "Coffee & Tea", "donut": "Donuts", "ice_cream": "Ice Cream & Frozen Yogurt",
    "seafood": "Seafood", "fish_and_chips": "Fish & Chips", "sandwich": "Sandwiches", "deli": "Delis",
    "salad": "Salad", "vegan": "Vegan", "vegetarian": "Vegetarian", "chicken": "Chicken Wings",
    "wings": "Chicken Wings", "bagel": "Bagels", "dessert": "Desserts", "bakery": "Bakeries",
}


def _query(m: Metro, radius_km: Optional[float] = None) -> str:
    r = int((radius_km or m.radius_km) * 1000)
    return (f"[out:json][timeout:180];(node(around:{r},{m.lat},{m.lon})[amenity~\"^({AMENITIES})$\"];"
            f"way(around:{r},{m.lat},{m.lon})[amenity~\"^({AMENITIES})$\"];);out center tags;")


def fetch_osm_pois(out_path: str, metros: Optional[Sequence[str]] = None, timeout: int = 240,
                   pause_s: float = 5.0, retries: int = 3) -> int:
    """Download POIs for each metro into a JSON-lines file of raw Overpass elements."""
    keys = list(metros) if metros else list(METROS)
    n = 0
    with open(out_path, "w") as out:
        for key in keys:
            m = METROS[key]
            data = urllib.parse.urlencode({"data": _query(m)}).encode()
            for attempt in range(retries):
                try:
                    req = urllib.request.Request(OVERPASS_URL, data=data,
                                                 headers={"User-Agent": "dishcovery-research/0.1"})
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        payload = json.loads(resp.read())
                    for el in payload.get("elements", []):
                        el["_metro"] = key
                        out.write(json.dumps(el) + "\n")
                        n += 1
                    break
                except Exception as e:  # network errors, 429 rate limits
                    wait = pause_s * (2 ** attempt)
                    print(f"overpass {key} attempt {attempt + 1} failed: {e}; retrying in {wait:.0f}s")
                    time.sleep(wait)
            time.sleep(pause_s)
    return n


def osm_to_businesses(path: str) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            el = json.loads(line)
            tags = el.get("tags", {})
            name = tags.get("name")
            if not name:
                continue
            lat = el.get("lat") or (el.get("center") or {}).get("lat")
            lon = el.get("lon") or (el.get("center") or {}).get("lon")
            if lat is None or lon is None:
                continue
            cats = []
            for c in (tags.get("cuisine") or "").split(";"):
                c = c.strip().lower()
                if c in CUISINE_TO_CATEGORY:
                    cats.append(CUISINE_TO_CATEGORY[c])
            amenity = tags.get("amenity", "restaurant")
            if amenity == "cafe" and "Coffee & Tea" not in cats:
                cats.append("Coffee & Tea")
            if amenity == "fast_food":
                cats.append("Fast Food")
            if amenity == "ice_cream":
                cats.append("Ice Cream & Frozen Yogurt")
            cats.append("Restaurants")
            m = METROS[el["_metro"]]
            out.append({
                "business_id": f"osm{el['type'][0]}{el['id']}", "name": name, "city": tags.get("addr:city", m.city),
                "state": m.state, "latitude": float(lat), "longitude": float(lon),
                "stars": None, "review_count": 0, "is_open": 1,
                "attributes": {}, "categories": ", ".join(dict.fromkeys(cats)),
                "categories_list": list(dict.fromkeys(cats)),
                "hours_hex": osm_hours_to_hex(tags.get("opening_hours")), "hours": None, "source": "osm",
            })
    return out


def _cell(lat: float, lon: float, size: float = 0.002):
    return int(math.floor(lat / size)), int(math.floor(lon / size))


def dedupe_against(yelp: Iterable[dict], osm: Iterable[dict], max_dist_m: float = 60.0,
                   name_sim: float = 0.8) -> List[dict]:
    grid: Dict[tuple, List[dict]] = defaultdict(list)
    for b in yelp:
        grid[_cell(b["latitude"], b["longitude"])].append(b)
    kept = []
    for o in osm:
        a, c = _cell(o["latitude"], o["longitude"])
        on = normalize(o["name"])
        dup = False
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for b in grid.get((a + i, c + j), ()):
                    if haversine_km(o["latitude"], o["longitude"], b["latitude"], b["longitude"]) * 1000 > max_dist_m:
                        continue
                    if difflib.SequenceMatcher(None, on, normalize(b["name"])).ratio() >= name_sim:
                        dup = True
                        break
                if dup:
                    break
            if dup:
                break
        if not dup:
            kept.append(o)
    return kept
