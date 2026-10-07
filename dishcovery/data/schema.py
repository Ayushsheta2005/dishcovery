"""The document contract shared by the data pipeline, the C++ indexer and the broker.

Every line of ``docs.jsonl`` is one JSON object with the fields below. Two
document types live in the same index:

* ``restaurant`` — one per business.
* ``dish``       — one per menu item; restaurant attributes are denormalised onto
  the dish so geo/hours/price filters work without a join.

The C++ engine reads the same field names (see ``configs/schema.json``).
"""
from __future__ import annotations

from typing import Any, Dict, List

DOC_TYPES = ("restaurant", "dish")

# Diet labels. "vegan" implies "veg"; producers must emit both.
DIETS = ("veg", "vegan", "non_veg", "egg", "gluten_free", "halal", "jain")

# Meal-time buckets used for meal_affinity (5 floats summing to 1).
MEAL_BUCKETS = ("breakfast", "lunch", "snack", "dinner", "late_night")

# Hours are a 336-bit bitmap: 7 days x 48 half-hour slots, Monday 00:00 = bit 0.
# Serialised as 84 lowercase hex chars; hex char i encodes bits 4i..4i+3 with
# bit 4i in the least-significant position of that nibble.
HOURS_SLOTS = 7 * 48
HOURS_HEX_LEN = HOURS_SLOTS // 4

REQUIRED_FIELDS: Dict[str, type] = {
    "id": str,                 # "r:<business_id>" or "d:<business_id>:<n>"
    "type": str,               # restaurant | dish
    "restaurant_id": str,      # "r:<business_id>" (== id for restaurants)
    "name": str,               # dish name or restaurant name
    "restaurant_name": str,
    "chain": str,              # normalised brand key ("dominos pizza") or ""
    "canonical_dish": str,     # taxonomy canonical dish ("chicken biryani") or ""
    "cuisines": list,          # taxonomy cuisine keys, e.g. ["indian"]
    "categories": list,        # raw Yelp-style categories
    "description": str,
    "tags": list,
    "reviews": str,            # review snippets (dish docs: sentences mentioning the dish)
    "diet": list,              # subset of DIETS
    "price": float,            # dish price (restaurants: median dish price)
    "price_tier": int,         # 1..4
    "rating": float,           # 0..5
    "review_count": int,
    "popularity": float,       # 0..1 prior
    "lat": float,
    "lon": float,
    "city": str,
    "state": str,
    "hours": str,              # HOURS_HEX_LEN hex chars
    "prep_time_min": float,
    "meal_affinity": list,     # 5 floats, MEAL_BUCKETS order
    "delivery_polygon": list,  # [[lat, lon], ...] simple polygon, not closed
    "delivery_radius_km": float,
    "source": str,             # yelp | osm | synthetic
}

# Fields returned to the broker per candidate (kept small: they ride in every
# phase-2 response).
SUMMARY_FIELDS = (
    "id", "type", "name", "restaurant_name", "restaurant_id", "chain",
    "canonical_dish", "cuisines", "diet", "price", "price_tier", "rating",
    "review_count", "city", "lat", "lon",
)


def validate_doc(doc: Dict[str, Any]) -> List[str]:
    """Return a list of problems (empty if the doc satisfies the contract)."""
    problems: List[str] = []
    for name, typ in REQUIRED_FIELDS.items():
        if name not in doc:
            problems.append(f"missing {name}")
            continue
        v = doc[name]
        if typ is float:
            if not isinstance(v, (int, float)):
                problems.append(f"{name} not numeric")
        elif not isinstance(v, typ):
            problems.append(f"{name} has type {type(v).__name__}, want {typ.__name__}")
    if doc.get("type") not in DOC_TYPES:
        problems.append("bad type")
    if isinstance(doc.get("hours"), str) and len(doc["hours"]) != HOURS_HEX_LEN:
        problems.append("hours wrong length")
    ma = doc.get("meal_affinity")
    if isinstance(ma, list) and len(ma) != len(MEAL_BUCKETS):
        problems.append("meal_affinity wrong length")
    for d in doc.get("diet", []) or []:
        if d not in DIETS:
            problems.append(f"unknown diet {d}")
    poly = doc.get("delivery_polygon")
    if isinstance(poly, list) and poly and len(poly) < 3:
        problems.append("delivery_polygon needs >= 3 vertices")
    return problems


def hours_bit(hours_hex: str, slot: int) -> bool:
    """True if the restaurant is open in half-hour ``slot`` (0..335)."""
    nib = int(hours_hex[slot // 4], 16)
    return bool((nib >> (slot % 4)) & 1)


def hours_from_slots(open_slots) -> str:
    bits = [0] * HOURS_SLOTS
    for s in open_slots:
        bits[s % HOURS_SLOTS] = 1
    out = []
    for i in range(HOURS_HEX_LEN):
        nib = bits[4 * i] | (bits[4 * i + 1] << 1) | (bits[4 * i + 2] << 2) | (bits[4 * i + 3] << 3)
        out.append("0123456789abcdef"[nib])
    return "".join(out)


def time_slot(day_of_week: int, hour: int, minute: int) -> int:
    """Monday=0. Returns the half-hour slot index 0..335."""
    return (day_of_week % 7) * 48 + hour * 2 + (1 if minute >= 30 else 0)
