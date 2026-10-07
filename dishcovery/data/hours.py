"""Opening hours <-> the 336-slot bitmap of the docs schema."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set

from dishcovery.data.schema import HOURS_SLOTS, hours_from_slots

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_OSM_DAYS = {"mo": 0, "tu": 1, "we": 2, "th": 3, "fr": 4, "sa": 5, "su": 6}


def _slot(day: int, minutes: int) -> int:
    return (day % 7) * 48 + minutes // 30


def _parse_hm(s: str) -> Optional[int]:
    m = re.match(r"^\s*(\d{1,2})[:.](\d{1,2})\s*$", s)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 24 or mi > 59:
        return None
    return h * 60 + mi


def interval_slots(day: int, start_min: int, end_min: int) -> Set[int]:
    """Half-hour slots open between start and end on `day`; end <= start wraps past midnight."""
    if end_min == start_min:  # "0:0-0:0" = open 24h
        return {_slot(day, m) for m in range(0, 24 * 60, 30)}
    total = end_min - start_min if end_min > start_min else 24 * 60 - start_min + end_min
    slots = set()
    m = start_min - start_min % 30
    while m < start_min + total:
        slots.add(_slot(day + m // (24 * 60), m % (24 * 60)) % HOURS_SLOTS)
        m += 30
    return slots


def yelp_hours_to_hex(hours: Optional[Dict[str, str]]) -> Optional[str]:
    """{"Monday": "11:0-22:0", ...} -> hex bitmap; None if missing/unparseable."""
    if not hours:
        return None
    slots: Set[int] = set()
    for day_name, span in hours.items():
        if day_name not in DAYS or not isinstance(span, str) or "-" not in span:
            continue
        a, b = span.split("-", 1)
        s, e = _parse_hm(a), _parse_hm(b)
        if s is None or e is None:
            continue
        slots |= interval_slots(DAYS.index(day_name), s, e)
    return hours_from_slots(slots) if slots else None


def osm_hours_to_hex(spec: Optional[str]) -> Optional[str]:
    """Common OSM opening_hours forms: "Mo-Fr 11:00-22:00; Sa,Su 12:00-23:00", "24/7"."""
    if not spec:
        return None
    spec = spec.strip()
    if spec == "24/7":
        return hours_from_slots(range(HOURS_SLOTS))
    slots: Set[int] = set()
    for rule in spec.split(";"):
        rule = rule.strip()
        m = re.match(r"^([A-Za-z,\- ]+)\s+([\d:,\-\s]+)$", rule)
        if not m:
            continue
        days: List[int] = []
        for part in m.group(1).replace(" ", "").lower().split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                if a[:2] in _OSM_DAYS and b[:2] in _OSM_DAYS:
                    i, j = _OSM_DAYS[a[:2]], _OSM_DAYS[b[:2]]
                    days += [(i + k) % 7 for k in range(((j - i) % 7) + 1)]
            elif part[:2] in _OSM_DAYS:
                days.append(_OSM_DAYS[part[:2]])
        for span in m.group(2).split(","):
            if "-" not in span:
                continue
            a, b = span.split("-", 1)
            s, e = _parse_hm(a), _parse_hm(b)
            if s is None or e is None:
                continue
            for d in days:
                slots |= interval_slots(d, s, e % (24 * 60))
    return hours_from_slots(slots) if slots else None


def default_hours(categories: Iterable[str]) -> str:
    """Category-typical schedule when the source has none."""
    cats = {c.lower() for c in categories}
    if cats & {"coffee & tea", "cafes", "bakeries", "breakfast & brunch", "donuts", "bagels"}:
        start, end = 7 * 60, 15 * 60
    elif cats & {"bars", "nightlife", "pubs", "sports bars", "cocktail bars"}:
        start, end = 16 * 60, 2 * 60
    elif cats & {"fast food", "pizza", "chicken wings", "hot dogs"}:
        start, end = 10 * 60 + 30, 23 * 60
    else:
        start, end = 11 * 60, 22 * 60
    slots: Set[int] = set()
    for d in range(7):
        slots |= interval_slots(d, start, end)
    return hours_from_slots(slots)
