"""Metro areas covered by the Yelp Open Dataset (and mirrored by the synthetic generator).

Both the data pipeline and the evaluation query set use these centres so that
eval queries issued "from Philadelphia" land inside the indexed corpus.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Metro:
    key: str
    city: str
    state: str
    lat: float
    lon: float
    radius_km: float   # rough extent of the metro's restaurant footprint
    weight: float      # share of restaurants in the synthetic generator


METROS: Dict[str, Metro] = {
    m.key: m
    for m in [
        Metro("philadelphia", "Philadelphia", "PA", 39.9526, -75.1652, 18.0, 0.20),
        Metro("tampa", "Tampa", "FL", 27.9506, -82.4572, 22.0, 0.12),
        Metro("indianapolis", "Indianapolis", "IN", 39.7684, -86.1581, 20.0, 0.10),
        Metro("nashville", "Nashville", "TN", 36.1627, -86.7816, 18.0, 0.10),
        Metro("tucson", "Tucson", "AZ", 32.2226, -110.9747, 18.0, 0.09),
        Metro("new_orleans", "New Orleans", "LA", 29.9511, -90.0715, 14.0, 0.09),
        Metro("edmonton", "Edmonton", "AB", 53.5461, -113.4938, 16.0, 0.07),
        Metro("saint_louis", "Saint Louis", "MO", 38.6270, -90.1994, 18.0, 0.08),
        Metro("reno", "Reno", "NV", 39.5296, -119.8138, 12.0, 0.06),
        Metro("boise", "Boise", "ID", 43.6150, -116.2023, 12.0, 0.05),
        Metro("santa_barbara", "Santa Barbara", "CA", 34.4208, -119.6982, 10.0, 0.04),
    ]
}


def nearest_metro(lat: float, lon: float) -> Metro:
    best, best_d = None, float("inf")
    for m in METROS.values():
        d = (m.lat - lat) ** 2 + (m.lon - lon) ** 2
        if d < best_d:
            best, best_d = m, d
    assert best is not None
    return best
