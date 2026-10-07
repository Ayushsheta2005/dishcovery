"""Geometry helpers: distances and synthetic delivery zones."""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, List, Sequence, Tuple

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def destination(lat: float, lon: float, bearing_rad: float, dist_km: float) -> Tuple[float, float]:
    d = dist_km / EARTH_RADIUS_KM
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(bearing_rad))
    l2 = l1 + math.atan2(math.sin(bearing_rad) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def point_in_polygon(lat: float, lon: float, poly: Sequence[Sequence[float]]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        yi, xi = poly[i][0], poly[i][1]
        yj, xj = poly[j][0], poly[j][1]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def polygon_area_km2(poly: Sequence[Sequence[float]]) -> float:
    if len(poly) < 3:
        return 0.0
    lat0 = sum(p[0] for p in poly) / len(poly)
    kx = 111.32 * math.cos(math.radians(lat0))
    pts = [(p[1] * kx, p[0] * 110.57) for p in poly]
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


class DensityGrid:
    """Coarse ~1 km grid counts used to shrink delivery zones in dense areas."""

    def __init__(self, points: Sequence[Tuple[float, float]], cell_deg: float = 0.01) -> None:
        self.cell = cell_deg
        self.counts: Counter = Counter(self._key(lat, lon) for lat, lon in points)

    def _key(self, lat: float, lon: float) -> Tuple[int, int]:
        return int(math.floor(lat / self.cell)), int(math.floor(lon / self.cell))

    def density(self, lat: float, lon: float, radius_cells: int = 2) -> int:
        a, b = self._key(lat, lon)
        return sum(self.counts.get((a + i, b + j), 0)
                   for i in range(-radius_cells, radius_cells + 1) for j in range(-radius_cells, radius_cells + 1))


def delivery_radius_km(density: int, price_tier: int, rng: random.Random) -> float:
    """2-8 km: dense neighbourhoods and pricier places deliver less far."""
    base = 7.5 - 1.6 * math.log1p(density / 10.0)
    base -= 0.4 * max(0, price_tier - 2)
    return round(min(8.0, max(2.0, base + rng.uniform(-0.8, 0.8))), 2)


def delivery_polygon(lat: float, lon: float, radius_km: float, rng: random.Random) -> List[List[float]]:
    """Irregular star-convex zone: 10-16 vertices, radial noise ±25%."""
    n = rng.randint(10, 16)
    out = []
    offset = rng.uniform(0, 2 * math.pi)
    for i in range(n):
        bearing = offset + 2 * math.pi * i / n
        r = radius_km * rng.uniform(0.75, 1.25)
        la, lo = destination(lat, lon, bearing, r)
        out.append([round(la, 6), round(lo, 6)])
    return out
