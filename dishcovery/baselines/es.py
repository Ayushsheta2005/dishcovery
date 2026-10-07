"""Elasticsearch baselines (the comparison a reviewer will trust).

* ``es_oob`` — out of the box: default ``standard`` analyzer on every text
  field, a ``multi_match`` (best_fields) over the raw query text, BM25 defaults.
* ``es_qu``  — the same Elasticsearch index, but queried with our query
  understanding output (corrected/expanded terms with weights, hard filters),
  which separates "better QU" from "better engine".

Both apply the delivery-zone constraint as a ``geo_shape`` filter on the same
polygons our engine indexes, so every system ranks the same admissible set.
Plain HTTP via urllib: no client library to pin to a server version.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Iterable, List, Optional

TEXT_FIELDS = ["name", "restaurant_name", "cuisines", "categories", "tags", "description", "reviews"]

MAPPING = {
    "settings": {"number_of_shards": 4, "number_of_replicas": 0, "refresh_interval": "30s"},
    "mappings": {"properties": {
        **{f: {"type": "text"} for f in TEXT_FIELDS},
        "id": {"type": "keyword"}, "type": {"type": "keyword"}, "diet": {"type": "keyword"},
        "chain": {"type": "keyword"}, "restaurant_id": {"type": "keyword"}, "city": {"type": "keyword"},
        "canonical_dish": {"type": "keyword"},
        "price": {"type": "float"}, "price_tier": {"type": "integer"}, "rating": {"type": "float"},
        "popularity": {"type": "float"}, "open_slots": {"type": "short"},
        "location": {"type": "geo_point"}, "delivery_zone": {"type": "geo_shape"},
    }},
}


def _open_slots(hours_hex: str) -> List[int]:
    out = []
    for i, ch in enumerate(hours_hex or ""):
        nib = int(ch, 16)
        for b in range(4):
            if (nib >> b) & 1:
                out.append(4 * i + b)
    return out


def es_doc(d: dict) -> dict:
    poly = [[p[1], p[0]] for p in d.get("delivery_polygon") or []]
    if poly:
        poly.append(poly[0])
    out = {f: d.get(f) for f in TEXT_FIELDS}
    out.update({k: d.get(k) for k in ("id", "type", "diet", "chain", "restaurant_id", "city", "canonical_dish",
                                       "price", "price_tier", "rating", "popularity")})
    out["location"] = {"lat": d["lat"], "lon": d["lon"]}
    if len(poly) >= 4:
        out["delivery_zone"] = {"type": "polygon", "coordinates": [poly]}
    out["open_slots"] = _open_slots(d.get("hours", ""))
    return out


class ElasticBaseline:
    def __init__(self, url: str = "http://localhost:9200", index: str = "dishcovery", timeout_s: float = 30) -> None:
        self.url = url.rstrip("/")
        self.index = index
        self.timeout = timeout_s

    def _req(self, method: str, path: str, body=None, ndjson: bool = False):
        data = None
        headers = {"Content-Type": "application/x-ndjson" if ndjson else "application/json"}
        if body is not None:
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ES {method} {path}: {e.code} {e.read()[:500]!r}") from e

    def wait(self, timeout_s: float = 120) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                self._req("GET", "/_cluster/health?wait_for_status=yellow&timeout=5s")
                return
            except Exception:
                time.sleep(2)
        raise RuntimeError(f"Elasticsearch at {self.url} not reachable")

    def create(self, recreate: bool = True) -> None:
        if recreate:
            try:
                self._req("DELETE", f"/{self.index}")
            except RuntimeError:
                pass
        self._req("PUT", f"/{self.index}", MAPPING)

    def bulk_index(self, docs: Iterable[dict], batch: int = 2000, log=print) -> dict:
        t0 = time.time()
        n, errors, buf = 0, 0, []

        def flush():
            nonlocal errors
            if not buf:
                return
            r = self._req("POST", f"/{self.index}/_bulk", "\n".join(buf) + "\n", ndjson=True)
            if r.get("errors"):
                errors += sum(1 for it in r["items"] if "error" in it.get("index", {}))
            buf.clear()

        for d in docs:
            buf.append(json.dumps({"index": {"_id": d["id"]}}))
            buf.append(json.dumps(es_doc(d)))
            n += 1
            if n % batch == 0:
                flush()
                if n % (batch * 50) == 0:
                    log(f"ES indexed {n} docs ({n / (time.time() - t0):.0f}/s)")
        flush()
        self._req("POST", f"/{self.index}/_refresh")
        self._req("PUT", f"/{self.index}/_settings", {"index": {"refresh_interval": "1s"}})
        stats = self._req("GET", f"/{self.index}/_stats/store,docs")
        prim = stats["_all"]["primaries"]
        return {"docs": n, "errors": errors, "seconds": round(time.time() - t0, 1),
                "store_bytes": prim["store"]["size_in_bytes"], "es_docs": prim["docs"]["count"]}

    def _geo_filter(self, lat: Optional[float], lon: Optional[float]) -> List[dict]:
        if lat is None or lon is None:
            return []
        return [{"geo_shape": {"delivery_zone": {"shape": {"type": "point", "coordinates": [lon, lat]},
                                                  "relation": "intersects"}}}]

    def query_body(self, text: str, lat: Optional[float], lon: Optional[float], k: int,
                   structured=None, when: Optional[str] = None) -> dict:
        filters = self._geo_filter(lat, lon)
        if structured is None:
            must = [{"multi_match": {"query": text, "fields": TEXT_FIELDS}}] if text.strip() else [{"match_all": {}}]
            return {"size": k, "_source": ["id"], "query": {"bool": {"must": must, "filter": filters}}}
        should = []
        for t in structured.terms:
            should.append({"multi_match": {"query": t.text, "fields": TEXT_FIELDS, "boost": max(t.weight, 0.01)}})
        f = structured.filters
        if f.diet:
            filters.append({"terms": {"diet": f.diet}})
        if f.chain:
            filters.append({"term": {"chain": f.chain}})
        if f.doc_type:
            filters.append({"term": {"type": f.doc_type}})
        rng = {}
        if f.price_tier_max is not None:
            rng.setdefault("price_tier", {})["lte"] = f.price_tier_max
        if f.price_tier_min is not None:
            rng.setdefault("price_tier", {})["gte"] = f.price_tier_min
        if f.max_price is not None:
            rng["price"] = {"lte": f.max_price}
        if f.min_rating is not None:
            rng["rating"] = {"gte": f.min_rating}
        filters += [{"range": {k2: v}} for k2, v in rng.items()]
        if f.open_now and when:
            from dishcovery.data.schema import time_slot
            from dishcovery.serving.broker import parse_when
            filters.append({"term": {"open_slots": time_slot(*parse_when(when))}})
        q = {"bool": {"should": should, "minimum_should_match": 1 if should else 0, "filter": filters}}
        return {"size": k, "_source": ["id"], "query": q}

    def search(self, text: str, lat: Optional[float], lon: Optional[float], k: int = 100, structured=None,
               when: Optional[str] = None) -> List[str]:
        r = self._req("POST", f"/{self.index}/_search", self.query_body(text, lat, lon, k, structured, when))
        return [h["_source"]["id"] for h in r["hits"]["hits"]]

    def search_timed(self, text: str, lat, lon, k: int = 20) -> float:
        r = self._req("POST", f"/{self.index}/_search", self.query_body(text, lat, lon, k))
        return float(r.get("took", 0))
