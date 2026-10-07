"""Streaming loaders for the Yelp Open Dataset.

The review file is ~5 GB / ~7M lines, so aggregation is one pass, split into
byte ranges processed by a process pool; each worker keeps bounded state per
business (top-k snippets, dish mention counters, a few dish sentences).
"""
from __future__ import annotations

import heapq
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Dict, Iterator, List, Optional, Set, Tuple

try:
    import orjson as _json

    def _loads(b):
        return _json.loads(b)
except ImportError:  # pragma: no cover
    def _loads(b):
        return json.loads(b)

FOOD_CATEGORIES = {
    "restaurants", "food", "cafes", "bakeries", "desserts", "coffee & tea", "fast food", "food trucks",
    "pizza", "sandwiches", "breakfast & brunch", "ice cream & frozen yogurt", "juice bars & smoothies",
    "donuts", "bagels", "delis", "food stands", "street vendors", "bubble tea", "food court",
}
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                yield _loads(line)


def is_food_business(b: dict) -> bool:
    cats = b.get("categories") or ""
    return any(c.strip().lower() in FOOD_CATEGORIES for c in cats.split(","))


def load_businesses(path: str, food_only: bool = True, include_closed: bool = False,
                    max_businesses: Optional[int] = None) -> List[dict]:
    out = []
    for b in iter_jsonl(path):
        if food_only and not is_food_business(b):
            continue
        if not include_closed and b.get("is_open") == 0:
            continue
        if b.get("latitude") is None or b.get("longitude") is None:
            continue
        b["categories_list"] = [c.strip() for c in (b.get("categories") or "").split(",") if c.strip()]
        out.append(b)
        if max_businesses and len(out) >= max_businesses:
            break
    return out


@dataclass
class ReviewAggregate:
    snippets: List[Tuple[float, str]] = field(default_factory=list)   # heap of (score, text)
    dish_mentions: Dict[str, int] = field(default_factory=dict)
    dish_stars: Dict[str, float] = field(default_factory=dict)
    dish_sentences: Dict[str, List[str]] = field(default_factory=dict)
    n_reviews: int = 0

    def merge(self, o: "ReviewAggregate", max_snippets: int, max_sent: int) -> None:
        for s in o.snippets:
            heapq.heappush(self.snippets, s)
            if len(self.snippets) > max_snippets:
                heapq.heappop(self.snippets)
        for d, c in o.dish_mentions.items():
            self.dish_mentions[d] = self.dish_mentions.get(d, 0) + c
            self.dish_stars[d] = self.dish_stars.get(d, 0.0) + o.dish_stars.get(d, 0.0)
        for d, ss in o.dish_sentences.items():
            cur = self.dish_sentences.setdefault(d, [])
            cur.extend(ss[: max(0, max_sent - len(cur))])
        self.n_reviews += o.n_reviews

    def mean_dish_stars(self, d: str) -> float:
        c = self.dish_mentions.get(d, 0)
        return self.dish_stars.get(d, 0.0) / c if c else 0.0

    def snippet_text(self, max_chars: int) -> str:
        best = sorted(self.snippets, reverse=True)
        out, n = [], 0
        for _, t in best:
            if n + len(t) > max_chars:
                break
            out.append(t)
            n += len(t) + 1
        return " ".join(out)


def _worker(args) -> Dict[str, ReviewAggregate]:
    path, start, stop, ids, taxonomy_path, max_snippets, snippet_chars, max_sent = args
    from dishcovery.data.taxonomy import Taxonomy

    matcher = Taxonomy.load(taxonomy_path).dish_matcher()
    aggs: Dict[str, ReviewAggregate] = defaultdict(ReviewAggregate)
    with open(path, "rb") as f:
        # Process exactly the lines that *start* in [start, stop). Seeking to
        # start-1 and discarding through the next newline lands on the first
        # line starting at or after `start` (a line straddling the boundary
        # belongs to the previous range).
        if start:
            f.seek(start - 1)
            f.readline()
        while f.tell() < stop:
            line = f.readline()
            if not line:
                break
            try:
                r = _loads(line)
            except Exception:
                continue
            bid = r.get("business_id")
            if bid not in ids:
                continue
            a = aggs[bid]
            a.n_reviews += 1
            text = (r.get("text") or "").replace("\n", " ")
            stars = float(r.get("stars") or 3)
            score = float(r.get("useful") or 0) + 0.3 * stars + (0.5 if (r.get("date") or "") >= "2019" else 0)
            item = (score, text[:snippet_chars])
            if len(a.snippets) < max_snippets:
                heapq.heappush(a.snippets, item)
            elif item > a.snippets[0]:
                heapq.heapreplace(a.snippets, item)
            mentions = matcher.find(text)
            if not mentions:
                continue
            sentences = None
            for d in set(m[0] for m in mentions):
                a.dish_mentions[d] = a.dish_mentions.get(d, 0) + 1
                a.dish_stars[d] = a.dish_stars.get(d, 0.0) + stars
                ss = a.dish_sentences.setdefault(d, [])
                if len(ss) < max_sent:
                    if sentences is None:
                        sentences = _SENT_RE.split(text)
                    for s in sentences:
                        if len(ss) >= max_sent:
                            break
                        if d.split()[-1] in s.lower():
                            ss.append(s[:200])
    return dict(aggs)


def aggregate_reviews(path: str, business_ids: Set[str], taxonomy_path: Optional[str] = None,
                      max_snippets: int = 5, snippet_chars: int = 300, max_dish_sentences: int = 2,
                      workers: Optional[int] = None) -> Dict[str, ReviewAggregate]:
    size = os.path.getsize(path)
    workers = max(1, workers or os.cpu_count() or 1)
    n_chunks = max(1, min(workers * 4, size // (8 << 20) + 1))
    bounds = [size * i // n_chunks for i in range(n_chunks + 1)]
    tasks = [(path, bounds[i], bounds[i + 1], business_ids, taxonomy_path,
              max_snippets, snippet_chars, max_dish_sentences) for i in range(n_chunks)]
    out: Dict[str, ReviewAggregate] = {}
    if workers == 1 or n_chunks == 1:
        parts = map(_worker, tasks)
    else:
        pool = Pool(workers)
        parts = pool.imap_unordered(_worker, tasks)
    for part in parts:
        for bid, agg in part.items():
            if bid in out:
                out[bid].merge(agg, max_snippets, max_dish_sentences)
            else:
                out[bid] = agg
    if workers != 1 and n_chunks != 1:
        pool.close()
        pool.join()
    return out
