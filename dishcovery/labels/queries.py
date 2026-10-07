"""Evaluation queries: the hand-labelled test set's query side.

Each query carries the user context it is issued in — a location inside one of
the indexed metros and a local time like ``"tue 13:00"`` — because relevance in
food search depends on it (a closed or non-delivering restaurant cannot be a
grade-3 answer). The 50 queries live in ``data/eval/queries.jsonl``, 5 per
class in :data:`QUERY_CLASSES`.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Iterable, List, Tuple, Union

from dishcovery.data.schema import time_slot
from dishcovery.eval.io import read_jsonl, write_jsonl

QUERY_CLASSES = (
    "exact_dish",
    "misspelled",
    "compound",
    "chain",
    "restaurant_name",
    "cuisine",
    "attribute",
    "synonym_multilingual",
    "vague_semantic",
    "dietary_structured",
)

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

DEFAULT_QUERIES_PATH = os.path.join("data", "eval", "queries.jsonl")


def parse_time(s: str) -> Tuple[int, int, int]:
    """``"tue 13:00" -> (1, 13, 0)``; Monday = 0, 24-hour clock.

    Accepts full day names (``"Tuesday 9:05"``) and is case-insensitive.
    Raises ``ValueError`` on anything else.
    """
    parts = s.strip().lower().split()
    if len(parts) != 2:
        raise ValueError(f"time must look like 'tue 13:00', got {s!r}")
    day, hm = parts
    dow = next((i for i, d in enumerate(DAYS) if day.startswith(d)), None)
    if dow is None:
        raise ValueError(f"bad day of week in {s!r}")
    try:
        h_str, m_str = hm.split(":")
        hour, minute = int(h_str), int(m_str)
    except ValueError:
        raise ValueError(f"bad HH:MM in {s!r}") from None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"time out of range in {s!r}")
    return dow, hour, minute


def format_time(dow: int, hour: int, minute: int) -> str:
    """Inverse of :func:`parse_time`: ``(1, 13, 0) -> "tue 13:00"``."""
    return f"{DAYS[dow % 7]} {hour:02d}:{minute:02d}"


@dataclass
class EvalQuery:
    """One evaluation query plus the context it is issued in."""
    qid: str
    text: str
    query_class: str
    metro: str
    lat: float
    lon: float
    time: str          # "tue 13:00": day-of-week + local 24h time
    notes: str = ""

    def __post_init__(self) -> None:
        if self.query_class not in QUERY_CLASSES:
            raise ValueError(f"{self.qid}: unknown query_class {self.query_class!r}")
        parse_time(self.time)  # validate early

    @property
    def dow_hour_minute(self) -> Tuple[int, int, int]:
        return parse_time(self.time)

    @property
    def slot(self) -> int:
        """Half-hour slot (0..335) of the query time, for ``schema.hours_bit``."""
        return time_slot(*parse_time(self.time))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalQuery":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def load_queries(path: Union[str, "os.PathLike[str]"] = DEFAULT_QUERIES_PATH) -> List[EvalQuery]:
    """Load evaluation queries from JSONL (file order is preserved)."""
    queries = [EvalQuery.from_dict(r) for r in read_jsonl(path)]
    seen = set()
    for q in queries:
        if q.qid in seen:
            raise ValueError(f"duplicate qid {q.qid}")
        seen.add(q.qid)
    return queries


def save_queries(path: Union[str, "os.PathLike[str]"], queries: Iterable[EvalQuery]) -> int:
    return write_jsonl(path, (q.to_dict() for q in queries))


def qid_to_class(queries: Iterable[EvalQuery]) -> dict:
    """``qid -> query_class`` for :func:`dishcovery.eval.metrics.per_class`."""
    return {q.qid: q.query_class for q in queries}
