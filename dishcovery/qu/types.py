"""Output contract of query understanding (consumed by the broker)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

INTENTS = ("dish", "restaurant", "cuisine", "chain", "attribute")


@dataclass
class Entity:
    start: int            # token span [start, end) over StructuredQuery.corrected tokens
    end: int
    text: str
    type: str             # dish | cuisine | chain | restaurant | ingredient | attribute
    canonical: str        # canonical key, e.g. "chicken biryani", "indian", "dominos pizza"
    score: float = 1.0


@dataclass
class QueryTerm:
    """A weighted surface term. The engine runs each field's analyzer over ``text``."""
    text: str
    weight: float = 1.0
    group: int = 0        # which original content token/entity this term expands
    source: str = "query"  # query | spell | segment | synonym | entity


@dataclass
class QueryFilters:
    diet: List[str] = field(default_factory=list)
    price: Optional[str] = None          # low | mid | high
    price_tier_max: Optional[int] = None
    price_tier_min: Optional[int] = None
    max_price: Optional[float] = None
    open_now: bool = False
    min_rating: Optional[float] = None
    chain: Optional[str] = None          # hard chain filter (brand key)
    doc_type: Optional[str] = None       # dish | restaurant
    cuisine: Optional[str] = None        # soft signal (boost), never a hard filter


@dataclass
class StructuredQuery:
    raw: str
    normalized: str = ""
    corrected: str = ""
    tokens: List[str] = field(default_factory=list)      # content tokens (attributes removed)
    terms: List[QueryTerm] = field(default_factory=list)  # expanded weighted terms
    intent: str = "dish"
    intent_scores: Dict[str, float] = field(default_factory=dict)
    entities: List[Entity] = field(default_factory=list)
    filters: QueryFilters = field(default_factory=QueryFilters)
    geo: str = "user"                    # user | none
    radius_km: Optional[float] = None
    sort_hint: Optional[str] = None      # rating | fast | price
    meal_time: Optional[str] = None      # breakfast | lunch | snack | dinner | late_night
    corrections: List[Tuple[str, str, str]] = field(default_factory=list)  # (from, to, kind)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def has_text(self) -> bool:
        return bool(self.terms)
