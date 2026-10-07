"""Weak-supervision relevance labels (qrels ``source = "weak"``).

These heuristic grades are TRAINING labels for LambdaMART. They are never used
to report NDCG — only hand labels are (see ``data/eval/LABELING_GUIDELINES.md``,
whose 0–3 scale this heuristic imitates).

Definition of the weak grade ``g(query, doc, ctx) ∈ {0,1,2,3}``
------------------------------------------------------------------
The query text is normalised (``dishcovery.qu.text.normalize``) and split into

* **constraints** — diets (``vegan``, ``veg``/``vegetarian``, ``gluten free``,
  ``halal``, ``jain``), price (``cheap``, ``under $N``, ``upscale``), and
  attributes (``open now``, ``late night``, ``top rated``, ``fast delivery``,
  meal words ``breakfast``/``brunch``/``lunch``/``dinner``);
* **content tokens** — everything else minus stop/generic words ("near me",
  "food", "restaurants", ...). Content is further split into cuisine phrases
  ``C`` (taxonomy cuisine keys) and dish tokens ``D``. Tokens are compared after
  light singularisation ("tacos" ~ "taco").

1. **Serviceability** (needs ``ctx``): if the user's lat/lon lies outside the
   doc's ``delivery_polygon`` (or beyond ``delivery_radius_km`` when there is
   no polygon) → **0**. If the doc is closed in the half-hour slot of the
   query time → grade capped at **1**. All-zero hours are treated as unknown.
2. **Navigational targets.** Chain (from ``target_type="chain"`` or the text
   matching a chain alias such as "dominos", "pizzahut"): doc of that chain →
   3 (restaurant) / 2 (its dishes); other docs sharing the chain's cuisines →
   1; else 0. Restaurant name (``target_type="restaurant"`` or the content
   equal to the doc's restaurant name, generic words like "restaurant"/"cafe"
   ignored) → 3 (restaurant) / 2 (its dishes); all query tokens inside the name
   → 2 / 1.
3. **Content match** (``target_type="dish"``/``"cuisine"`` replaces the text's
   content by the canonical target, which makes misspelled/compound training
   queries label correctly):
   dish docs — canonical dish == target, or all of ``D`` in the dish
   name/canonical tokens → 3 (2 if a requested cuisine mismatches); head token
   (last of ``D``) matches → 2; any other overlap, same cuisine, category or
   tag/review/description mention → 1; cuisine-only query and dish of that
   cuisine → 2 (1 when the query asks for "restaurants").
   restaurant docs — serves the dish (one of its dish docs covers ``D``), or
   ``D`` in its name/categories → 2; same cuisine as the dish or a mention → 1;
   cuisine-only query and restaurant of that cuisine/category → 2.
   No content at all (pure attribute query) → ``1 + min(attribute levels)``.
4. **Rating modulation** (±1 at most, never for navigational matches):
   a 3 requires ``rating >= 3.5`` (else 2); a category-level or attribute-only
   2 becomes 3 when ``rating >= 4.5`` with ``review_count >= 50``.
5. **Constraint caps** (applied last, hard): each attribute has a level 0–2
   (e.g. ``cheap``: tier 1 → 2, tier 2 → 1, tier ≥ 3 → 0) and caps the grade at
   ``1 + level``; violated diet → **0** (restaurants: 0 if none of its known
   dishes satisfies it, else capped at 1 when its menu is unknown); dish price
   above ``under $N`` → cap **1**.

Known blind spots (by design — they are what the hand labels measure):
synonyms/transliterations ("baingan" = eggplant), vague semantic queries
(only tag/review mentions → ≤ 1), and anything requiring world knowledge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from dishcovery.data.schema import HOURS_HEX_LEN, MEAL_BUCKETS, hours_bit, time_slot
from dishcovery.qu.text import normalize

WEAK_ANNOTATOR = "weak_v1"

# ---------------------------------------------------------------------------
# Geo / hours helpers (shared by querylog, hand_label)
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def point_in_polygon(lat: float, lon: float, polygon: Sequence[Sequence[float]]) -> bool:
    """Ray-casting test; ``polygon`` is ``[[lat, lon], ...]`` (not closed).

    Points exactly on an edge may fall either way (irrelevant at metre scale).
    """
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i][0], polygon[i][1]
        yj, xj = polygon[j][0], polygon[j][1]
        if (yi > lat) != (yj > lat):
            x_cross = xi + (lat - yi) * (xj - xi) / (yj - yi)
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


def can_deliver(doc: Mapping[str, Any], lat: float, lon: float) -> Optional[bool]:
    """Whether ``(lat, lon)`` is inside the doc's delivery zone; ``None`` if unknown."""
    poly = doc.get("delivery_polygon") or []
    if len(poly) >= 3:
        return point_in_polygon(lat, lon, poly)
    radius = float(doc.get("delivery_radius_km") or 0.0)
    if radius > 0 and "lat" in doc and "lon" in doc:
        return haversine_km(lat, lon, float(doc["lat"]), float(doc["lon"])) <= radius
    return None


def hours_known(hours: Optional[str]) -> bool:
    """Hours bitmap present and not all-zero (all-zero = missing data)."""
    return bool(hours) and len(hours) == HOURS_HEX_LEN and any(c != "0" for c in hours)


def is_open_at(doc: Mapping[str, Any], dow: int, hour: int, minute: int) -> Optional[bool]:
    """Open in the half-hour slot of (dow, hour, minute)? ``None`` if hours unknown."""
    hours = doc.get("hours")
    if not hours_known(hours):
        return None
    return hours_bit(hours, time_slot(dow, hour, minute))


def singular(tok: str) -> str:
    """Light English singularisation used for matching only ("tacos" -> "taco",
    "curries" -> "curry", "cookie(s)" -> "cooky" on both sides)."""
    n = len(tok)
    if n > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if n > 4 and tok.endswith("ie"):
        return tok[:-2] + "y"
    if n > 4 and tok.endswith(("sses", "shes", "ches", "xes", "zes")):
        return tok[:-2]
    if n > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def _sing_set(text: str) -> FrozenSet[str]:
    return frozenset(singular(t) for t in normalize(text).split())


# ---------------------------------------------------------------------------
# Query parsing
# ---------------------------------------------------------------------------

_STOP = frozenset("""
a an the and or for with of in on at to me my i im want wanna some any please near nearby
around close closest local now today tonight best good great nice order get find delivery
deliver delivered deliveries is are that this from by like style day
""".split())
_GENERIC = frozenset("""
food foods eats eat restaurant restaurants place places spot spots joint joints something
anything options option menu dish dishes meal meals stuff
""".split())
_RESTAURANT_INTENT = frozenset("restaurant restaurants place places spot spots joint joints".split())
_DIET_WORDS = {"vegan": "vegan", "vegetarian": "veg", "veg": "veg", "veggie": "veg",
               "halal": "halal", "jain": "jain", "glutenfree": "gluten_free"}
_CHEAP = frozenset("cheap inexpensive affordable budget".split())
_UPSCALE = frozenset("expensive upscale fancy luxury".split())
_MEAL_WORDS = {"breakfast": "breakfast", "brunch": "breakfast", "lunch": "lunch",
               "dinner": "dinner", "supper": "dinner"}
_PRICE_WORDS = frozenset("under below max upto within".split())
_TWO_WORD = {
    ("gluten", "free"): ("diet", "gluten_free"),
    ("late", "night"): ("attr", "late_night"),
    ("open", "late"): ("attr", "late_night"),
    ("open", "now"): ("attr", "open_now"),
    ("top", "rated"): ("attr", "top_rated"),
    ("highly", "rated"): ("attr", "top_rated"),
    ("best", "rated"): ("attr", "top_rated"),
    ("well", "rated"): ("attr", "top_rated"),
    ("fast", "delivery"): ("attr", "fast_delivery"),
    ("quick", "delivery"): ("attr", "fast_delivery"),
    ("fine", "dining"): ("attr", "upscale"),
    ("near", "me"): ("skip", ""),
}

# Fallback cuisine vocabulary (merged with the cuisines found in the corpus).
KNOWN_CUISINES = frozenset("""
indian thai chinese japanese korean mexican italian ethiopian mediterranean greek vietnamese
american french spanish lebanese turkish caribbean cajun southern pakistani nepalese persian
moroccan peruvian brazilian cuban filipino indonesian malaysian german irish british
middle_eastern tex_mex seafood desserts
""".split())

# Words stripped from restaurant / chain names before comparing them with a query.
_NAME_GENERIC = frozenset("restaurant restaurants cafe the and bar grill kitchen eatery llc inc co".split())
_CHAIN_SUFFIX = frozenset("""
pizza restaurant restaurants grill cafe coffee kitchen bar company co express mexican burgers
burger chicken sandwiches subs bakery donuts house the
""".split())


@dataclass
class ParsedQuery:
    normalized: str
    content: List[str]          # content tokens (normalised, not singularised), in order
    diets: Set[str]
    attributes: Set[str]        # cheap upscale top_rated open_now late_night fast_delivery
    meal: Optional[str]         # breakfast | lunch | dinner | late_night
    max_price: Optional[float]
    restaurant_intent: bool


def parse_query(text: str) -> ParsedQuery:
    """Split a query into content tokens and constraints (see module docstring)."""
    norm = normalize(text)
    toks = norm.split()
    content: List[str] = []
    diets: Set[str] = set()
    attrs: Set[str] = set()
    meal: Optional[str] = None
    max_price: Optional[float] = None
    rest_intent = False
    i = 0
    while i < len(toks):
        t = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        pair = (t, nxt) if nxt is not None else None
        if pair in _TWO_WORD:
            kind, val = _TWO_WORD[pair]
            if kind == "diet":
                diets.add(val)
            elif kind == "attr":
                attrs.add(val)
                if val == "late_night":
                    meal = "late_night"
            i += 2
            continue
        if t in ("less", "up") and nxt in ("than", "to") and i + 2 < len(toks) and toks[i + 2].isdigit():
            max_price = float(toks[i + 2])
            i += 3
            continue
        if t in _PRICE_WORDS and nxt is not None and nxt.isdigit():
            max_price = float(nxt)
            i += 2
            continue
        if t == "fast" and nxt == "food":
            content += ["fast", "food"]
            i += 2
            continue
        if t in _DIET_WORDS:
            diets.add(_DIET_WORDS[t])
        elif t in _CHEAP:
            attrs.add("cheap")
        elif t in _UPSCALE:
            attrs.add("upscale")
        elif t in _MEAL_WORDS:
            meal = _MEAL_WORDS[t]
        elif t == "open":
            attrs.add("open_now")
        elif t in ("fast", "quick", "quickly"):
            attrs.add("fast_delivery")
        elif t in _RESTAURANT_INTENT:
            rest_intent = True
        elif t in _STOP or t in _GENERIC:
            pass
        else:
            content.append(t)
        i += 1
    if "vegan" in diets:
        diets.discard("veg")  # vegan implies veg; keep the stricter label only
    return ParsedQuery(norm, content, diets, attrs, meal, max_price, rest_intent)


def _name_key(name: str, generic: FrozenSet[str] = _NAME_GENERIC) -> str:
    toks = normalize(name).split()
    kept = [t for t in toks if t not in generic]
    return "".join(kept or toks)


def _chain_aliases(chain_key: str, names: Iterable[str]) -> Set[str]:
    out = {chain_key.replace(" ", ""), "".join(t for t in chain_key.split() if t not in _STOP)}
    toks = chain_key.split()
    while len(toks) > 1 and toks[-1] in _CHAIN_SUFFIX:
        toks = toks[:-1]
    out.add("".join(toks))
    for n in names:
        nn = normalize(n)
        if nn:
            out.add(nn.replace(" ", ""))
    return {a for a in out if len(a) >= 4}


# ---------------------------------------------------------------------------
# Corpus index + per-doc features
# ---------------------------------------------------------------------------

@dataclass
class _DocFeat:
    is_dish: bool
    rid: str
    name_toks: FrozenSet[str]        # dish: name ∪ canonical; restaurant: its name
    rest_name_toks: FrozenSet[str]
    rest_key: str                    # restaurant name, generic words stripped, no spaces
    canonical: str
    cuisines: FrozenSet[str]
    cat_toks: FrozenSet[str]         # categories + cuisines
    mention_toks: FrozenSet[str]     # tags + description + reviews
    chain: str
    diets: FrozenSet[str]


class _CorpusIndex:
    def __init__(self) -> None:
        self.cuisine_phrases: Set[Tuple[str, ...]] = {tuple(normalize(c).split()) for c in KNOWN_CUISINES}
        self.chain_alias: Dict[str, str] = {}
        self.chain_cuisines: Dict[str, Set[str]] = {}
        self.canon_cuisines: Dict[str, Set[str]] = {}
        self.canon_by_token: Dict[str, Set[str]] = {}
        self.rest_dishes: Dict[str, Set[str]] = {}      # rid -> normalised dish names/canonicals
        self.rest_canon: Dict[str, Set[str]] = {}
        self.rest_diets: Dict[str, Set[str]] = {}

    @classmethod
    def build(cls, docs: Iterable[Mapping[str, Any]]) -> "_CorpusIndex":
        idx = cls()
        chain_names: Dict[str, Set[str]] = {}
        for d in docs:
            cuis = {normalize(c) for c in d.get("cuisines") or [] if c}
            for c in cuis:
                idx.cuisine_phrases.add(tuple(c.split()))
            chain = normalize(d.get("chain") or "")
            if chain:
                idx.chain_cuisines.setdefault(chain, set()).update(cuis)
                chain_names.setdefault(chain, set()).add(d.get("restaurant_name") or "")
            if d.get("type") == "dish":
                rid = d.get("restaurant_id") or ""
                canon = normalize(d.get("canonical_dish") or "")
                if canon:
                    idx.canon_cuisines.setdefault(canon, set()).update(cuis)
                    for t in canon.split():
                        idx.canon_by_token.setdefault(singular(t), set()).add(canon)
                    idx.rest_canon.setdefault(rid, set()).add(canon)
                dishes = idx.rest_dishes.setdefault(rid, set())
                dishes.add(normalize(d.get("name") or ""))
                if canon:
                    dishes.add(canon)
                diet = set(d.get("diet") or [])
                if "vegan" in diet:
                    diet.add("veg")
                idx.rest_diets.setdefault(rid, set()).update(diet)
        for chain, names in chain_names.items():
            for a in _chain_aliases(chain, names):
                idx.chain_alias.setdefault(a, chain)
        return idx


def _feat(doc: Mapping[str, Any]) -> _DocFeat:
    is_dish = doc.get("type") == "dish"
    rname = doc.get("restaurant_name") or (doc.get("name", "") if not is_dish else "")
    canon = normalize(doc.get("canonical_dish") or "")
    name_toks = _sing_set(doc.get("name") or "")
    if is_dish and canon:
        name_toks = name_toks | _sing_set(canon)
    cuis = frozenset(normalize(c) for c in doc.get("cuisines") or [] if c)
    cat_toks: Set[str] = set()
    for c in list(doc.get("categories") or []) + list(cuis):
        cat_toks |= _sing_set(c)
    mention = set()
    for t in doc.get("tags") or []:
        mention |= _sing_set(t)
    mention |= _sing_set(doc.get("description") or "")
    mention |= _sing_set((doc.get("reviews") or "")[:4000])
    diet = set(doc.get("diet") or [])
    if "vegan" in diet:
        diet.add("veg")
    return _DocFeat(
        is_dish=is_dish,
        rid=doc.get("restaurant_id") or doc.get("id", ""),
        name_toks=frozenset(name_toks),
        rest_name_toks=_sing_set(rname),
        rest_key=_name_key(rname),
        canonical=canon,
        cuisines=cuis,
        cat_toks=frozenset(cat_toks),
        mention_toks=frozenset(mention),
        chain=normalize(doc.get("chain") or ""),
        diets=frozenset(diet),
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def ctx_time(ctx: Any) -> Optional[Tuple[int, int, int]]:
    """(dow, hour, minute) from a ctx dict/object: ``time`` ("tue 13:00") or
    ``dow``/``hour``/``minute`` fields; ``None`` if absent."""
    t = _get(ctx, "time")
    if isinstance(t, str) and t.strip():
        from dishcovery.labels.queries import parse_time
        return parse_time(t)
    if _get(ctx, "dow") is not None and _get(ctx, "hour") is not None:
        return int(_get(ctx, "dow")), int(_get(ctx, "hour")), int(_get(ctx, "minute", 0) or 0)
    return None


def ctx_latlon(ctx: Any) -> Optional[Tuple[float, float]]:
    lat, lon = _get(ctx, "lat"), _get(ctx, "lon")
    if lat is None or lon is None:
        return None
    return float(lat), float(lon)


# ---------------------------------------------------------------------------
# The labeler
# ---------------------------------------------------------------------------

class WeakLabeler:
    """Heuristic 0–3 grader (see the module docstring for the exact definition).

    Args:
        docs: ``doc_id -> doc`` for the whole corpus (or the part being
            labelled). It is scanned once, lazily, to learn cuisine keys,
            chain aliases and each restaurant's menu.
        feature_cache_size: max number of per-doc feature entries kept.
    """

    def __init__(self, docs: Mapping[str, Mapping[str, Any]], feature_cache_size: int = 200_000):
        self.docs = docs
        self._index: Optional[_CorpusIndex] = None
        self._feats: Dict[str, _DocFeat] = {}
        self._cache_size = feature_cache_size

    # -- internals -----------------------------------------------------------
    @property
    def index(self) -> _CorpusIndex:
        if self._index is None:
            self._index = _CorpusIndex.build(self.docs.values())
        return self._index

    def _features(self, doc: Mapping[str, Any]) -> _DocFeat:
        key = doc.get("id")
        if key is not None and key in self._feats:
            return self._feats[key]
        f = _feat(doc)
        if key is not None:
            if len(self._feats) >= self._cache_size:
                self._feats.clear()
            self._feats[key] = f
        return f

    def _split_cuisines(self, content: List[str]) -> Tuple[Set[str], List[str]]:
        """Greedy longest-match of cuisine phrases; returns (cuisines, other tokens)."""
        phrases = self.index.cuisine_phrases
        maxlen = max((len(p) for p in phrases), default=1)
        found: Set[str] = set()
        rest: List[str] = []
        i = 0
        while i < len(content):
            for n in range(min(maxlen, len(content) - i), 0, -1):
                if tuple(content[i:i + n]) in phrases:
                    found.add(" ".join(content[i:i + n]))
                    i += n
                    break
            else:
                rest.append(content[i])
                i += 1
        # A single-token "cuisine" that ends the query and is also a dish word
        # ("margherita pizza", "pizza") is the dish head, not a cuisine.
        last = content[-1] if content else ""
        if last in found and singular(last) in self.index.canon_by_token:
            found.discard(last)
            rest.append(last)
        return found, rest

    # -- public --------------------------------------------------------------
    def grade(self, query_text: str, doc: Mapping[str, Any], ctx: Any = None,
              target_type: Optional[str] = None, target: Optional[str] = None) -> int:
        """Weak relevance grade 0..3 of ``doc`` for ``query_text``.

        Args:
            query_text: raw query.
            doc: full document (schema.REQUIRED_FIELDS).
            ctx: optional user context (dict or object, e.g. ``EvalQuery`` or a
                query-log record) with ``lat``/``lon`` and ``time``
                ("tue 13:00") or ``dow``/``hour``/``minute``.
            target_type: ``dish`` | ``cuisine`` | ``chain`` | ``restaurant`` |
                ``attribute`` — the query log's ground-truth intent, if known.
            target: canonical key of the target (canonical dish, cuisine key,
                chain key, normalised restaurant name).
        """
        return self.explain(query_text, doc, ctx, target_type, target)[0]

    def explain(self, query_text: str, doc: Mapping[str, Any], ctx: Any = None,
                target_type: Optional[str] = None, target: Optional[str] = None
                ) -> Tuple[int, List[str]]:
        """Like :meth:`grade` but also returns the reasons behind the grade."""
        why: List[str] = []
        f = self._features(doc)
        pq = parse_query(query_text)

        # 1. Serviceability --------------------------------------------------
        ll = ctx_latlon(ctx)
        if ll is not None and can_deliver(doc, ll[0], ll[1]) is False:
            return 0, ["outside delivery zone"]
        t = ctx_time(ctx)

        # 2-3. Content ----------------------------------------------------------
        g, kind = self._content(pq, f, target_type, target, why)
        levels = self._attribute_levels(pq, doc, f, t, ll)
        if kind == "attribute":  # no content: the attributes are the request
            g = 1 + min(levels.values()) if levels else 2
            why.append(f"attribute-only query, levels {levels}")

        # 4. Rating modulation ---------------------------------------------------
        rating = float(doc.get("rating") or 0.0)
        reviews = int(doc.get("review_count") or 0)
        if kind != "navigational":
            if g == 3 and reviews > 0 and rating < 3.5:
                g = 2
                why.append(f"rating {rating:.1f} < 3.5: 3 -> 2")
            elif g == 2 and kind in ("category", "attribute") and rating >= 4.5 and reviews >= 50:
                g = 3
                why.append(f"rating {rating:.1f} with {reviews} reviews: 2 -> 3")

        # 5. Constraint caps -----------------------------------------------------
        if levels and kind != "attribute":
            worst = min(levels.values())
            if 1 + worst < g:
                g = 1 + worst
                why.append(f"attribute cap {levels} -> {g}")
        if pq.diets:
            if not pq.diets <= f.diets:
                if f.is_dish:
                    g = 0
                    why.append(f"diet {sorted(pq.diets)} violated")
                else:
                    menu = self.index.rest_diets.get(f.rid)
                    if menu is not None or f.rid in self.index.rest_dishes:
                        if not pq.diets <= (menu or set()) | f.diets:
                            g = 0
                            why.append(f"no dish satisfies {sorted(pq.diets)}")
                    elif g > 1:
                        g = 1
                        why.append(f"diet {sorted(pq.diets)} unknown for restaurant: cap 1")
        if pq.max_price is not None:
            price = float(doc.get("price") or 0.0)
            if price > pq.max_price and g > 1:
                g = 1
                why.append(f"price {price:.2f} > {pq.max_price:.0f}: cap 1")
        if t is not None and g > 1 and is_open_at(doc, *t) is False:
            g = 1
            why.append("closed at query time: cap 1")
        return max(0, min(3, g)), why

    # -- content grading -----------------------------------------------------
    def _content(self, pq: ParsedQuery, f: _DocFeat, target_type: Optional[str],
                 target: Optional[str], why: List[str]) -> Tuple[int, str]:
        idx = self.index
        tgt = normalize(target) if target else ""
        joined = "".join(pq.content)

        # Chain (navigational).
        chain = ""
        if target_type == "chain" and tgt:
            chain = tgt
        elif joined:
            chain = idx.chain_alias.get(joined, "") or idx.chain_alias.get(pq.normalized.replace(" ", ""), "")
        if chain:
            if f.chain == chain:
                why.append(f"chain {chain!r}")
                return (2 if f.is_dish else 3), "navigational"
            if f.cuisines & idx.chain_cuisines.get(chain, set()):
                why.append(f"same cuisine as chain {chain!r}")
                return 1, "navigational"
            return 0, "navigational"

        # Restaurant name.
        if target_type == "restaurant" and tgt:
            if f.rest_key == _name_key(tgt):
                why.append("restaurant name == target")
                return (2 if f.is_dish else 3), "navigational"
            return 0, "navigational"
        name_grade = 0
        if pq.content:
            qkey = "".join(t for t in pq.content if t not in _NAME_GENERIC)
            qset = {singular(t) for t in pq.content}
            if qkey and qkey == f.rest_key:
                name_grade = 2 if f.is_dish else 3
                why.append("query == restaurant name")
            elif len(qset) >= 2 and qset <= f.rest_name_toks:
                name_grade = 1 if f.is_dish else 2
                why.append("query tokens inside restaurant name")

        # Dish / cuisine content.
        canon_target = ""
        if target_type == "dish" and tgt:
            cuisines: Set[str] = set()
            dish_toks = tgt.split()
            canon_target = tgt
        elif target_type == "cuisine" and tgt:
            cuisines, dish_toks = {tgt}, []
        else:
            cuisines, dish_toks = self._split_cuisines(pq.content)
        D = [singular(t) for t in dish_toks]
        if not D and not cuisines:
            if name_grade:
                return name_grade, "navigational"
            return self._attribute_only(pq, why)
        g, kind = self._dish_cuisine(pq, f, D, cuisines, canon_target, why)
        if name_grade > g:
            return name_grade, "navigational"
        return g, kind

    def _dish_cuisine(self, pq: ParsedQuery, f: _DocFeat, D: List[str], C: Set[str],
                      canon_target: str, why: List[str]) -> Tuple[int, str]:
        idx = self.index
        Dset = set(D)
        c_toks = {singular(t) for c in C for t in c.split()}
        cuisine_ok = (not C) or bool(f.cuisines & C) or c_toks <= f.name_toks or c_toks <= f.cat_toks
        if not D:  # cuisine-only query
            if f.cuisines & C or (c_toks and c_toks <= f.cat_toks):
                g = (1 if pq.restaurant_intent else 2) if f.is_dish else 2
                why.append(f"cuisine {sorted(C)}")
                return g, "category"
            if c_toks & f.mention_toks:
                why.append("cuisine mentioned")
                return 1, "mention"
            return 0, "none"

        head = D[-1]
        if f.is_dish:
            if canon_target and f.canonical == canon_target:
                why.append("canonical dish == target")
                return 3, "exact"
            if Dset <= f.name_toks:
                why.append("all dish tokens in dish name")
                return (3 if cuisine_ok else 2), "exact"
            if head in f.name_toks:
                why.append(f"head token {head!r} in dish name")
                return (2 if cuisine_ok else 1), "partial"
            if Dset & f.name_toks:
                why.append("partial dish-name overlap")
                return 1, "partial"
            if C and f.cuisines & C:
                why.append("dish of requested cuisine")
                return 1, "category"
            if Dset <= f.cat_toks:
                why.append("dish tokens in categories")
                return 1, "category"
            if Dset & f.mention_toks:
                why.append("mentioned in tags/description/reviews")
                return 1, "mention"
            return 0, "none"

        # Restaurant doc.
        serves = False
        if canon_target and canon_target in idx.rest_canon.get(f.rid, ()):
            serves = True
        else:
            for dish in idx.rest_dishes.get(f.rid, ()):
                if Dset <= {singular(t) for t in dish.split()}:
                    serves = True
                    break
        if serves or Dset <= f.name_toks or Dset <= f.cat_toks:
            why.append("serves the dish" if serves else "dish in restaurant name/categories")
            return (2 if cuisine_ok else 1), "serves"
        if C and f.cuisines & C:
            why.append("restaurant of requested cuisine")
            return 1, "category"
        dish_cuisines = self._cuisines_of_dish(D, canon_target)
        if dish_cuisines & f.cuisines:
            why.append(f"restaurant cuisine matches the dish's {sorted(dish_cuisines & f.cuisines)}")
            return 1, "category"
        if Dset & (f.mention_toks | f.name_toks):
            why.append("mentioned in tags/description/reviews")
            return 1, "mention"
        return 0, "none"

    def _cuisines_of_dish(self, D: List[str], canon_target: str) -> Set[str]:
        idx = self.index
        if canon_target:
            return set(idx.canon_cuisines.get(canon_target, ()))
        cands: Optional[Set[str]] = None
        for t in D:
            s = idx.canon_by_token.get(t, set())
            cands = set(s) if cands is None else cands & s
            if not cands:
                return set()
        out: Set[str] = set()
        for c in cands or ():
            out |= idx.canon_cuisines.get(c, set())
        return out

    @staticmethod
    def _attribute_only(pq: ParsedQuery, why: List[str]) -> Tuple[int, str]:
        """Kind marker for content-free queries; explain() turns the attribute
        levels into the grade. With no constraints at all, everything is marginal."""
        if not pq.attributes and not pq.meal and not pq.diets and pq.max_price is None:
            why.append("no content and no constraints")
            return 1, "none"
        return 0, "attribute"

    def _attribute_levels(self, pq: ParsedQuery, doc: Mapping[str, Any], f: _DocFeat,
                          t: Optional[Tuple[int, int, int]],
                          ll: Optional[Tuple[float, float]]) -> Dict[str, int]:
        """Satisfaction level 0..2 of each determinable attribute constraint."""
        lv: Dict[str, int] = {}
        tier = int(doc.get("price_tier") or 0)
        rating = float(doc.get("rating") or 0.0)
        reviews = int(doc.get("review_count") or 0)
        aff = list(doc.get("meal_affinity") or [])
        if "cheap" in pq.attributes and tier:
            lv["cheap"] = 2 if tier <= 1 else (1 if tier == 2 else 0)
        if "upscale" in pq.attributes and tier:
            lv["upscale"] = 2 if tier >= 3 else (1 if tier == 2 else 0)
        if "top_rated" in pq.attributes:
            lv["top_rated"] = 2 if (rating >= 4.5 and reviews >= 20) else (1 if rating >= 4.0 else 0)
        if "open_now" in pq.attributes and t is not None:
            o = is_open_at(doc, *t)
            if o is not None:
                lv["open_now"] = 2 if o else 0
        if "late_night" in pq.attributes:
            if t is not None and (t[1] >= 21 or t[1] < 5):
                o = is_open_at(doc, *t)
            else:  # query issued earlier in the day: check tonight at 23:30
                o = is_open_at(doc, t[0] if t else 4, 23, 30)
            if o is not None:
                lv["late_night"] = 2 if o else 0
            elif len(aff) == len(MEAL_BUCKETS):
                a = aff[MEAL_BUCKETS.index("late_night")]
                lv["late_night"] = 2 if a >= 0.15 else (1 if a >= 0.05 else 0)
        if "fast_delivery" in pq.attributes:
            prep = float(doc.get("prep_time_min") or 0.0)
            if prep > 0:
                level = 2 if prep <= 15 else (1 if prep <= 25 else 0)
                if ll is not None and "lat" in doc and "lon" in doc:
                    if haversine_km(ll[0], ll[1], float(doc["lat"]), float(doc["lon"])) > 5.0:
                        level = max(0, level - 1)
                lv["fast_delivery"] = level
        if pq.meal in ("breakfast", "lunch", "dinner") and len(aff) == len(MEAL_BUCKETS):
            a = aff[MEAL_BUCKETS.index(pq.meal)]
            if pq.meal == "breakfast":
                cat_hit = bool({"breakfast", "brunch"} & f.cat_toks)
                lv["breakfast"] = 2 if (a >= 0.3 or cat_hit) else (1 if a >= 0.15 else 0)
            else:
                lv[pq.meal] = 2 if a >= 0.2 else (1 if a >= 0.1 else 0)
        return lv


def label_candidates(queries: Iterable[Any], candidates_by_qid: Mapping[str, Sequence[str]],
                     docs: Mapping[str, Mapping[str, Any]], labeler: Optional[WeakLabeler] = None,
                     annotator: str = WEAK_ANNOTATOR) -> List[dict]:
    """Weakly label every (query, candidate) pair.

    Args:
        queries: ``EvalQuery`` objects or query-log records (need ``qid``,
            ``text``; optionally ``lat``/``lon``/``time``/``target_type``/``target``).
        candidates_by_qid: ``qid -> doc ids`` to label (e.g. top-k of a
            first-stage retriever).
        docs: ``doc_id -> doc``.
        labeler: reuse an existing labeler (its corpus index) if given.

    Returns:
        qrel records ``{"qid","doc_id","grade","source":"weak","annotator"}``
        (load into ``qid -> {doc: grade}`` with ``eval.io.qrels_from_records``).
    """
    lab = labeler or WeakLabeler(docs)
    out: List[dict] = []
    for q in queries:
        qid = str(_get(q, "qid"))
        cands = candidates_by_qid.get(qid)
        if not cands:
            continue
        text = _get(q, "text", "")
        tt, tg = _get(q, "target_type"), _get(q, "target")
        for doc_id in cands:
            doc = docs.get(doc_id)
            if doc is None:
                continue
            g = lab.grade(text, doc, ctx=q, target_type=tt, target=tg)
            out.append({"qid": qid, "doc_id": str(doc_id), "grade": int(g), "source": "weak",
                        "annotator": annotator})
    return out
