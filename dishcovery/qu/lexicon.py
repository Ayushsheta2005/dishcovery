"""Query-understanding lexicon: term frequencies + a typed gazetteer.

Built by streaming ``docs.jsonl`` (see :mod:`dishcovery.data.schema`) plus the
dish taxonomy (``dishcovery/data/taxonomy.yaml``) and the curated synonym
table. Persisted as plain TSV/JSON so loading needs no pickle and no sklearn.

Gazetteer entry types (in linker priority order): chain, dish, cuisine,
restaurant, ingredient.
"""
from __future__ import annotations

import gzip
import heapq
import json
import os
import re
from collections import Counter, defaultdict
from typing import (Any, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Set,
                    Tuple, Union)

from dishcovery.qu.synonyms import SynonymTable
from dishcovery.qu.text import _APOS_RE, _NON_ALNUM_RE, normalize, tokenize

GAZ_TYPES = ("chain", "dish", "cuisine", "restaurant", "ingredient")
TYPE_PRIORITY: Dict[str, int] = {t: i for i, t in enumerate(GAZ_TYPES)}

DEFAULT_TAXONOMY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "taxonomy.yaml")

LEXICON_FORMAT = 1


class GazEntry(NamedTuple):
    """One gazetteer surface form (normalised) and what it refers to."""
    surface: str
    type: str
    canonical: str
    weight: float = 1.0


# Words that may be stripped from the end of a brand / restaurant name to derive
# an alias ("dominos pizza" -> "dominos", "chipotle mexican grill" -> "chipotle").
GENERIC_NAME_SUFFIXES = frozenset("""
pizza pizzas pizzeria restaurant restaurants grill grille kitchen kitchens cafe bar express house co
company inc llc ltd corp eatery bistro diner cantina taqueria bakery coffee burger burgers taco tacos
donut donuts doughnut doughnuts sandwich sandwiches subs sub wings bbq barbecue shop place joint spot
truck food foods catering steakhouse chicken bagel bagels deli market cuisine eats pub tavern lounge
brewery brewing bread noodles noodle sushi and n of the at on
mexican italian chinese thai indian japanese american asian mediterranean greek korean vietnamese french
""".split())
GENERIC_NAME_PREFIXES = frozenset({"the"})
_CONNECTORS = frozenset({"and", "n", "of", "the", "at", "on", "a", "in"})

# Never registered as a chain alias or as a whole restaurant name.
GENERIC_FOOD_WORDS = frozenset("""
pizza pizzas burger burgers taco tacos burrito burritos sushi ramen pho chicken wings pasta noodles
noodle coffee tea bbq barbecue sandwich sandwiches subs sub donut donuts doughnut doughnuts bagel bagels
salad salads soup soups steak seafood fish curry rice bread bakery cafe bar grill kitchen restaurant
restaurants diner deli food foods eats express house pub tavern market shop truck dessert desserts
juice smoothie smoothies waffle waffles pancake pancakes breakfast brunch lunch dinner snack snacks
kebab gyro gyros falafel shawarma dumplings dumpling boba cream icecream
mexican italian chinese thai indian japanese american asian mediterranean greek korean vietnamese french
""".split())
_VENUE_WORDS = frozenset("""
restaurant restaurants grill grille kitchen cafe bar house co company eatery bistro diner cantina pub
tavern lounge express place joint spot shop truck deli market inc llc the and n of at on a in
""".split())

# Taxonomy dish tags that describe a dish rather than name an ingredient.
DESCRIPTOR_TAGS = frozenset("""
spicy sweet savory savoury sour tangy mild hot cold warm crispy crunchy creamy rich light heavy fried
grilled baked roasted steamed smoky smoked healthy vegan veg vegetarian comfort street classic
traditional popular signature kids sharing fresh quick side sides appetizer appetizers starter
starters main mains entree entrees dessert desserts drink drinks beverage beverages breakfast brunch
lunch dinner snack snacks bowl soup salad gluten free non spiced tangy filling handheld finger
""".split())
_MEAL_OR_VENUE = frozenset("""
breakfast brunch lunch dinner food foods bars bar restaurants restaurant shops shop trucks truck stands
stand delis deli cafes cafe fast new
""".split())
# Last words that are not dish heads ("lunch special", "family combo").
_HEAD_STOP = frozenset("""
special specials combo combos platter platters plate plates meal meals box set deluxe style regular
large small medium order side sides piece pieces pc pcs dozen half full family kids lunch dinner
breakfast appetizer starter entree plain original classic and with of in
""".split())


def _fast_tokens(s: str) -> List[str]:
    """Same output as :func:`text.tokenize`, with an ASCII fast path for bulk builds."""
    if s.isascii():
        return _NON_ALNUM_RE.sub(" ", _APOS_RE.sub("", s.lower())).split()
    return tokenize(s)


def iter_docs(path: str, max_docs: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Stream JSON objects from a (optionally gzipped) JSONL file."""
    try:
        import orjson
        loads = orjson.loads
    except ImportError:  # pragma: no cover - orjson is a declared dependency
        loads = json.loads
    opener = gzip.open if path.endswith(".gz") else open
    n = 0
    with opener(path, "rb") as f:
        for line in f:
            if max_docs is not None and n >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            n += 1
            yield loads(line)


def load_taxonomy(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load the taxonomy YAML; returns None when ``path`` is empty or missing."""
    if not path or not os.path.exists(path):
        return None
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    return data if isinstance(data, dict) else None


def derive_aliases(key: str, blocked: Iterable[str] = (), min_tokens: int = 1) -> List[str]:
    """Aliases of a brand / restaurant name made by stripping generic words.

    ``"dominos pizza" -> ["dominos"]``; ``"pizza hut" -> []``. An alias that is
    itself a generic dish / cuisine / venue word (or in ``blocked``) is never
    produced, nor one shorter than ``min_tokens`` tokens.
    """
    blocked = blocked if isinstance(blocked, (set, frozenset)) else set(blocked)
    out: List[str] = []
    cur = key.split()
    while len(cur) > 1:
        if cur[0] in GENERIC_NAME_PREFIXES:
            cur = cur[1:]
        elif cur[-1] in GENERIC_NAME_SUFFIXES:
            cur = cur[:-1]
        else:
            break
        if len(cur) < min_tokens:
            break
        if cur[0] in _CONNECTORS or cur[-1] in _CONNECTORS:
            continue
        alias = " ".join(cur)
        if alias == key or alias in out or alias in blocked or alias in GENERIC_FOOD_WORDS:
            continue
        if len(cur) == 1 and (len(alias) < 3 or alias.isdigit()):
            continue
        if all(t in GENERIC_FOOD_WORDS or t in _VENUE_WORDS for t in cur):
            continue
        out.append(alias)
    return out


def _valid_restaurant_name(name: str, blocked: Set[str]) -> bool:
    if len(name) < 3 or not any(ch.isalpha() for ch in name):
        return False
    if name in blocked or name in GENERIC_FOOD_WORDS:
        return False
    toks = name.split()
    if all(t in _VENUE_WORDS for t in toks):
        return False
    return True


EntryLike = Union[GazEntry, Sequence[Any], Mapping[str, Any]]


class Lexicon:
    """Term frequencies plus a typed gazetteer of normalised surface forms.

    Attributes:
        freqs: token -> corpus frequency.
        gazetteer: surface -> entries sorted by (type priority, -weight).
        protected: tokens never spell-corrected (entity names, ingredients).
        meta: build statistics/parameters.
    """

    def __init__(
        self,
        freqs: Optional[Mapping[str, int]] = None,
        entries: Iterable[EntryLike] = (),
        protected: Iterable[str] = (),
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.freqs: Dict[str, int] = dict(freqs or {})
        self.gazetteer: Dict[str, List[GazEntry]] = {}
        self.protected: Set[str] = set(protected)
        self.meta: Dict[str, Any] = dict(meta or {})
        self.total = 0
        self.max_ngram = 1
        self._by_type: Optional[Dict[str, List[GazEntry]]] = None
        self.nospace: Dict[str, GazEntry] = {}
        for e in entries:
            self._add_entry_like(e)
        self.finalize()

    # ------------------------------------------------------------ entries
    def add_entry(self, surface: str, type: str, canonical: Optional[str] = None,
                  weight: float = 1.0) -> bool:
        """Register ``surface`` as an entity; keeps the max weight per (surface, type)."""
        if type not in TYPE_PRIORITY:
            raise ValueError(f"unknown gazetteer type {type!r}")
        s = normalize(surface)
        if not s:
            return False
        c = normalize(canonical) if canonical else s
        c = c or s
        lst = self.gazetteer.setdefault(s, [])
        for k, e in enumerate(lst):
            if e.type == type:
                if weight > e.weight:
                    lst[k] = GazEntry(s, type, c, float(weight))
                    lst.sort(key=lambda x: (TYPE_PRIORITY[x.type], -x.weight))
                return False
        lst.append(GazEntry(s, type, c, float(weight)))
        lst.sort(key=lambda x: (TYPE_PRIORITY[x.type], -x.weight))
        self._by_type = None
        return True

    def _add_entry_like(self, e: EntryLike) -> None:
        if isinstance(e, GazEntry):
            self.add_entry(e.surface, e.type, e.canonical, e.weight)
        elif isinstance(e, Mapping):
            self.add_entry(e["surface"], e["type"], e.get("canonical"), float(e.get("weight", 1.0)))
        else:
            t = list(e)
            self.add_entry(t[0], t[1], t[2] if len(t) > 2 else None, float(t[3]) if len(t) > 3 else 1.0)

    def finalize(self) -> None:
        """Recompute derived indexes; call after bulk mutation."""
        self.total = sum(self.freqs.values())
        self.max_ngram = max((len(s.split()) for s in self.gazetteer), default=1)
        self._by_type = None
        # "mc donalds" / "pizzahut": chain names typed with odd spacing.
        self.nospace = {}
        for s, es in self.gazetteer.items():
            if " " not in s:
                continue
            for e in es:
                if e.type == "chain":
                    key = s.replace(" ", "")
                    if key not in self.gazetteer:
                        self.nospace.setdefault(key, e)
                    break

    def entries(self, surface: str) -> List[GazEntry]:
        return self.gazetteer.get(surface, [])

    def by_type(self, type: str) -> List[GazEntry]:
        if self._by_type is None:
            idx: Dict[str, List[GazEntry]] = {t: [] for t in GAZ_TYPES}
            for es in self.gazetteer.values():
                for e in es:
                    idx[e.type].append(e)
            for lst in idx.values():
                lst.sort(key=lambda e: (-e.weight, e.surface))
            self._by_type = idx
        return self._by_type.get(type, [])

    def freq(self, term: str) -> int:
        return self.freqs.get(term, 0)

    def __contains__(self, term: object) -> bool:
        return term in self.freqs

    def __repr__(self) -> str:
        counts = {t: len(self.by_type(t)) for t in GAZ_TYPES}
        return f"Lexicon(terms={len(self.freqs)}, total={self.total}, gazetteer={counts})"

    # --------------------------------------------------------- constructors
    @classmethod
    def from_entries(
        cls,
        freqs: Optional[Mapping[str, int]] = None,
        entries: Iterable[EntryLike] = (),
        protected: Optional[Iterable[str]] = None,
        derive_chain_aliases: bool = True,
        default_freq: int = 1,
    ) -> "Lexicon":
        """Small hand-built lexicon (tests, demos).

        ``entries`` are GazEntry / ``(surface, type[, canonical[, weight]])`` /
        dicts. Every surface token is guaranteed a frequency >= ``default_freq``.
        Chain aliases are derived as in :meth:`build` unless disabled.
        """
        lex = cls(freqs=freqs, entries=entries)
        lex._add_dish_heads()  # "pizza" from "margherita pizza" + "pepperoni pizza", as in from_docs
        if derive_chain_aliases:
            blocked = lex._blocked_surfaces()
            for e in list(lex.by_type("chain")):
                if e.surface == e.canonical:
                    for a in derive_aliases(e.surface, blocked):
                        lex.add_entry(a, "chain", e.canonical, min(e.weight, 0.95))
        for s in lex.gazetteer:
            for t in s.split():
                if lex.freqs.get(t, 0) < default_freq:
                    lex.freqs[t] = default_freq
        if protected is None:
            protected = lex._default_protected()
        lex.protected = set(protected)
        lex.finalize()
        return lex

    @classmethod
    def build(
        cls,
        docs_path: str,
        taxonomy_path: Optional[str] = None,
        max_docs: Optional[int] = None,
        min_freq: int = 2,
        synonyms: Optional[SynonymTable] = None,
        **kwargs: Any,
    ) -> "Lexicon":
        """Stream ``docs_path`` (JSONL, optionally .gz) and build the lexicon.

        ``taxonomy_path=None`` uses ``dishcovery/data/taxonomy.yaml`` when it
        exists; pass ``""`` to build without a taxonomy. Extra keyword
        arguments go to :meth:`from_docs`.
        """
        if taxonomy_path is None:
            taxonomy_path = DEFAULT_TAXONOMY_PATH
        taxonomy = load_taxonomy(taxonomy_path)
        lex = cls.from_docs(iter_docs(docs_path, max_docs), taxonomy=taxonomy, synonyms=synonyms,
                            min_freq=min_freq, **kwargs)
        lex.meta.update({"docs_path": os.path.abspath(docs_path),
                         "taxonomy_path": os.path.abspath(taxonomy_path) if taxonomy else None,
                         "max_docs": max_docs})
        return lex

    @classmethod
    def from_docs(
        cls,
        docs: Iterable[Mapping[str, Any]],
        taxonomy: Optional[Mapping[str, Any]] = None,
        synonyms: Optional[SynonymTable] = None,
        max_docs: Optional[int] = None,
        min_freq: int = 2,
        max_restaurants: int = 20000,
        min_dish_count: int = 3,
        taxonomy_boost: int = 20,
        review_weight: float = 0.2,
    ) -> "Lexicon":
        """Build from an iterable of document dicts (fields per data/schema.py)."""
        if synonyms is None:
            synonyms = SynonymTable.load()
        field_counts: Counter = Counter()
        review_counts: Counter = Counter()
        dish_names: Counter = Counter()
        dish_name_canon: Dict[str, Counter] = defaultdict(Counter)
        canon_counts: Counter = Counter()
        doc_cuisines: Counter = Counter()
        chain_counts: Counter = Counter()
        restaurants: Dict[str, float] = {}
        n_docs = 0
        for doc in docs:
            if max_docs is not None and n_docs >= max_docs:
                break
            n_docs += 1
            name_toks = _fast_tokens(doc.get("name") or "")
            field_counts.update(name_toks)
            field_counts.update(_fast_tokens(doc.get("restaurant_name") or ""))
            field_counts.update(_fast_tokens(doc.get("description") or ""))
            for c in doc.get("cuisines") or ():
                ct = _fast_tokens(c)
                field_counts.update(ct)
                if ct:
                    doc_cuisines[" ".join(ct)] += 1
            for t in doc.get("tags") or ():
                field_counts.update(_fast_tokens(t))
            rv = doc.get("reviews")
            if rv:
                review_counts.update(_fast_tokens(rv))
            chain = " ".join(_fast_tokens(doc.get("chain") or ""))
            if chain:
                chain_counts[chain] += 1
            dtype = doc.get("type")
            nn = " ".join(name_toks)
            if dtype == "dish":
                if nn:
                    dish_names[nn] += 1
                cd = " ".join(_fast_tokens(doc.get("canonical_dish") or ""))
                if cd:
                    canon_counts[cd] += 1
                    if nn:
                        dish_name_canon[nn][cd] += 1
            elif dtype == "restaurant" and not chain and nn:
                rc = float(doc.get("review_count") or 0)
                score = float(doc.get("popularity") or 0.0) * rc + 1e-6 * rc
                if score > restaurants.get(nn, -1.0):
                    restaurants[nn] = score

        lex = cls()
        boost: Counter = Counter()
        if taxonomy:
            lex._add_taxonomy(taxonomy, boost, taxonomy_boost)
        for nn, cnt in dish_names.items():
            if cnt >= min_dish_count:
                canon = dish_name_canon[nn].most_common(1)[0][0] if nn in dish_name_canon else nn
                lex.add_entry(nn, "dish", canon, 0.7)
        for cd, cnt in canon_counts.items():
            if cnt >= min_dish_count:
                lex.add_entry(cd, "dish", cd, 0.85)
        for c in doc_cuisines:
            lex.add_entry(c, "cuisine", c, 0.85)
        for term, canon, w in synonyms.terms_of_type("ingredient"):
            lex.add_entry(term, "ingredient", canon, round(0.6 * w, 4))
            for t in term.split():
                boost[t] += 3
        lex._add_dish_heads()

        blocked = lex._blocked_surfaces() | set(synonyms.all_terms())
        # An alias made by stripping words ("golden" from "golden kitchen") is only
        # safe if no *other* known name starts with it ("golden dragon", ...).
        prefix_names: Dict[str, Set[str]] = defaultdict(set)
        for nm in list(chain_counts) + list(restaurants):
            toks = nm.split()
            for L in range(1, len(toks)):
                prefix_names[" ".join(toks[:L])].add(nm)
        for chain in chain_counts:
            lex.add_entry(chain, "chain", chain, 1.0)
            for alias in derive_aliases(chain, blocked, min_tokens=1):
                if prefix_names.get(alias, set()) - {chain}:
                    continue  # ambiguous: other restaurants share this prefix
                lex.add_entry(alias, "chain", chain, 0.95)
        n_rest = 0
        for name, _score in heapq.nlargest(max_restaurants * 2, restaurants.items(), key=lambda kv: kv[1]):
            if n_rest >= max_restaurants:
                break
            if not _valid_restaurant_name(name, blocked):
                continue
            n_rest += 1
            lex.add_entry(name, "restaurant", name, 0.8)
            for alias in derive_aliases(name, blocked, min_tokens=2):
                lex.add_entry(alias, "restaurant", name, 0.6)

        freqs: Counter = Counter(field_counts)
        for t, c in review_counts.items():
            v = int(round(c * review_weight))
            if v:
                freqs[t] += v
        freqs.update(boost)
        vocab = {t: c for t, c in freqs.items() if c >= min_freq and len(t) <= 30}
        for s in lex.gazetteer:
            for t in s.split():
                if t not in vocab:
                    vocab[t] = max(freqs.get(t, 0), 1)
        lex.freqs = vocab
        lex.protected = lex._default_protected()
        lex.meta.update({
            "format": LEXICON_FORMAT,
            "n_docs": n_docs,
            "min_freq": min_freq,
            "review_weight": review_weight,
            "taxonomy_boost": taxonomy_boost,
            "has_taxonomy": bool(taxonomy),
            "n_chains": len(chain_counts),
            "n_restaurants": n_rest,
        })
        lex.finalize()
        return lex

    def _add_taxonomy(self, tax: Mapping[str, Any], boost: Counter, tb: int) -> None:
        cuisines = tax.get("cuisines") or {}
        if isinstance(cuisines, list):  # tolerate [{key: ..., ...}] form
            cuisines = {str(c.get("key") or c.get("name")): c for c in cuisines if isinstance(c, dict)}
        for ckey, cdef in cuisines.items():
            cdef = cdef or {}
            ck = normalize(str(ckey))
            if not ck:
                continue
            self.add_entry(ck, "cuisine", ck, 1.0)
            boost.update({t: tb for t in ck.split()})
            for a in cdef.get("aliases") or ():
                if self.add_entry(str(a), "cuisine", ck, 0.95) or True:
                    boost.update({t: max(1, tb // 2) for t in tokenize(str(a))})
            for yc in cdef.get("yelp_categories") or ():
                for part in re.split(r"[/&,]", str(yc)):
                    p = normalize(part)
                    if (p and p not in GENERIC_FOOD_WORDS
                            and not any(t in _MEAL_OR_VENUE for t in p.split())):
                        self.add_entry(p, "cuisine", ck, 0.6)
            for d in cdef.get("dishes") or ():
                if isinstance(d, str):
                    d = {"name": d}
                name = normalize(str(d.get("name") or ""))
                if not name:
                    continue
                self.add_entry(name, "dish", name, 1.0)
                boost.update({t: tb for t in name.split()})
                for v in d.get("variants") or ():
                    self.add_entry(str(v), "dish", name, 0.9)
                    boost.update({t: max(1, tb // 4) for t in tokenize(str(v))})
                for tag in d.get("tags") or ():
                    nt = normalize(str(tag))
                    if nt and nt not in DESCRIPTOR_TAGS and not all(t in DESCRIPTOR_TAGS for t in nt.split()):
                        self.add_entry(nt, "ingredient", nt, 0.4)
        for ing in tax.get("ingredients") or ():
            if isinstance(ing, str):
                self.add_entry(ing, "ingredient", ing, 0.6)
            elif isinstance(ing, dict) and ing.get("name"):
                self.add_entry(str(ing["name"]), "ingredient", str(ing["name"]), 0.6)
                for a in ing.get("aliases") or ():
                    self.add_entry(str(a), "ingredient", str(ing["name"]), 0.55)

    def _add_dish_heads(self, min_names: int = 2) -> None:
        """Register the last word shared by >= ``min_names`` dish names ("pizza", "biryani")."""
        canon = {e.canonical for e in self.by_type("dish") if e.weight >= 0.85}
        heads: Counter = Counter()
        for c in canon:
            toks = c.split()
            if len(toks) >= 2:
                heads[toks[-1]] += 1
        for h, n in heads.items():
            if n < min_names or len(h) < 3 or h in _HEAD_STOP or h.isdigit():
                continue
            if any(e.type in ("chain", "dish", "cuisine") for e in self.gazetteer.get(h, ())):
                continue
            self.add_entry(h, "dish", h, 0.5)

    def _blocked_surfaces(self) -> Set[str]:
        return {s for s, es in self.gazetteer.items()
                if any(e.type in ("dish", "cuisine", "ingredient") for e in es)}

    def _default_protected(self) -> Set[str]:
        out: Set[str] = set()
        for s, es in self.gazetteer.items():
            for e in es:
                if e.type in ("chain", "cuisine", "ingredient") or (e.type == "dish" and e.weight >= 1.0):
                    out.update(s.split())
                    break
        return out

    # ---------------------------------------------------------------- I/O
    def save(self, out_dir: str) -> None:
        """Write vocab.tsv, gazetteer.tsv, protected.txt and meta.json to ``out_dir``."""
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "vocab.tsv"), "w", encoding="utf-8") as f:
            for t, c in sorted(self.freqs.items(), key=lambda kv: (-kv[1], kv[0])):
                f.write(f"{t}\t{c}\n")
        with open(os.path.join(out_dir, "gazetteer.tsv"), "w", encoding="utf-8") as f:
            for s in sorted(self.gazetteer):
                for e in self.gazetteer[s]:
                    f.write(f"{e.surface}\t{e.type}\t{e.canonical}\t{e.weight:.4f}\n")
        with open(os.path.join(out_dir, "protected.txt"), "w", encoding="utf-8") as f:
            for t in sorted(self.protected):
                f.write(t + "\n")
        meta = dict(self.meta)
        meta.update({"format": LEXICON_FORMAT, "n_terms": len(self.freqs), "total": self.total,
                     "gazetteer": {t: len(self.by_type(t)) for t in GAZ_TYPES}})
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, in_dir: str) -> "Lexicon":
        lex = cls()
        freqs: Dict[str, int] = {}
        with open(os.path.join(in_dir, "vocab.tsv"), "r", encoding="utf-8") as f:
            for line in f.read().split("\n"):
                if line:
                    t, _, c = line.partition("\t")
                    freqs[t] = int(c)
        lex.freqs = freqs
        gaz: Dict[str, List[GazEntry]] = {}
        with open(os.path.join(in_dir, "gazetteer.tsv"), "r", encoding="utf-8") as f:
            for line in f.read().split("\n"):
                if line:
                    s, typ, canon, w = line.split("\t")
                    gaz.setdefault(s, []).append(GazEntry(s, typ, canon, float(w)))
        for lst in gaz.values():
            lst.sort(key=lambda x: (TYPE_PRIORITY[x.type], -x.weight))
        lex.gazetteer = gaz
        ppath = os.path.join(in_dir, "protected.txt")
        if os.path.exists(ppath):
            with open(ppath, "r", encoding="utf-8") as f:
                lex.protected = {t for t in f.read().split("\n") if t}
        else:
            lex.protected = lex._default_protected()
        mpath = os.path.join(in_dir, "meta.json")
        if os.path.exists(mpath):
            with open(mpath, "r", encoding="utf-8") as f:
                lex.meta = json.load(f)
        lex.finalize()
        return lex
