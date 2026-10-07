"""Query understanding pipeline.

    raw -> normalize -> spell-correct / segment -> extract attributes
        -> entity-link -> classify intent -> expand synonyms -> StructuredQuery

Each stage can be disabled through :class:`QUConfig` (used by the ablations).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from dishcovery.qu.extract import extract
from dishcovery.qu.intent import IntentClassifier
from dishcovery.qu.lexicon import Lexicon
from dishcovery.qu.linker import EntityLinker
from dishcovery.qu.segment import Segmenter
from dishcovery.qu.symspell import SymSpell
from dishcovery.qu.synonyms import SynonymTable
from dishcovery.qu.text import normalize
from dishcovery.qu.types import QueryFilters, QueryTerm, StructuredQuery

# Attribute vocabulary: never spell-corrected, and known to the segmenter so
# "nearme" splits into "near me".
_ATTRIBUTE_WORDS = {
    "near": 500, "me": 500, "open": 500, "now": 500, "cheap": 300, "vegan": 300, "veg": 300, "vegetarian": 300,
    "halal": 100, "jain": 50, "gluten": 100, "free": 200, "late": 200, "night": 200, "best": 300, "top": 200,
    "rated": 200, "fast": 200, "quick": 100, "delivery": 300, "under": 200, "dollars": 100, "stars": 100,
    "nearby": 200, "budget": 100, "affordable": 100, "breakfast": 300, "lunch": 300, "dinner": 300,
    "brunch": 100, "food": 500, "restaurant": 300, "restaurants": 300, "place": 100, "places": 100,
}


@dataclass
class QUConfig:
    spell: bool = True
    segment: bool = True
    extract: bool = True
    link: bool = True
    intent: bool = True
    expand: bool = True
    synonym_weight: float = 0.6
    spell_original_weight: float = 0.0
    entity_boost: float = 1.0
    spell_vocab_size: int = 60000
    spell_min_ratio: float = 10.0
    spell_trust_freq: int = 10


class QueryUnderstanding:
    def __init__(self, lexicon: Lexicon, synonyms: Optional[SynonymTable] = None,
                 intent_model: Optional[IntentClassifier] = None, config: Optional[QUConfig] = None) -> None:
        self.lex = lexicon
        self.syn = synonyms if synonyms is not None else SynonymTable.load()
        self.intent_model = intent_model or IntentClassifier()
        self.cfg = config or QUConfig()
        self.linker = EntityLinker(lexicon)
        self.segmenter = Segmenter(lexicon.freqs, total=lexicon.total or None, extra=_ATTRIBUTE_WORDS)
        self.spell = SymSpell(max_edit_distance=2, prefix_length=7)
        vocab = sorted(((t, c) for t, c in lexicon.freqs.items() if t.isalpha() and len(t) >= 3),
                       key=lambda kv: -kv[1])[: self.cfg.spell_vocab_size]
        self.spell.add_many(vocab)
        for w, c in _ATTRIBUTE_WORDS.items():
            if w not in self.spell:
                self.spell.add(w, c)

    # ------------------------------------------------------------- build/load
    @classmethod
    def build(cls, docs_path: str, out_dir: str, taxonomy_path: Optional[str] = None,
              max_docs: Optional[int] = None, seed: int = 0, config: Optional[QUConfig] = None
              ) -> "QueryUnderstanding":
        syn = SynonymTable.load()
        lex = Lexicon.build(docs_path, taxonomy_path=taxonomy_path, max_docs=max_docs, synonyms=syn)
        os.makedirs(out_dir, exist_ok=True)
        lex.save(os.path.join(out_dir, "lexicon"))
        intent = IntentClassifier().fit(lex, seed=seed)
        intent.save(os.path.join(out_dir, "intent.pkl"))
        cfg = config or QUConfig()
        with open(os.path.join(out_dir, "qu_config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        return cls(lex, syn, intent, cfg)

    @classmethod
    def load(cls, art_dir: str, config: Optional[QUConfig] = None) -> "QueryUnderstanding":
        lex = Lexicon.load(os.path.join(art_dir, "lexicon"))
        intent_path = os.path.join(art_dir, "intent.pkl")
        intent = IntentClassifier.load(intent_path) if os.path.exists(intent_path) else None
        if config is None:
            cpath = os.path.join(art_dir, "qu_config.json")
            config = QUConfig(**json.load(open(cpath))) if os.path.exists(cpath) else QUConfig()
        return cls(lex, SynonymTable.load(), intent, config)

    def with_config(self, **overrides) -> "QueryUnderstanding":
        """Cheap copy sharing the heavy indexes, with some stages toggled."""
        clone = object.__new__(QueryUnderstanding)
        clone.__dict__.update(self.__dict__)
        cfg = QUConfig(**{**asdict(self.cfg), **overrides})
        clone.cfg = cfg
        return clone

    # ------------------------------------------------------------- correction
    def _is_trusted(self, tok: str) -> bool:
        return (tok in self.lex.protected or tok in _ATTRIBUTE_WORDS or not tok.isalpha() or len(tok) < 4
                or self.lex.freq(tok) >= self.cfg.spell_trust_freq or bool(self.lex.entries(tok)))

    def _correct(self, tokens: List[str]) -> Tuple[List[str], List[str], List[Tuple[str, str, str]]]:
        """Returns (tokens, per-token source, corrections)."""
        out: List[str] = []
        src: List[str] = []
        corrections: List[Tuple[str, str, str]] = []
        for tok in tokens:
            if self._is_trusted(tok):
                out.append(tok)
                src.append("query")
                continue
            # Brand typed without spaces ("pizzahut").
            ns = self.lex.nospace.get(tok)
            if ns is not None and self.cfg.segment:
                parts = ns.surface.split()
                out.extend(parts)
                src.extend(["segment"] * len(parts))
                corrections.append((tok, ns.surface, "segment"))
                continue
            sug = None
            if self.cfg.spell:
                sug = self.spell.best_correction(tok, self.lex.freq(tok), min_ratio=self.cfg.spell_min_ratio,
                                                 trust_freq=self.cfg.spell_trust_freq)
            seg = self.segmenter.split(tok) if self.cfg.segment and len(tok) >= 6 else None
            # Prefer a single-edit correction; otherwise a clean segmentation;
            # otherwise a 2-edit correction.
            if sug is not None and sug.distance == 1:
                out.append(sug.term)
                src.append("spell")
                corrections.append((tok, sug.term, "spell"))
            elif seg is not None:
                parts = seg[0]
                fixed = []
                for p in parts:  # segments may still carry a typo ("chikenbiryani")
                    s2 = None
                    if self.cfg.spell and not self._is_trusted(p):
                        s2 = self.spell.best_correction(p, self.lex.freq(p))
                    fixed.append(s2.term if s2 else p)
                out.extend(fixed)
                src.extend(["segment"] * len(fixed))
                corrections.append((tok, " ".join(fixed), "segment"))
            elif sug is not None:
                out.append(sug.term)
                src.append("spell")
                corrections.append((tok, sug.term, "spell"))
            else:
                out.append(tok)
                src.append("query")
        return out, src, corrections

    # ------------------------------------------------------------------ parse
    def parse(self, query: str) -> StructuredQuery:
        cfg = self.cfg
        sq = StructuredQuery(raw=query)
        sq.normalized = normalize(query)
        tokens = sq.normalized.split()
        if cfg.spell or cfg.segment:
            tokens, sources, sq.corrections = self._correct(tokens)
        else:
            sources = ["query"] * len(tokens)
        sq.corrected = " ".join(tokens)

        if cfg.extract:
            ex = extract(tokens)
            sq.filters = ex.filters
            sq.geo = ex.geo
            sq.radius_km = ex.radius_km
            sq.sort_hint = ex.sort_hint
            sq.meal_time = ex.meal_time
            content = ex.content
        else:
            sq.filters = QueryFilters()
            content = tokens
        # Map content tokens back to their correction source.
        src_of: Dict[str, str] = {}
        for t, s in zip(tokens, sources):
            src_of.setdefault(t, s)
        sq.tokens = list(content)

        sq.entities = self.linker.link(content) if cfg.link else []
        if cfg.intent:
            sq.intent, sq.intent_scores = self.intent_model.predict(content, sq.entities,
                                                                    has_diet=bool(sq.filters.diet))
        else:
            sq.intent, sq.intent_scores = ("dish" if content else "attribute"), {}

        # Chain intent: hard brand filter from the dominant chain entity.
        if sq.intent == "chain":
            chains = [e for e in sq.entities if e.type == "chain"]
            if chains:
                sq.filters.chain = max(chains, key=lambda e: (e.end - e.start, e.score)).canonical
        if sq.intent == "cuisine":
            cz = [e for e in sq.entities if e.type == "cuisine"]
            if cz:
                sq.filters.cuisine = cz[0].canonical

        # Weighted terms: one per content token, then expansions.
        terms: List[QueryTerm] = []
        for i, t in enumerate(content):
            terms.append(QueryTerm(t, 1.0, i, src_of.get(t, "query")))
        if cfg.spell_original_weight > 0:
            for orig, fixed, kind in sq.corrections:
                if kind == "spell" and fixed in content:
                    terms.append(QueryTerm(orig, cfg.spell_original_weight, content.index(fixed), "spell"))
        if cfg.expand:
            seen = {t.text for t in terms}
            spans = [(e.start, e.end) for e in sq.entities if e.end - e.start > 1]
            spans += [(i, i + 1) for i in range(len(content))]
            for start, end in spans:
                phrase = " ".join(content[start:end])
                for alt, w in self.syn.lookup_normalized(phrase):
                    if alt in seen or alt == phrase:
                        continue
                    seen.add(alt)
                    terms.append(QueryTerm(alt, round(cfg.synonym_weight * w, 4), start, "synonym"))
        sq.terms = terms
        return sq
