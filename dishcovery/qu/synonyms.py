"""Curated synonym / transliteration table used for query expansion."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from dishcovery.qu.text import normalize

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synonyms.yaml")


@dataclass
class SynonymGroup:
    """A set of interchangeable surface forms (normalised)."""
    terms: List[str]
    weight: float = 1.0
    type: str = "general"   # ingredient | dish | cuisine | drink | general


@dataclass
class SynonymTable:
    """Phrase -> weighted alternatives, precomputed for O(1) expansion.

    Expansion is symmetric within a group and never transitive across groups;
    when a phrase occurs in several groups each alternative keeps its highest
    group weight.
    """
    groups: List[SynonymGroup] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._index: Dict[str, List[Tuple[str, float]]] = {}
        self.max_len = 1
        merged: Dict[str, Dict[str, float]] = {}
        for g in self.groups:
            for t in g.terms:
                alts = merged.setdefault(t, {})
                for u in g.terms:
                    if u != t and alts.get(u, -1.0) < g.weight:
                        alts[u] = g.weight
                self.max_len = max(self.max_len, len(t.split()))
        for t, alts in merged.items():
            self._index[t] = sorted(alts.items(), key=lambda kv: (-kv[1], kv[0]))

    # ------------------------------------------------------------- loading
    @classmethod
    def from_groups(cls, groups: Iterable[Union[dict, Sequence[str], SynonymGroup]]) -> "SynonymTable":
        """Build from dicts ``{terms, weight?, type?}``, plain term lists or groups."""
        out: List[SynonymGroup] = []
        for g in groups:
            if isinstance(g, SynonymGroup):
                raw_terms, weight, typ = g.terms, g.weight, g.type
            elif isinstance(g, dict):
                raw_terms = g.get("terms") or []
                weight = float(g.get("weight", 1.0))
                typ = str(g.get("type", "general"))
            else:
                raw_terms, weight, typ = list(g), 1.0, "general"
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"synonym weight out of range: {weight} for {raw_terms}")
            terms: List[str] = []
            for t in raw_terms:
                n = normalize(str(t))
                if n and n not in terms:
                    terms.append(n)
            if len(terms) >= 2:
                out.append(SynonymGroup(terms, weight, typ))
        return cls(out)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "SynonymTable":
        """Load a YAML list of groups (defaults to the packaged synonyms.yaml)."""
        import yaml

        with open(path or DEFAULT_PATH, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
        if isinstance(data, dict):
            data = data.get("groups", [])
        return cls.from_groups(data or [])

    def save(self, path: str) -> None:
        import yaml

        rows = [{"type": g.type, "weight": g.weight, "terms": list(g.terms)} for g in self.groups]
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(rows, f, allow_unicode=True, sort_keys=False, default_flow_style=None)

    # ----------------------------------------------------------- expansion
    def __contains__(self, phrase: object) -> bool:
        return phrase in self._index

    def __len__(self) -> int:
        return len(self.groups)

    def expand(self, phrase: str) -> List[Tuple[str, float]]:
        """Weighted alternatives for ``phrase`` (normalised first), best first."""
        alts = self._index.get(phrase)
        if alts is None:
            alts = self._index.get(normalize(phrase), [])
        return list(alts)

    def lookup_normalized(self, phrase: str) -> List[Tuple[str, float]]:
        """Like :meth:`expand` for an already-normalised phrase; no copy (hot path)."""
        return self._index.get(phrase, _EMPTY)

    def terms_of_type(self, typ: str) -> List[Tuple[str, str, float]]:
        """``(term, canonical, weight)`` for every term of groups of ``typ``.

        The canonical form is the first term of the group.
        """
        out = []
        for g in self.groups:
            if g.type == typ:
                for t in g.terms:
                    out.append((t, g.terms[0], g.weight))
        return out

    def all_terms(self) -> List[str]:
        return list(self._index)


_EMPTY: List[Tuple[str, float]] = []
