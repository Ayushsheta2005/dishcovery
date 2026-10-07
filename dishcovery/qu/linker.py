"""Entity linking: greedy longest-match of token n-grams against the gazetteer.

Overlaps resolve left-to-right by length; for an equal span the gazetteer's
own ordering decides (chain > dish > cuisine > restaurant > ingredient, then
weight), so "pizza hut" links to the chain while "pizza" alone links to the
dish.
"""
from __future__ import annotations

from typing import List, Sequence

from dishcovery.qu.lexicon import Lexicon
from dishcovery.qu.types import Entity


class EntityLinker:
    def __init__(self, lexicon: Lexicon, max_n: int = 5) -> None:
        self.lex = lexicon
        self.max_n = max(1, min(max_n, max(lexicon.max_ngram, 1)))

    def link(self, tokens: Sequence[str]) -> List[Entity]:
        out: List[Entity] = []
        i, n = 0, len(tokens)
        while i < n:
            found = None
            for L in range(min(self.max_n, n - i), 0, -1):
                surface = " ".join(tokens[i:i + L])
                entries = self.lex.entries(surface)
                if not entries and L == 1:
                    e = self.lex.nospace.get(surface)
                    entries = [e] if e else []
                if entries:
                    e = entries[0]
                    found = Entity(start=i, end=i + L, text=surface, type=e.type, canonical=e.canonical,
                                   score=float(e.weight) * (1.0 + 0.1 * (L - 1)))
                    break
            if found is not None:
                out.append(found)
                i = found.end
            else:
                i += 1
        return out
