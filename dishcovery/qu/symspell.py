"""SymSpell spelling correction (symmetric delete), implemented from scratch.

The dictionary side pre-computes every string reachable from a word's
``prefix_length``-character prefix by deleting up to ``max_edit_distance``
characters. At lookup time the same deletes are generated from the input, so
candidates are found by hash lookups instead of edit-distance scans; each
candidate is then verified with the optimal-string-alignment (restricted
Damerau-Levenshtein) distance, which counts an adjacent transposition as one
edit (``briyani`` -> ``biryani`` is distance 1).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

VERBOSITIES = ("top", "closest", "all")


class Suggestion(NamedTuple):
    """A correction candidate."""
    term: str
    distance: int
    freq: int


def osa_distance(a: str, b: str, max_distance: int) -> int:
    """Optimal-string-alignment distance between ``a`` and ``b``.

    Returns ``max_distance + 1`` as soon as the distance is known to exceed
    ``max_distance`` (row-minimum early exit), so callers can test ``> max``.
    """
    if a == b:
        return 0
    over = max_distance + 1
    la, lb = len(a), len(b)
    if abs(la - lb) > max_distance:
        return over
    # Trim the common suffix and prefix: they never contribute edits.
    while la and lb and a[la - 1] == b[lb - 1]:
        la -= 1
        lb -= 1
    start = 0
    while start < la and start < lb and a[start] == b[start]:
        start += 1
    a = a[start:la]
    b = b[start:lb]
    la -= start
    lb -= start
    if la == 0 or lb == 0:
        d = la or lb
        return d if d <= max_distance else over

    prev2: List[int] = []
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        ca = a[i - 1]
        cur = [i] + [0] * lb
        row_min = i
        for j in range(1, lb + 1):
            cb = b[j - 1]
            v = prev[j - 1] + (ca != cb)
            t = prev[j] + 1
            if t < v:
                v = t
            t = cur[j - 1] + 1
            if t < v:
                v = t
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                t = prev2[j - 2] + 1
                if t < v:
                    v = t
            cur[j] = v
            if v < row_min:
                row_min = v
        if row_min > max_distance:
            return over
        prev2, prev = prev, cur
    d = prev[lb]
    return d if d <= max_distance else over


class SymSpell:
    """Symmetric-delete spelling corrector.

    >>> s = SymSpell()
    >>> s.add_many([("biryani", 500), ("chicken", 900)])
    2
    >>> s.lookup("briyani")[0]
    Suggestion(term='biryani', distance=1, freq=500)
    """

    def __init__(self, max_edit_distance: int = 2, prefix_length: int = 7) -> None:
        if max_edit_distance < 0:
            raise ValueError("max_edit_distance must be >= 0")
        if prefix_length <= max_edit_distance:
            raise ValueError("prefix_length must exceed max_edit_distance")
        self.max_edit_distance = max_edit_distance
        self.prefix_length = prefix_length
        self._words: Dict[str, int] = {}
        # delete-variant of a word prefix -> words producing it
        self._deletes: Dict[str, List[str]] = {}
        self._max_length = 0

    # ------------------------------------------------------------------ build
    def __len__(self) -> int:
        return len(self._words)

    def __contains__(self, term: object) -> bool:
        return term in self._words

    def freq(self, term: str) -> int:
        return self._words.get(term, 0)

    def add(self, term: str, freq: int = 1) -> bool:
        """Add ``term`` (or increase its count). Returns True if it was new."""
        if not term or freq <= 0:
            return False
        if term in self._words:
            self._words[term] += int(freq)
            return False
        self._words[term] = int(freq)
        if len(term) > self._max_length:
            self._max_length = len(term)
        deletes = self._deletes
        for d in self._prefix_deletes(term[: self.prefix_length]):
            lst = deletes.get(d)
            if lst is None:
                deletes[d] = [term]
            else:
                lst.append(term)
        return True

    def add_many(self, items: Iterable[Tuple[str, int]]) -> int:
        n = 0
        for term, freq in items:
            n += self.add(term, freq)
        return n

    def _prefix_deletes(self, key: str) -> Set[str]:
        out = {key}
        frontier = [key]
        for _ in range(self.max_edit_distance):
            nxt = []
            for w in frontier:
                for i in range(len(w)):
                    d = w[:i] + w[i + 1:]
                    if d not in out:
                        out.add(d)
                        nxt.append(d)
            if not nxt:
                break
            frontier = nxt
        return out

    # ----------------------------------------------------------------- lookup
    def lookup(
        self,
        term: str,
        max_edits: Optional[int] = None,
        verbosity: str = "closest",
        exclude_term: bool = False,
    ) -> List[Suggestion]:
        """Suggestions for ``term`` within ``max_edits`` (capped at the index's max).

        verbosity: ``top`` -> the single best suggestion; ``closest`` -> all at
        the smallest distance found; ``all`` -> everything within ``max_edits``.
        Results are ordered by distance asc, then frequency desc. An in-vocab
        ``term`` is returned as a distance-0 suggestion (and stops the search
        unless verbosity is ``all``) unless ``exclude_term`` is set.
        """
        if verbosity not in VERBOSITIES:
            raise ValueError(f"verbosity must be one of {VERBOSITIES}")
        max_ed = self.max_edit_distance if max_edits is None else min(max_edits, self.max_edit_distance)
        if max_ed < 0:
            return []
        tlen = len(term)
        if tlen - max_ed > self._max_length:
            return []
        suggestions: List[Suggestion] = []
        if not exclude_term:
            f = self._words.get(term)
            if f is not None:
                suggestions.append(Suggestion(term, 0, f))
                if verbosity != "all":
                    return suggestions
        if max_ed == 0:
            return suggestions

        words = self._words
        deletes = self._deletes
        plen = self.prefix_length
        bound = max_ed  # shrinks for top/closest once a closer match is found
        considered: Set[str] = {term}
        tprefix_len = min(tlen, plen)
        candidates = [term[:tprefix_len]]
        seen_deletes = {candidates[0]}
        ptr = 0
        while ptr < len(candidates):
            cand = candidates[ptr]
            ptr += 1
            clen = len(cand)
            len_diff = tprefix_len - clen
            if len_diff > bound:
                # candidates are generated in order of increasing deletes
                if verbosity == "all":
                    continue
                break
            hits = deletes.get(cand)
            if hits:
                for sugg in hits:
                    if sugg in considered:
                        continue
                    slen = len(sugg)
                    if abs(slen - tlen) > bound:
                        continue
                    if min(slen, plen) - clen > bound:
                        continue
                    considered.add(sugg)
                    d = osa_distance(term, sugg, bound)
                    if d > bound:
                        continue
                    sf = words[sugg]
                    if verbosity == "top":
                        if not suggestions:
                            suggestions.append(Suggestion(sugg, d, sf))
                        else:
                            best = suggestions[0]
                            if d < best.distance or (d == best.distance and sf > best.freq):
                                suggestions[0] = Suggestion(sugg, d, sf)
                        bound = suggestions[0].distance
                    elif verbosity == "closest":
                        if suggestions and d < suggestions[0].distance:
                            suggestions = []
                        suggestions.append(Suggestion(sugg, d, sf))
                        bound = d
                    else:
                        suggestions.append(Suggestion(sugg, d, sf))
            if len_diff < max_ed and clen <= plen:
                if verbosity != "all" and len_diff >= bound:
                    continue
                for i in range(clen):
                    dl = cand[:i] + cand[i + 1:]
                    if dl not in seen_deletes:
                        seen_deletes.add(dl)
                        candidates.append(dl)
        if verbosity == "closest" and suggestions:
            dmin = min(s.distance for s in suggestions)
            suggestions = [s for s in suggestions if s.distance == dmin]
        suggestions.sort(key=lambda s: (s.distance, -s.freq, s.term))
        return suggestions

    def best_correction(
        self,
        term: str,
        term_freq: int = 0,
        min_ratio: float = 10.0,
        trust_freq: int = 10,
        min_length: int = 4,
        max_edits: Optional[int] = None,
    ) -> Optional[Suggestion]:
        """The correction policy used by query understanding.

        Returns a suggestion only when ``term`` looks like a misspelling: its
        own corpus frequency ``term_freq`` is below ``trust_freq`` and a
        candidate at the closest distance is at least ``min_ratio`` times more
        frequent. Short terms allow 1 edit, terms of length >= 5 allow 2.
        """
        if len(term) < min_length or term_freq >= trust_freq:
            return None
        if max_edits is None:
            max_edits = 1 if len(term) <= 4 else 2
        need = min_ratio * max(term_freq, 1)
        for s in self.lookup(term, max_edits, "closest", exclude_term=True):
            if s.freq >= need:
                return s
        return None
