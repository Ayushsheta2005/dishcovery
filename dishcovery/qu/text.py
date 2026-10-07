"""Surface-text normalisation shared by query understanding, labelling and data prep.

Mirrors the first stages of the C++ analyzer (NFKC/accent strip, case fold,
apostrophe removal, split on non-alphanumerics) so that Python-side lexicons
see the same tokens the engine indexes — minus stemming, which only the
engine applies.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

_APOSTROPHES = "'’‘`´"
_APOS_RE = re.compile("[" + re.escape(_APOSTROPHES) + "]")
# Unicode-aware: keep letters/digits of any script, split on everything else.
_NON_ALNUM_RE = re.compile(r"[\W_]+", re.UNICODE)


def fold(s: str) -> str:
    """Unicode compatibility-normalise, strip accents and case-fold."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.casefold()


def normalize(s: str) -> str:
    """'Domino’s  Pizza!' -> 'dominos pizza'; 'Jalapeño-Poppers' -> 'jalapeno poppers'."""
    s = fold(s)
    s = _APOS_RE.sub("", s)
    s = _NON_ALNUM_RE.sub(" ", s)
    return " ".join(s.split())


def tokenize(s: str) -> List[str]:
    return normalize(s).split()
