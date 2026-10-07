"""Word segmentation for run-together query tokens ("chickenbiryani").

Viterbi dynamic programme over a unigram language model: every split point is
scored by add-k smoothed log-probabilities, unknown pieces pay an extra
per-character penalty, and every emitted word pays a constant insertion
penalty so that a real word is never split into rarer pieces.
"""
from __future__ import annotations

import math
from typing import List, Mapping, Optional, Tuple

_NEG_INF = float("-inf")


class Segmenter:
    """Unigram Viterbi segmenter over a frequency table.

    Args:
        freqs: term -> corpus frequency (not copied).
        total: sum of frequencies (computed if omitted).
        extra: secondary frequency table consulted when ``freqs`` misses
            (e.g. attribute keywords such as "near" or "open").
        k: add-k smoothing constant.
        unknown_penalty: extra negative log-prob per character of an unknown piece.
        word_penalty: negative log-prob paid per emitted word.
        max_word_len: longest piece considered.
        min_part_freq / min_part_len: what counts as a "known" piece in :meth:`split`.
    """

    def __init__(
        self,
        freqs: Mapping[str, int],
        total: Optional[int] = None,
        extra: Optional[Mapping[str, int]] = None,
        k: float = 1.0,
        unknown_penalty: float = 2.0,
        word_penalty: float = 1.0,
        max_word_len: int = 20,
        min_part_freq: int = 2,
        min_part_len: int = 2,
    ) -> None:
        self._freqs = freqs
        self._extra = extra or {}
        n_total = total if total is not None else sum(freqs.values())
        vocab = len(freqs) + len(self._extra)
        self.k = k
        self.unknown_penalty = unknown_penalty
        self.word_penalty = word_penalty
        self.max_word_len = max_word_len
        self.min_part_freq = min_part_freq
        self.min_part_len = min_part_len
        self._log_denom = math.log(max(n_total, 1) + k * max(vocab, 1))
        self._unk_base = math.log(k) - self._log_denom

    def freq(self, w: str) -> int:
        f = self._freqs.get(w)
        if f is None:
            f = self._extra.get(w, 0)
        return f

    def word_logp(self, w: str) -> float:
        """Smoothed unigram log-probability; unknown words pay a length penalty."""
        f = self.freq(w)
        if f > 0:
            return math.log(f + self.k) - self._log_denom
        return self._unk_base - self.unknown_penalty * len(w)

    def segment_scored(self, s: str) -> Tuple[List[str], float]:
        """Best segmentation of ``s`` and its score (sum of logp - word penalties)."""
        n = len(s)
        if n == 0:
            return [], 0.0
        best = [0.0] + [_NEG_INF] * n
        back = [0] * (n + 1)
        maxl = self.max_word_len
        wp = self.word_penalty
        for i in range(1, n + 1):
            bi = _NEG_INF
            bj = i - 1
            for j in range(max(0, i - maxl), i):
                prev = best[j]
                if prev == _NEG_INF:
                    continue
                sc = prev + self.word_logp(s[j:i]) - wp
                if sc > bi:
                    bi, bj = sc, j
            best[i] = bi
            back[i] = bj
        words: List[str] = []
        i = n
        while i > 0:
            j = back[i]
            words.append(s[j:i])
            i = j
        words.reverse()
        return words, best[n]

    def segment(self, s: str) -> List[str]:
        """``"chickenbiryani" -> ["chicken", "biryani"]``; known words stay whole."""
        return self.segment_scored(s)[0]

    def whole_score(self, s: str) -> float:
        """Score of keeping ``s`` as one word (comparable with segment scores)."""
        return self.word_logp(s) - self.word_penalty

    def split(self, s: str) -> Optional[Tuple[List[str], float]]:
        """A segmentation into >= 2 known pieces that beats keeping ``s`` whole.

        Returns ``(parts, score)`` or None. Every part must be a known word
        (frequency >= ``min_part_freq``, length >= ``min_part_len``).
        """
        parts, score = self.segment_scored(s)
        if len(parts) < 2:
            return None
        for p in parts:
            if len(p) < self.min_part_len or self.freq(p) < self.min_part_freq:
                return None
        if score <= self.whole_score(s):
            return None
        return parts, score
