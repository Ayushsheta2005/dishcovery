"""Significance testing and confidence intervals for per-query metric values.

All procedures operate on per-query values (one number per query, e.g. NDCG@10)
and are deterministic given ``seed``.

* ``bootstrap_ci`` — percentile bootstrap CI of the mean over queries.
* ``paired_randomization_test`` — two-sided sign-flip permutation test on the
  per-query differences (Fisher's randomization test, recommended for IR by
  Smucker, Allan & Carterette, CIKM 2007). Exact when ``2**n <= n_resamples``.
* ``paired_ttest`` — Student's paired t-test (scipy), reported for comparison.
* ``holm_bonferroni`` — step-down family-wise error correction.
"""
from __future__ import annotations

import itertools
import math
from typing import List, Mapping, NamedTuple, Sequence, Union

import numpy as np

# Keep resampling matrices at most this many cells per chunk (~16 MB of int64).
_CHUNK_CELLS = 2_000_000


class ConfidenceInterval(NamedTuple):
    mean: float
    lo: float
    hi: float

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2.0


class TTestResult(NamedTuple):
    statistic: float
    pvalue: float


def _as_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


def bootstrap_ci(values: Sequence[float], n_resamples: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> ConfidenceInterval:
    """Percentile bootstrap ``1 - alpha`` CI of the mean of ``values``.

    Queries are resampled with replacement ``n_resamples`` times. Returns
    ``(nan, nan, nan)`` for no values and a degenerate interval for one value.
    """
    x = _as_array(values)
    n = x.size
    if n == 0:
        nan = float("nan")
        return ConfidenceInterval(nan, nan, nan)
    mean = float(x.mean())
    if n == 1 or np.all(x == x[0]):
        return ConfidenceInterval(mean, mean, mean)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    chunk = max(1, _CHUNK_CELLS // n)
    for start in range(0, n_resamples, chunk):
        stop = min(n_resamples, start + chunk)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return ConfidenceInterval(mean, float(lo), float(hi))


def paired_randomization_test(a: Sequence[float], b: Sequence[float], n: int = 10000,
                              seed: int = 0) -> float:
    """Two-sided paired randomization (sign-flip) test of ``mean(a) == mean(b)``.

    Under H0 the system labels are exchangeable per query, so each per-query
    difference ``a_i - b_i`` is equally likely to have either sign. The p-value
    is the fraction of sign assignments whose ``|mean difference|`` is at least
    the observed one. When ``2**len(a) <= n`` all assignments are enumerated
    (exact test); otherwise ``n`` random assignments are drawn and the
    conservative ``(count + 1) / (n + 1)`` estimate is returned.
    """
    x, y = _as_array(a), _as_array(b)
    if x.shape != y.shape:
        raise ValueError(f"paired samples must have equal length ({x.size} != {y.size})")
    d = x - y
    m = d.size
    if m == 0 or np.all(np.abs(d) < 1e-15):
        return 1.0
    observed = abs(d.mean())
    tol = 1e-12 * max(1.0, observed)
    if m <= 30 and 2 ** m <= n:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=m)))
        stats = np.abs((signs * d).mean(axis=1))
        return float(np.count_nonzero(stats >= observed - tol)) / signs.shape[0]
    rng = np.random.default_rng(seed)
    count = 0
    chunk = max(1, _CHUNK_CELLS // m)
    for start in range(0, n, chunk):
        rows = min(n, start + chunk) - start
        signs = rng.integers(0, 2, size=(rows, m)) * 2.0 - 1.0
        stats = np.abs((signs * d).mean(axis=1))
        count += int(np.count_nonzero(stats >= observed - tol))
    return (count + 1.0) / (n + 1.0)


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> TTestResult:
    """Two-sided paired t-test (``scipy.stats.ttest_rel``).

    Degenerate cases scipy reports as ``nan`` are resolved explicitly: all
    differences zero -> ``(0, 1)``; constant non-zero difference -> ``(±inf, 0)``.
    """
    from scipy import stats as sps  # local import: scipy is only needed here

    x, y = _as_array(a), _as_array(b)
    if x.shape != y.shape:
        raise ValueError("paired samples must have equal length")
    d = x - y
    if d.size < 2:
        return TTestResult(float("nan"), float("nan"))
    if np.all(np.abs(d - d[0]) < 1e-15):
        if abs(d[0]) < 1e-15:
            return TTestResult(0.0, 1.0)
        return TTestResult(math.copysign(math.inf, d[0]), 0.0)
    res = sps.ttest_rel(x, y)
    return TTestResult(float(res.statistic), float(res.pvalue))


PValues = Union[Sequence[float], Mapping[str, float]]


def holm_bonferroni(pvalues: PValues) -> Union[List[float], dict]:
    """Holm–Bonferroni adjusted p-values (step-down, monotone, capped at 1).

    Reject hypothesis ``i`` at family-wise level ``alpha`` iff
    ``adjusted[i] < alpha``. Accepts a sequence (returns a list in the same
    order) or a mapping (returns a dict with the same keys). ``nan`` p-values
    are left as ``nan`` and do not count towards the family size.
    """
    if isinstance(pvalues, Mapping):
        keys = list(pvalues.keys())
        adj = holm_bonferroni([pvalues[k] for k in keys])
        return dict(zip(keys, adj))
    ps = [float(p) for p in pvalues]
    valid = [i for i, p in enumerate(ps) if not math.isnan(p)]
    order = sorted(valid, key=lambda i: ps[i])
    m = len(order)
    out = [float("nan")] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        out[i] = running
    return out
