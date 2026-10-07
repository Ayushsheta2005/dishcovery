"""Latency histograms and Prometheus text exposition for the gateway."""
from __future__ import annotations

import bisect
import math
import threading
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

# Microsecond bucket upper bounds: 50us .. ~100s, ~15% apart.
_BOUNDS: List[float] = []
_b = 50.0
while _b < 1e8:
    _BOUNDS.append(_b)
    _b *= 1.15


class LatencyHistogram:
    def __init__(self) -> None:
        self.counts = [0] * (len(_BOUNDS) + 1)
        self.n = 0
        self.total = 0.0
        self.max = 0.0
        self._lock = threading.Lock()

    def record_us(self, us: float) -> None:
        i = bisect.bisect_left(_BOUNDS, us)
        with self._lock:
            self.counts[i] += 1
            self.n += 1
            self.total += us
            self.max = max(self.max, us)

    def record_ms(self, ms: float) -> None:
        self.record_us(ms * 1000.0)

    def percentile_ms(self, q: float) -> float:
        if not self.n:
            return 0.0
        rank = max(1, math.ceil(q * self.n))
        seen = 0
        for i, c in enumerate(self.counts):
            seen += c
            if seen >= rank:
                hi = _BOUNDS[i] if i < len(_BOUNDS) else self.max
                lo = _BOUNDS[i - 1] if i > 0 else 0.0
                return min((lo + hi) / 2.0, self.max) / 1000.0
        return self.max / 1000.0

    def summary(self) -> Dict[str, float]:
        return {"count": self.n, "mean_ms": (self.total / self.n / 1000.0) if self.n else 0.0,
                "p50_ms": self.percentile_ms(0.50), "p95_ms": self.percentile_ms(0.95),
                "p99_ms": self.percentile_ms(0.99), "max_ms": self.max / 1000.0}


class Metrics:
    """Named histograms + counters; one instance per gateway."""

    def __init__(self) -> None:
        self.hist: Dict[str, LatencyHistogram] = defaultdict(LatencyHistogram)
        self.counters: Dict[str, float] = defaultdict(float)

    def observe_ms(self, name: str, ms: float) -> None:
        self.hist[name].record_ms(ms)

    def inc(self, name: str, v: float = 1.0) -> None:
        self.counters[name] += v

    def observe_timings(self, timings: Dict[str, float], prefix: str = "stage_") -> None:
        for k, v in timings.items():
            if isinstance(v, (int, float)):
                self.observe_ms(prefix + k, float(v))

    def summary(self) -> Dict[str, dict]:
        return {"latency": {k: h.summary() for k, h in sorted(self.hist.items())},
                "counters": dict(sorted(self.counters.items()))}

    def prometheus(self) -> str:
        lines = []
        for name, h in sorted(self.hist.items()):
            m = f"dishcovery_{name}_seconds"
            lines.append(f"# TYPE {m} summary")
            for q in (0.5, 0.95, 0.99):
                lines.append(f'{m}{{quantile="{q}"}} {h.percentile_ms(q) / 1000.0:.6f}')
            lines.append(f"{m}_sum {h.total / 1e6:.6f}")
            lines.append(f"{m}_count {h.n}")
        for name, v in sorted(self.counters.items()):
            lines.append(f"# TYPE dishcovery_{name}_total counter")
            lines.append(f"dishcovery_{name}_total {v}")
        return "\n".join(lines) + "\n"
