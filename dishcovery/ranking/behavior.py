"""Behavioural features (historical CTR / conversion) from click logs.

Built from a *history* window of simulated impressions that is disjoint from
the sessions used to derive click training labels, so the model cannot learn
to copy its own labels.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

from dishcovery.labels.clicks import CtrTable, ctr_table, load_impressions
from dishcovery.qu.text import normalize


class BehaviorStore:
    def __init__(self, table: Optional[CtrTable] = None) -> None:
        self.table = table

    @classmethod
    def from_impressions_file(cls, path: str, alpha: float = 1.0, beta: float = 20.0) -> "BehaviorStore":
        return cls(ctr_table(load_impressions(path), alpha=alpha, beta=beta))

    @classmethod
    def maybe_load(cls, path: Optional[str]) -> "BehaviorStore":
        if path and os.path.exists(path):
            return cls.from_impressions_file(path)
        return cls(None)

    def lookup(self, query: str, doc_id: str) -> Dict[str, float]:
        if self.table is None:
            return {}
        row = dict(self.table.lookup(normalize(query), doc_id))
        d = self.table.docs.get(doc_id)
        row["doc_ctr"] = d["ctr_smoothed"] if d else self.table.global_ctr
        return row
