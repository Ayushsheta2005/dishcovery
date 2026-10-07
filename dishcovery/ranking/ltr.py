"""LambdaMART learning-to-rank.

Backends:
  * ``lightgbm`` (default when importable): LGBMRanker(objective="lambdarank").
  * ``numpy``: a from-scratch LambdaMART — gradient-boosted regression trees
    fit to LambdaRank gradients (Burges, 2010) with Newton leaf values, using
    scikit-learn's DecisionTreeRegressor as the weak learner. Slower, but it
    runs anywhere (e.g. macOS without libomp) and makes the algorithm explicit.

Training groups are queries; labels are graded 0..3. Model files store the
feature list so serving can verify the column order.
"""
from __future__ import annotations

import json
import math
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dishcovery.ranking.features import FEATURE_VERSION, FEATURES


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False


# ----------------------------------------------------------------- numpy LambdaMART

def _dcg_discounts(n: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, n + 2))


def _lambdas(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """LambdaRank gradients/hessians for one query (NDCG@k-weighted pairwise logistic)."""
    n = len(scores)
    lam = np.zeros(n)
    hess = np.zeros(n)
    if n < 2:
        return lam, hess
    gains = (2.0 ** labels) - 1.0
    ideal = np.sort(gains)[::-1][:k]
    idcg = float((ideal * _dcg_discounts(len(ideal))).sum())
    if idcg <= 0:
        return lam, hess
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    disc = np.where(rank < k, 1.0 / np.log2(rank + 2.0), 0.0)
    # Pairwise (i more relevant than j) terms, vectorised over the n x n grid.
    mask = labels[:, None] > labels[None, :]
    delta = np.abs((gains[:, None] - gains[None, :]) * (disc[:, None] - disc[None, :])) / idcg
    sig = 1.0 / (1.0 + np.exp(np.clip(scores[:, None] - scores[None, :], -50, 50)))
    m = np.where(mask, delta * sig, 0.0)
    h = np.where(mask, delta * sig * (1.0 - sig), 0.0)
    lam = m.sum(axis=1) - m.sum(axis=0)
    hess = h.sum(axis=1) + h.sum(axis=0)
    return lam, hess


class NumpyLambdaMART:
    def __init__(self, n_trees: int = 200, learning_rate: float = 0.1, max_leaf_nodes: int = 31,
                 min_samples_leaf: int = 20, k: int = 10, seed: int = 0) -> None:
        self.n_trees, self.lr, self.max_leaf_nodes = n_trees, learning_rate, max_leaf_nodes
        self.min_samples_leaf, self.k, self.seed = min_samples_leaf, k, seed
        self.trees: list = []

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Sequence[int], X_val=None, y_val=None, groups_val=None,
            early_stopping: int = 30, log=print) -> "NumpyLambdaMART":
        from sklearn.tree import DecisionTreeRegressor

        bounds = np.cumsum([0] + list(groups))
        F = np.zeros(len(y))
        best, best_iter, val_F = -1.0, 0, None
        if X_val is not None:
            val_F = np.zeros(len(y_val))
        for t in range(self.n_trees):
            lam = np.zeros(len(y))
            hess = np.zeros(len(y))
            for a, b in zip(bounds[:-1], bounds[1:]):
                lam[a:b], hess[a:b] = _lambdas(F[a:b], y[a:b], self.k)
            w = np.maximum(hess, 1e-9)
            tree = DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes, min_samples_leaf=self.min_samples_leaf,
                                         random_state=self.seed + t)
            # Newton step: fitting lam/w with weights w makes each leaf = Σλ / Σw.
            tree.fit(X, lam / w, sample_weight=w)
            self.trees.append(tree)
            F += self.lr * tree.predict(X)
            if X_val is not None:
                val_F += self.lr * tree.predict(X_val)
                score = mean_ndcg(val_F, y_val, groups_val, self.k)
                if score > best:
                    best, best_iter = score, t + 1
                elif t + 1 - best_iter >= early_stopping:
                    log(f"early stop at {t + 1}, best iter {best_iter} (val ndcg@{self.k}={best:.4f})")
                    break
        if best_iter:
            self.trees = self.trees[:best_iter]
        return self

    def _pack(self) -> None:
        """Flatten every tree into padded (n_trees x max_nodes) arrays for vectorised traversal."""
        T = len(self.trees)
        m = max(t.tree_.node_count for t in self.trees)
        self._feat = np.full((T, m), -2, dtype=np.int64)
        self._thr = np.zeros((T, m))
        self._left = np.zeros((T, m), dtype=np.int64)
        self._right = np.zeros((T, m), dtype=np.int64)
        self._val = np.zeros((T, m))
        self._depth = max(t.tree_.max_depth for t in self.trees)
        for i, t in enumerate(self.trees):
            tr = t.tree_
            n = tr.node_count
            self._feat[i, :n] = tr.feature
            self._thr[i, :n] = tr.threshold
            self._left[i, :n] = tr.children_left
            self._right[i, :n] = tr.children_right
            self._val[i, :n] = tr.value.reshape(n, -1)[:, 0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """All trees traversed at once: one fancy-indexing step per depth level instead of
        one scikit-learn call per tree (~50x faster for a few hundred candidates)."""
        X = np.asarray(X, dtype=np.float64)
        if not self.trees or len(X) == 0:
            return np.zeros(len(X))
        if getattr(self, "_feat", None) is None or self._feat.shape[0] != len(self.trees):
            self._pack()
        n, T = len(X), len(self.trees)
        m = self._feat.shape[1]
        feat, thr = self._feat.ravel(), self._thr.ravel()
        left, right, val = self._left.ravel(), self._right.ravel(), self._val.ravel()
        base = (np.arange(T, dtype=np.int64) * m)[None, :]          # flat offset of each tree
        xflat = X.ravel()
        xrow = (np.arange(n, dtype=np.int64) * X.shape[1])[:, None]
        pos = np.broadcast_to(base, (n, T)).copy()                  # flat node index per (row, tree)
        for _ in range(self._depth + 1):
            f = np.take(feat, pos)
            internal = f >= 0
            if not internal.any():
                break
            x = np.take(xflat, xrow + np.maximum(f, 0))
            nxt = np.where(x <= np.take(thr, pos), np.take(left, pos), np.take(right, pos)) + base
            pos = np.where(internal, nxt, pos)
        return self.lr * np.take(val, pos).sum(axis=1)

    def feature_importance(self) -> np.ndarray:
        if not self.trees:
            return np.zeros(0)
        return np.mean([t.feature_importances_ for t in self.trees], axis=0)


def mean_ndcg(scores: np.ndarray, labels: np.ndarray, groups: Sequence[int], k: int = 10) -> float:
    bounds = np.cumsum([0] + list(groups))
    vals = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        s, l = scores[a:b], labels[a:b]
        gains = (2.0 ** l) - 1.0
        ideal = np.sort(gains)[::-1][:k]
        idcg = (ideal * _dcg_discounts(len(ideal))).sum()
        if idcg <= 0:
            continue
        top = gains[np.argsort(-s, kind="stable")][:k]
        vals.append((top * _dcg_discounts(len(top))).sum() / idcg)
    return float(np.mean(vals)) if vals else 0.0


# ----------------------------------------------------------------- model wrapper

class LTRModel:
    def __init__(self, backend: str = "auto", params: Optional[dict] = None) -> None:
        if backend == "auto":
            backend = "lightgbm" if lightgbm_available() else "numpy"
        self.backend = backend
        self.params = params or {}
        self.model = None
        self.features = list(FEATURES)
        self.meta: Dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Sequence[int], X_val=None, y_val=None, groups_val=None,
            log=print) -> "LTRModel":
        if self.backend == "lightgbm":
            import lightgbm as lgb

            p = {"objective": "lambdarank", "n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63,
                 "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
                 "lambdarank_truncation_level": 20, "verbose": -1}
            p.update(self.params)
            self.model = lgb.LGBMRanker(**p)
            fit_kw = {}
            if X_val is not None:
                fit_kw = {"eval_set": [(X_val, y_val)], "eval_group": [list(groups_val)], "eval_at": [10],
                          "callbacks": [lgb.early_stopping(50, verbose=False)]}
            self.model.fit(X, y, group=list(groups), **fit_kw)
        else:
            p = {"n_trees": 200, "learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 20}
            p.update(self.params)
            self.model = NumpyLambdaMART(**p).fit(X, y, groups, X_val, y_val, groups_val, log=log)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not trained/loaded")
        return np.asarray(self.model.predict(X), dtype=np.float64)

    def importance(self) -> Dict[str, float]:
        if self.backend == "lightgbm":
            imp = self.model.booster_.feature_importance(importance_type="gain")
        else:
            imp = self.model.feature_importance()
        total = float(np.sum(imp)) or 1.0
        return {f: float(v) / total for f, v in sorted(zip(self.features, imp), key=lambda kv: -kv[1])}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"backend": self.backend, "model": self.model, "features": self.features,
                         "feature_version": FEATURE_VERSION, "meta": self.meta}, f)

    @classmethod
    def load(cls, path: str) -> "LTRModel":
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d.get("feature_version") != FEATURE_VERSION or d["features"] != FEATURES:
            raise ValueError(f"{path}: model was trained on a different feature schema; retrain it")
        m = cls(d["backend"])
        m.model, m.features, m.meta = d["model"], d["features"], d.get("meta", {})
        return m
