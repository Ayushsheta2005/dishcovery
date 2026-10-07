"""Query intent: dish / restaurant / cuisine / chain / attribute.

A multinomial logistic regression over character n-grams plus gazetteer
indicator features, trained on synthetic queries generated from the lexicon
(templates + injected typos), wrapped in deterministic rules for the clear
cases:

* the whole content is a chain alias              -> chain   ("domino's")
* no content left and a diet filter was extracted -> dish    ("cheap vegan near me")
* no content left otherwise                       -> attribute ("open now")
* the whole content is a cuisine / restaurant / dish entity -> that type

Intent changes retrieval: chain queries filter to the brand, restaurant
queries boost restaurant-name fields, cuisine queries boost cuisine fields.
"""
from __future__ import annotations

import pickle
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dishcovery.qu.lexicon import Lexicon
from dishcovery.qu.types import INTENTS, Entity

_GAZ_FEATURES = ("chain", "dish", "cuisine", "restaurant", "ingredient")

_TEMPLATES: Dict[str, List[str]] = {
    "dish": ["{x}", "{x} near me", "spicy {x}", "best {x}", "{x} delivery", "cheap {x}", "veg {x}",
             "{x} for lunch", "hot {x}", "order {x}", "{x} and {y}", "fresh {x}", "{x} combo", "homemade {x}"],
    "restaurant": ["{x}", "{x} restaurant", "{x} menu", "{x} near me", "order from {x}", "{x} delivery"],
    "cuisine": ["{x}", "{x} food", "{x} restaurants", "best {x} food", "{x} near me", "authentic {x}",
                "{x} cuisine", "cheap {x} food", "{x} places"],
    "chain": ["{x}", "{x} near me", "{x} delivery", "{x} menu", "order {x}", "{x} open now", "nearest {x}"],
    "attribute": ["open now", "cheap eats", "late night", "near me", "something cheap", "fast delivery",
                  "top rated near me", "open late", "food near me", "cheap food open now", "best rated",
                  "under 10 dollars", "late night food", "quick bite", "budget food", "24 hours"],
}


def _typo(rng: random.Random, s: str) -> str:
    if len(s) < 5:
        return s
    i = rng.randrange(1, len(s) - 1)
    op = rng.randrange(3)
    if op == 0:
        return s[:i] + s[i + 1:]
    if op == 1:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + s[i] + s[i:]


def gazetteer_features(tokens: Sequence[str], entities: Sequence[Entity]) -> np.ndarray:
    """[has_chain, has_dish, has_cuisine, has_restaurant, has_ingredient, coverage, empty]."""
    v = np.zeros(len(_GAZ_FEATURES) + 2, dtype=np.float32)
    covered = 0
    for e in entities:
        if e.type in _GAZ_FEATURES:
            v[_GAZ_FEATURES.index(e.type)] = 1.0
        covered += e.end - e.start
    v[-2] = covered / max(len(tokens), 1)
    v[-1] = 1.0 if not tokens else 0.0
    return v


class IntentClassifier:
    def __init__(self) -> None:
        self.vectorizer = None
        self.model = None

    # ------------------------------------------------------------ training
    @staticmethod
    def synthesize(lexicon: Lexicon, n_per_class: int = 1500, seed: int = 0) -> List[Tuple[str, str]]:
        rng = random.Random(seed)
        pools = {
            "dish": [e.surface for e in lexicon.by_type("dish")][:5000],
            "restaurant": [e.surface for e in lexicon.by_type("restaurant")][:5000],
            "cuisine": [e.surface for e in lexicon.by_type("cuisine")][:500],
            "chain": [e.surface for e in lexicon.by_type("chain")][:2000],
        }
        data: List[Tuple[str, str]] = []
        for intent in INTENTS:
            tmpl = _TEMPLATES[intent]
            pool = pools.get(intent)
            if intent != "attribute" and not pool:
                continue
            for _ in range(n_per_class):
                t = rng.choice(tmpl)
                if intent == "attribute":
                    q = t
                else:
                    x = rng.choice(pool)
                    y = rng.choice(pools["dish"]) if pools["dish"] else x
                    if rng.random() < 0.15:
                        x = _typo(rng, x)
                    q = t.format(x=x, y=y)
                data.append((q, intent))
        rng.shuffle(data)
        return data

    def fit(self, lexicon: Lexicon, n_per_class: int = 1500, seed: int = 0) -> "IntentClassifier":
        from scipy.sparse import csr_matrix, hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        from dishcovery.qu.extract import extract
        from dishcovery.qu.linker import EntityLinker

        data = self.synthesize(lexicon, n_per_class, seed)
        linker = EntityLinker(lexicon)
        texts, labels, gaz = [], [], []
        for q, y in data:
            toks = q.split()
            content = extract(toks).content
            texts.append(" ".join(content) or q)
            labels.append(y)
            gaz.append(gazetteer_features(content, linker.link(content)))
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        X = hstack([self.vectorizer.fit_transform(texts), csr_matrix(np.vstack(gaz) * 3.0)]).tocsr()
        self.model = LogisticRegression(max_iter=2000, C=4.0)
        self.model.fit(X, labels)
        return self

    # ----------------------------------------------------------- inference
    def model_scores(self, content: Sequence[str], entities: Sequence[Entity]) -> Dict[str, float]:
        if self.model is None or self.vectorizer is None:
            return {}
        from scipy.sparse import csr_matrix, hstack

        X = hstack([self.vectorizer.transform([" ".join(content)]),
                    csr_matrix(gazetteer_features(content, entities)[None, :] * 3.0)]).tocsr()
        probs = self.model.predict_proba(X)[0]
        return {c: float(p) for c, p in zip(self.model.classes_, probs)}

    def predict(self, content: Sequence[str], entities: Sequence[Entity], has_diet: bool = False
                ) -> Tuple[str, Dict[str, float]]:
        content = list(content)
        if not content:
            intent = "dish" if has_diet else "attribute"
            return intent, {intent: 1.0}
        full = [e for e in entities if e.start == 0 and e.end == len(content)]
        if full:
            e = full[0]
            if e.type in ("chain", "cuisine", "restaurant", "dish"):
                return e.type, {e.type: 1.0}
        scores = self.model_scores(content, entities)
        if scores:
            # A chain span covering most of the query wins over the model.
            for e in entities:
                if e.type == "chain" and (e.end - e.start) * 2 >= len(content):
                    scores["chain"] = max(scores.get("chain", 0.0), 0.9)
            return max(scores.items(), key=lambda kv: kv[1])[0], scores
        # No model: entity heuristics.
        for t in ("chain", "dish", "cuisine", "restaurant"):
            if any(e.type == t for e in entities):
                return t, {t: 1.0}
        return "dish", {"dish": 0.5}

    # ----------------------------------------------------------------- I/O
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "model": self.model}, f)

    @classmethod
    def load(cls, path: str) -> "IntentClassifier":
        with open(path, "rb") as f:
            d = pickle.load(f)
        c = cls()
        c.vectorizer, c.model = d["vectorizer"], d["model"]
        return c
