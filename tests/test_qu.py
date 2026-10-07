"""Query-understanding behaviour on a small inline lexicon."""
from __future__ import annotations

import time

import pytest

from dishcovery.qu import QueryUnderstanding
from dishcovery.qu.extract import extract
from dishcovery.qu.intent import IntentClassifier
from dishcovery.qu.lexicon import Lexicon

DISHES = ["chicken biryani", "veg biryani", "mutton biryani", "margherita pizza", "pepperoni pizza",
          "baingan bharta", "butter chicken", "garlic naan", "chapati", "roti", "pad thai", "masala dosa",
          "paneer tikka", "fish and chips", "egg fried rice", "chicken tikka masala", "fish tacos", "ramen"]
CUISINES = ["indian", "italian", "thai", "chinese", "mexican", "japanese"]
CHAINS = ["dominos pizza", "pizza hut", "mcdonalds", "chipotle mexican grill"]
RESTAURANTS = ["paradise biryani house", "golden dragon", "luigis trattoria"]


@pytest.fixture(scope="module")
def qu() -> QueryUnderstanding:
    freqs = {}
    for phrase in DISHES + CUISINES + CHAINS + RESTAURANTS:
        for t in phrase.split():
            freqs[t] = freqs.get(t, 0) + 200
    for w, c in {"spicy": 300, "curry": 400, "rice": 500, "eggplant": 50, "sandwich": 80, "hut": 60}.items():
        freqs[w] = freqs.get(w, 0) + c
    entries = [(d, "dish") for d in DISHES] + [(c, "cuisine") for c in CUISINES]
    entries += [(c, "chain") for c in CHAINS] + [(r, "restaurant") for r in RESTAURANTS]
    lex = Lexicon.from_entries(freqs=freqs, entries=entries)
    intent = IntentClassifier().fit(lex, n_per_class=300, seed=1)
    return QueryUnderstanding(lex, intent_model=intent)


@pytest.mark.parametrize("q,want", [("biriyani", "biryani"), ("briyani", "biryani"),
                                    ("chikken biryani", "chicken biryani")])
def test_spell_correction(qu, q, want):
    sq = qu.parse(q)
    assert sq.corrected == want
    assert any(kind == "spell" for _, _, kind in sq.corrections)


def test_segmentation(qu):
    sq = qu.parse("chickenbiryani")
    assert sq.tokens == ["chicken", "biryani"]
    assert ("chickenbiryani", "chicken biryani", "segment") in sq.corrections


def test_brand_without_space(qu):
    sq = qu.parse("pizzahut")
    assert sq.tokens == ["pizza", "hut"]
    assert sq.intent == "chain"
    assert sq.filters.chain == "pizza hut"


def test_chain_vs_dish(qu):
    sq = qu.parse("domino's")
    assert sq.intent == "chain"
    assert sq.filters.chain == "dominos pizza"
    sq = qu.parse("pizza")
    assert sq.intent == "dish"
    assert sq.filters.chain is None
    assert qu.parse("pizza hut").intent == "chain"


def test_cuisine_and_restaurant(qu):
    assert qu.parse("italian").intent == "cuisine"
    assert qu.parse("thai food").intent == "cuisine"
    assert qu.parse("paradise biryani house").intent == "restaurant"


def test_structured_extraction(qu):
    sq = qu.parse("cheap vegan near me open now")
    assert sq.intent == "dish"
    assert sq.filters.diet == ["vegan"]
    assert sq.filters.price == "low"
    assert sq.geo == "user"
    assert sq.filters.open_now is True
    assert sq.tokens == []
    assert not sq.has_text


def test_attribute_intent(qu):
    sq = qu.parse("late night food")
    assert sq.intent == "attribute"
    assert sq.filters.open_now and sq.meal_time == "late_night"


def test_price_and_rating_extraction():
    ex = extract("vegetarian pizza under 15 dollars".split())
    assert ex.filters.diet == ["veg"] and ex.filters.max_price == 15 and ex.content == ["pizza"]
    ex = extract("4 stars sushi within 3 km".split())
    assert ex.filters.min_rating == 4 and ex.radius_km == 3 and ex.content == ["sushi"]
    ex = extract("indian restaurants".split())
    assert ex.filters.doc_type == "restaurant" and ex.content == ["indian"]


def test_synonym_expansion(qu):
    sq = qu.parse("brinjal curry")
    alts = {t.text: t for t in sq.terms if t.source == "synonym"}
    assert "eggplant" in alts and "baingan" in alts
    assert all(t.weight < 1.0 for t in alts.values())
    assert alts["eggplant"].group == 0
    alts = {t.text for t in qu.parse("roti").terms if t.source == "synonym"}
    assert "chapati" in alts


def test_parse_latency(qu):
    queries = ["chicken biryani", "biriyani near me", "cheap vegan open now", "domino's", "brinjal curry",
               "chickenbiryani", "best thai food", "paneer tikka masala"]
    for q in queries:
        qu.parse(q)
    t0 = time.perf_counter()
    for i in range(1000):
        qu.parse(queries[i % len(queries)])
    per = (time.perf_counter() - t0) / 1000
    assert per < 0.003, per
