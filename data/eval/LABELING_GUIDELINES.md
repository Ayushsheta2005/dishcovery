# Relevance labelling guidelines (graded 0–3)

You are judging whether a result is a good answer **for this user, at this
place and time**. Each query comes with a location (inside one of the indexed
metros) and a local time (for example "tue 13:00"). The labelling tool shows
distance, whether the place is open at that time, and whether the user is
inside its delivery zone.

| Grade | Meaning |
|---|---|
| **3 — exact** | Exactly what was asked for, and it can be ordered now. It is open and delivers to the user, or is within the stated radius. |
| **2 — good substitute** | Clearly relevant, and most users would be happy. Examples: a close variant of the dish, the right cuisine, or the right dish from a slightly worse option. |
| **1 — marginal** | Related but probably not what the user wanted. Examples: the right cuisine but the wrong dish type, or relevant but closed or outside the delivery zone. |
| **0 — irrelevant** | Off-topic, violates an explicit constraint (diet, price, chain), or can't be served at all. |

General rules:

* **Constraints in the query are hard:**
  * A non-vegetarian dish for "vegan …" is **0**.
  * "under $15" with a $22 dish is **at most 1**.
  * "open now" with a closed place is **at most 1**.
* **Closed at the query time, or outside the delivery zone,** caps the grade at **1**, even for a perfect text match.
* **Judge dishes and restaurants on their own terms.**
  * For a dish query, a restaurant document that serves that dish prominently is usually **2**. The dish document itself can be **3**.
  * For a restaurant or chain query, the restaurant document is **3** and its dishes are **2**.
* **Spelling does not matter.** Judge what the user meant: "biriyani" means biryani.
* **Don't reward popularity by itself.** A famous place that doesn't match the intent is still **0** or **1**.
* **Skip (s)** if you genuinely can't tell. Unjudged documents count as non-relevant, and we report `judged@10` so pool coverage is visible.

## Worked examples

| Query | Result | Grade | Why |
|---|---|---|---|
| chicken biryani (Fri 19:30) | "Hyderabadi Chicken Dum Biryani", open, delivers | 3 | Exact dish |
| chicken biryani | "Veg Biryani" from the same place | 2 | Close variant |
| chicken biryani | "Butter Chicken" | 1 | Same cuisine, different dish |
| chicken biryani | Exact match but closed at 19:30 | 1 | Can't be ordered now |
| domino's | Domino's Pizza restaurant, delivers | 3 | The chain asked for |
| domino's | Pepperoni pizza from an independent pizzeria | 1 | Generic substitute for a brand query |
| baingan bharta | "Roasted Eggplant Curry (Baingan Bharta)" | 3 | Same dish under a synonym |
| baingan bharta | "Eggplant Parmesan" | 1 | Shares the ingredient, but it's a different dish |
| cheap vegan near me open now | $9 vegan bowl, open, delivers | 3 | Meets every constraint |
| cheap vegan near me open now | $11 chicken wrap | 0 | Violates the vegan constraint |
| something spicy and warm | Tom yum soup | 3 | Matches the vague intent well |
| something spicy and warm | Cold-brew coffee | 0 | Neither spicy nor warm |

## Process

1. Build pools: `dishcovery labels pool --systems exact,bm25,hybrid,ltr,es_oob --depth 10`.
2. Label: `dishcovery labels hand --pool data/eval/pool.jsonl`. It saves after every judgement and resumes where you stopped.
3. Spreadsheet alternative: `dishcovery labels export-csv`, then `dishcovery labels import-csv`.

About 300 judged pairs (50 queries × 6) is the floor. Depth-10 pools over
4–5 systems give roughly 25 documents per query and a much more reliable
NDCG@10. Report NDCG **only** on these hand labels. Weak and click labels are
training data.
