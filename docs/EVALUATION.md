# Evaluation methodology

## Test set: 50 hand-labelled queries

`data/eval/queries.jsonl` has 5 queries in each of 10 classes:

- exact dish
- misspelled
- compound (missing spaces)
- chain
- restaurant name
- cuisine
- attribute ("late night food")
- synonym/multilingual ("baingan bharta", "brinjal curry")
- vague/semantic ("comfort food for a rainy day")
- dietary/structured ("cheap vegan near me open now")

Every query is issued **from a location inside an indexed metro, at a local time**. In food delivery a closed
or non-delivering restaurant is not a correct answer, however good the text match.

**Judgements** are graded 0–3 per `data/eval/LABELING_GUIDELINES.md`. They are collected by TREC-style
**pooling**: the union of the top-10 results of several systems (exact, BM25, hybrid, LTR,
Elasticsearch), judging only documents not already judged.

## Metrics

| Metric | Definition |
|---|---|
| NDCG@10 | gain 2^g − 1, discount log2(i + 2); ideal ranking over *all* judged documents of the query; unjudged = 0 |
| MRR | reciprocal rank of the first document with grade ≥ 2 |
| Recall@100 | fraction of judged-relevant (grade ≥ 2) documents retrieved in the top 100 |
| judged@10 | fraction of the top 10 that is judged; a pool-bias diagnostic (a new system with low judged@10 is under-credited) |

Each system gets mean ± 95% percentile-bootstrap CI over queries. Differences from the baseline are tested with
a two-sided **paired randomization (sign-flip) test**, Holm–Bonferroni corrected across systems. With 5 queries
per class, per-class tables are indicative only, and the report says so.

## Systems

| System | Description |
|---|---|
| `exact` | AND of all raw query tokens on the un-stemmed name field, ranked by popularity |
| `bm25` | our engine, raw query, weighted multi-field BM25 |
| `es_oob` | Elasticsearch defaults: `standard` analyzer, `multi_match` best_fields over the same fields |
| `bm25_qu` | BM25 with query understanding: corrections, expansions, intent field boosts, hard filters |
| `es_qu` | Elasticsearch queried with the same QU output (isolates QU from engine) |
| `vector`, `bm25_vector` | dense retrieval alone, and BM25 + dense via RRF |
| `hybrid` | BM25 + HNSW + geo prior via RRF |
| `hybrid_nospell`, `hybrid_noexpand` | QU stage ablations |
| `ltr` | hybrid candidates reranked by LambdaMART |
| `full` | `ltr` + MMR diversification (expect a small NDCG cost for diversity; the report includes it) |

All systems apply the same delivery-zone constraint, Elasticsearch via `geo_shape` on the same polygons.

## Training data (never used for reported numbers)

- **Query log:** a synthetic training log of ~4,000 queries with Zipf popularity, 10% typos and 3% compounds.
  It never contains the text of an evaluation query.
- **Weak labels** (`labels/weak.py`): grades come from canonical-dish / name / cuisine / chain matches, review
  mentions and constraint satisfaction. A diet violation caps the grade at 0. Outside the delivery zone is 0.
  Closed at the query time caps the grade at 1.
- **Click simulation** (position-based model):
  - P(examine rank r) = (1/r)^η
  - P(click | examined) = attract[grade]
  - Orders depend on the grade, given a click.

  Training labels are the **inverse-propensity-weighted** click rate, bucketed into grades. Behavioural features
  (CTR, CVR) come from a *separate* simulated history window.

Any report computed on weak labels is stamped "DIAGNOSTIC ONLY" in its provenance line.
