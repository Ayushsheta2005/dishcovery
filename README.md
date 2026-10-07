# Dishcovery

**A multi-stage retrieval and ranking system for geo-constrained food search, measured against Elasticsearch.**

This is not "I built an inverted index". It is the full stack a food-delivery search surface runs on:

- **Query understanding:** spelling, segmentation, intent, entities, synonyms and structured filters.
- **Three retrieval paths:** a BM25 inverted index, an HNSW vector index and an H3 geo index, fused with RRF.
- **LambdaMART reranking** over about 45 features, with diversification and facets.
- **Sharded serving:** scatter-gather with replicas, hedged requests, caching and near-real-time indexing.
- **Measurement:** a hand-labelled NDCG harness, an ablation for every component, and an Elasticsearch baseline.

The index core is written from scratch in C++20: analyzers, Porter2, posting codecs, skip lists, Roaring
bitmaps, segments and merges, an FST, HNSW and query execution. Query understanding, the broker,
ranking and evaluation are in Python.

```
             ┌──────────────────────────────────────────────────────┐
  query ───▶ │ Query understanding (Python)                         │
             │ normalize → SymSpell → Viterbi segmentation →        │
             │ attribute extraction → entity linking → intent (LR)  │
             │ → synonym / multilingual expansion → StructuredQuery │
             └──────────────────────────┬───────────────────────────┘
                                        ▼  phase 1: scatter (hedged, P2C replica choice)
             ┌──────────────────────────────────────────────────────┐
             │ Shard × N (C++, mmap'd immutable segments + NRT)     │
             │  filter bitmap: keyword ∩ H3 delivery cover ∩ hours  │
             │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐ │
             │  │ BM25F +     │  │ HNSW (from   │  │ geo prior    │ │
             │  │ MaxScore,   │  │ scratch),    │  │ over the     │ │
             │  │ filter-aware│  │ filtered     │  │ deliverable  │ │
             │  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘ │
             └─────────┼────────────────┼─────────────────┼─────────┘
                       └───── RRF fusion (broker) ────────┘  top-N candidates
                                        ▼  phase 2: /features (pinned to the same replica + PIT context)
             ┌──────────────────────────────────────────────────────┐
             │ LambdaMART (LightGBM, or a from-scratch numpy one)   │
             │ ~45 features: text · semantic · entity · geo ·       │
             │ temporal · behavioural (CTR/CVR from click logs)     │
             └──────────────────────────┬───────────────────────────┘
                                        ▼  top-k
             ┌──────────────────────────────────────────────────────┐
             │ MMR diversification · facets · result cache          │
             └──────────────────────────────────────────────────────┘
```

## Status

Everything below is implemented and tested:

- **C++:** 97 unit tests, also clean under ASan/UBSan.
- **Python:** 30 tests.
- **End to end:** the pipeline runs on a laptop with synthetic data (`make smoke`, about 5 minutes).

The headline numbers in [Results](#results) come from the real Yelp corpus on external compute and
from hand labels. They are left as **[N]** until that run is done. Smoke numbers are not results: they
use synthetic data and a hashing "embedding" that has no semantics.

## Why this project for a search-infra role

| Interview round | What this project exercises |
|---|---|
| DSA | Postings intersection with multi-level skip lists; MaxScore dynamic pruning; Roaring container algebra; minimal-DAWG FST with perfect hashing, RMQ top-k and Levenshtein-automaton search; HNSW construction and greedy layered search; SymSpell; Viterbi segmentation |
| Low-level design | Analyzer pipeline (pluggable per field); codec interface; segment lifecycle (build → mmap → tombstones → merge → GC); `SegmentStorage` (disk or RAM); query-execution paths; LTR model backends |
| High-level design | Document-partitioned sharding with scatter-gather; replicas with P2C/EWMA and hedged requests; two-phase query-then-fetch with point-in-time contexts; NRT refresh/flush/WAL; result cache with generation-vector invalidation; per-stage observability |
| Customer impact | NDCG@10 / MRR / Recall@100 per query class against BM25 and Elasticsearch, with bootstrap CIs and randomization tests; p50/p95/p99 per stage at 1M and 10M docs |

## Components

<details><summary><b>1. Index core (C++, <code>core/src</code>)</b></summary>

- **Analyzer pipeline** (`analysis/`). It runs NFKC, case folding, accent stripping (utf8proc), apostrophe joining
  ("domino's" becomes "dominos"), Unicode tokenization, stopwords, **Porter2** and edge n-grams. Porter2 is
  written from scratch and verified against Snowball on 3,000 golden words. Analyzers are configured per field
  in `configs/schema.json`:
  - dish names use a plural-only stemmer
  - descriptions and reviews use Porter2
  - an un-stemmed `name_exact` field supports exact and phrase matching
- **Postings** (`index/postings.*`). Each posting is (docID, tf, positions), stored in delta-encoded blocks of 128.
  Blocks are compressed with **raw, VByte or PForDelta**. The PForDelta variant is OptPFD: it picks the bit
  width per block, with exceptions.
- **Skip lists.** A **multi-level skip list** with fan-out 8 sits over the blocks, so `advance()` touches
  O(log₈ B) entries before decoding one block. Positions live in their own file, and per-block offsets let
  phrase and proximity checks skip straight to them.
- **Roaring bitmaps** (`codec/roaring.*`). These are written from scratch, with array, bitmap and run containers,
  galloping intersection and an `advance()` iterator. They hold filter fields, H3 cells and tombstones.
- **Segments** (`index/segment*`, `merger.cc`, `shard_index.*`). Segments are immutable and every file is
  **mmap'd**. A segment holds:
  - a front-coded term dictionary
  - column files
  - stored summaries
  - an id map
  - vectors and an HNSW graph, whose level 0 is used in place from the mmap
- **Deletes, merges and durability.**
  - Deletes are copy-on-write tombstone bitmaps per snapshot.
  - **Tiered** (Lucene-style) and **leveled** merge policies are both available.
  - Merges stream postings directly (k-way term merge, doc-id remapping) instead of re-analysing documents.
  - A WAL and commit points provide crash safety.
- **Near-real-time indexing.** Writes are buffered and become a RAM segment every `refresh_ms` (200 ms, so a
  new document is searchable in under 1 s). RAM segments are flushed to disk on a schedule.

</details>

<details><summary><b>2. Query understanding (Python, <code>dishcovery/qu</code>)</b></summary>

- **SymSpell** with Damerau (OSA) verification and a frequency-ratio guard: biriyani, briyani and biriani all
  become biryani.
- **Unigram Viterbi segmentation:** "chickenbiryani" becomes "chicken biryani". Brands typed without spaces are
  handled too: "pizzahut" becomes the Pizza Hut chain.
- **Attribute extraction:** "cheap vegan near me open now" becomes
  `{diet: vegan, price: low, geo: user, open_now: true}` with intent `dish`.
- **Entity linking:** longest match against a gazetteer of dishes, cuisines, chains, restaurants and ingredients
  built from the corpus and the taxonomy. Ambiguous brand aliases are rejected.
- **Intent classifier:** dish, restaurant, cuisine, chain or attribute. It is a logistic regression over
  character n-grams plus gazetteer features, wrapped in deterministic rules. "domino's" gets a hard chain
  filter; "pizza" does not.
- **Synonyms:** more than 90 groups, covering Hindi/English (brinjal/eggplant/baingan), British/American and
  transliteration variants. Expansions are added at a lower weight.

</details>

<details><summary><b>3. Vector retrieval</b></summary>

- **Embeddings:** all-MiniLM-L6-v2 (384-d) for dish and restaurant text. A dependency-free hashing encoder
  exists for CI only.
- **HNSW** (`vector/hnsw.*`), written from scratch:
  - heuristic neighbour selection (Malkov & Yashunin, Alg. 4)
  - parallel insertion with per-node locks
  - generation-stamped visited sets
  - zero-copy level-0 adjacency read from the mmap
  - **filtered search**
- **Brute-force fallback** when the filter admits few documents, and for small RAM segments.
- **Fusion with BM25 via Reciprocal Rank Fusion.** The ablation reports *which query classes* hybrid retrieval
  helps on; expect mostly vague and semantic queries.

</details>

<details><summary><b>4. Geospatial</b></summary>

- **Delivery zones** are real polygons. Each is indexed as a **compacted H3 cover** at `cover_res` (7). The
  "who delivers to me" lookup checks the user's cell and its ancestors: at most 8 bitmap lookups.
- **Radius queries** use a k-ring over the point index plus an exact haversine check.
- **Resolution** is chosen by measurement: `dishcovery build --steps geostat` reports the docs-per-cell
  distribution, cells per 3 km disk, and cover sizes before and after compaction for each resolution.
- **Filter-aware retrieval:** the geo, keyword, range and open-hours filter is materialised once per segment as a
  bitmap (and cached). Its iterator then takes part in the BM25 leapfrog, so only admissible docs are ever
  scored. `post_filter` (score everything, then drop) and `post_filter_topk` (over-fetch k×10) are the
  baselines; the ablation reports latency, docs scored and recall loss.

</details>

<details><summary><b>5. Ranking</b></summary>

- **Features** (`ranking/features.py`), about 45 in total:
  - textual: per-field BM25, proximity, exact and phrase match, coverage
  - semantic and fusion: cosine similarity, RRF score, per-path ranks
  - entity: rating, review count, price, popularity, doc type
  - geo: distance, inside-zone, estimated delivery time
  - temporal: open now, prep time, time-of-day match against the dish's meal affinity
  - QU agreement: intent type, chain, cuisine and diet match
  - behavioural: historical CTR, CVR and doc CTR
- **LambdaMART:** LightGBM `lambdarank`, or a from-scratch numpy LambdaMART (LambdaRank gradients, Newton leaves)
  when LightGBM isn't available.
- **Training labels:**
  - weak labels (review text, categories and constraint satisfaction), or
  - inverse-propensity-weighted clicks from a position-based click simulator.

  Behavioural features come from a *separate* simulated history window, so labels can't leak into features.
- **Diversification:** MMR with a per-restaurant cap, or a submodular coverage objective.

</details>

<details><summary><b>6. Serving</b></summary>

- **Sharding:** document-partitioned, routed by restaurant so a restaurant and its dishes co-locate.
  `bench partition` quantifies the choice against term partitioning on this corpus and query log.
- **Replicas:**
  - power-of-two-choices over (outstanding requests × EWMA latency)
  - **hedged requests** after the shard's rolling p95, with a 5% budget
  - failover and cool-down for failing replicas
- **Two-phase query-then-fetch:** phase 2 reuses the phase-1 replica's point-in-time snapshot, so (segment, doc)
  addresses stay valid across refreshes and merges.
- **Caches:**
  - shard filter-bitmap cache and decoded-postings cache, invalidated explicitly when segments retire
  - broker result cache, invalidated by the shards' generation vector
- **Observability:** every response carries per-stage timings (QU, embed, phase 1, phase 2, rank, plus each
  shard's filter/BM25/vector/geo). The gateway exports Prometheus `/metrics`.

</details>

## Quick start (laptop)

```bash
make venv build          # Python 3.9+, CMake 3.20+, a C++20 compiler; fetches H3/utf8proc/json/httplib
make test                # 97 C++ + 30 Python unit tests
make smoke               # synthetic data → index → cluster → train → eval → every benchmark (~5 min)
```

To try queries interactively (the gateway also serves a small search page at http://127.0.0.1:8090/):

```bash
.venv/bin/python -m dishcovery.cli --config configs/smoke.yaml cluster up
.venv/bin/python -m dishcovery.cli --config configs/smoke.yaml search "chickenbiryani" --time "fri 19:30"
.venv/bin/python -m dishcovery.cli --config configs/smoke.yaml qu "cheap vegan near me open now"
```

## Full experiment (external compute)

See **[docs/RUNBOOK.md](docs/RUNBOOK.md)** for hardware sizing, timings and every command. In short:

```bash
bash scripts/setup.sh                                   # toolchain, venv (LightGBM, sentence-transformers), build, tests
# download the Yelp Open Dataset JSON into data/raw/yelp/
docker compose -f docker/docker-compose.yml up -d       # Elasticsearch baseline
bash scripts/run_full.sh configs/yelp.yaml              # ~2M docs: quality + latency + ablations
dishcovery --config configs/yelp.yaml labels hand       # the one manual step: grade the pooled results 0–3
dishcovery --config configs/yelp.yaml eval --labels hand
bash scripts/run_full.sh configs/1m.yaml latency        # latency at 1M docs
bash scripts/run_full.sh configs/10m.yaml latency       # latency at 10M docs
```

## Results

Every number is reproducible with the commands above. `dishcovery report` writes
`work/<config>/results/RESULTS.md` and plots.

**Quality** (hand labels: 50 queries × ~25 pooled docs, grades 0–3; 95% bootstrap CI; † = significant vs BM25):

| System | NDCG@10 | MRR | Recall@100 |
|---|---|---|---|
| exact match | [N] | [N] | [N] |
| BM25 (our engine) | [N] | [N] | [N] |
| Elasticsearch out-of-the-box | [N] | [N] | [N] |
| BM25 + query understanding | [N] | [N] | [N] |
| hybrid (BM25 + HNSW + geo, RRF) | [N] | [N] | [N] |
| hybrid + LambdaMART | [N] | [N] | [N] |
| + MMR diversification | [N] | [N] | [N] |

**Latency** (p50 / p95 / p99 end to end, open-loop load, result cache off):

| Corpus | QPS | p50 | p95 | p99 | slowest stage |
|---|---|---|---|---|---|
| 1M docs | [N] | [N] ms | [N] ms | [N] ms | [N] |
| 10M docs | [N] | [N] ms | [N] ms | [N] ms | [N] |

**Ablations:**

- hybrid vs BM25 by query class
- LambdaMART vs RRF order
- post-filter vs filter-aware (latency, docs scored)
- raw / VByte / PFor / Roaring (bits per posting vs decode speed) and skip lists vs linear AND
- tiered vs leveled vs no merging (write amplification vs p99)
- HNSW ef sweep (recall@100 vs latency)
- hedging on/off with a slow replica
- cache on/off
- NRT visibility (new doc searchable in [N] ms)

All of these are in `RESULTS.md`.

## Evaluation methodology (short)

See [docs/EVALUATION.md](docs/EVALUATION.md).

- **Hand labels only for reported NDCG.** 50 queries across 10 classes, each issued at a location and a local
  time; graded 0–3 by [data/eval/LABELING_GUIDELINES.md](data/eval/LABELING_GUIDELINES.md); pooled from
  depth-10 runs of several systems. `judged@10` is reported to expose pool bias.
- **Weak supervision and simulated clicks are training data.** Any report computed on them is stamped
  *diagnostic only*.
- **Every system faces the same delivery-zone constraint,** Elasticsearch included (a `geo_shape` filter on the
  same polygons), so the comparison is about ranking.

## Repository layout

```
core/src/analysis   normalizer, tokenizer, stopwords, Porter2, analyzers
core/src/codec      VByte, bit-packing, PForDelta, Roaring
core/src/index      schema, term dictionary, postings + skip lists, segments, builder, merger,
                    merge policies, NRT shard index (WAL, refresh, flush, tombstones)
core/src/search     request model, filter materialisation + cache, MaxScore BM25, exact, vector, geo, features
core/src/vector     HNSW, distance kernels
core/src/geo        H3 covers, k-rings, point-in-polygon
core/src/fst        minimal DAWG FST: ordinals, top-k prefix completion, Levenshtein automaton
core/src/server     shard HTTP server (PIT contexts, fault injection)
core/src/tools      dishcovery-shard, -indexer, -suggest, -geostat
core/bench          codecs, merge, HNSW, FST micro-benchmarks
core/tests          97 unit tests (dependency-free harness)
dishcovery/data     Yelp/OSM loaders, synthetic Yelp generator, taxonomy (463 dishes), menus, polygons
dishcovery/qu       SymSpell, segmentation, extraction, linking, intent, synonyms, pipeline
dishcovery/embed    sentence-transformer / hashing encoders
dishcovery/serving  broker, shard client (hedging), cache, gateway, cluster launcher
dishcovery/ranking  RRF, features, LambdaMART, behaviour store, diversification, training
dishcovery/labels   eval queries, query log, weak labels, PBM click simulation, pooling, labelling tool
dishcovery/eval     metrics, significance tests, reports, system runner
dishcovery/baselines Elasticsearch out-of-the-box and +QU
dishcovery/bench    open-loop load generator, partitioning model, benchmark runners, plots
configs/            schema + smoke / yelp / 1m / 10m experiment configs
docs/               architecture, design decisions, runbook, evaluation
```

## Sixteen-week plan → where it lives

| Weeks | Deliverable | Code |
|---|---|---|
| 1–2 | Analyzer, tokenizer, inverted index, boolean queries | `analysis/`, `index/segment_builder.cc`, `search/` exact path |
| 3–4 | Compression, skip lists, positions, BM25, persistence | `codec/`, `index/postings.*`, `search/searcher.cc` |
| 5–6 | Segments, merge policy, deletes, incremental indexing | `index/shard_index.*`, `merger.cc`, `merge_policy.*` |
| 7–8 | Spell correction, autocomplete FST, segmentation | `dishcovery/qu/`, `fst/`, `dishcovery-suggest` |
| 9–10 | HNSW, embeddings, RRF | `vector/`, `dishcovery/embed`, `ranking/fusion.py` |
| 11 | H3 geo index, filter-aware retrieval | `geo/`, `search/filter.*` |
| 12–13 | Features, labels, LambdaMART, NDCG harness | `ranking/`, `labels/`, `eval/` |
| 14 | Sharding, scatter-gather, replication, caching | `serving/`, `server/` |
| 15 | Load testing, profiling, tail latency | `bench/loadgen.py`, hedging, per-stage timings |
| 16 | Ablations, README, write-up | `experiments/`, `docs/` |

## License

MIT. The Yelp Open Dataset is not redistributed here; download it yourself under Yelp's terms.
