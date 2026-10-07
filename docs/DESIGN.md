# Design decisions and trade-offs

These are the choices an interviewer will probe. Each one names the
alternative and the measurement that settles it.

## 1. Document-partitioned sharding, routed by restaurant

**Choice.** Each shard holds a disjoint subset of documents and indexes every term for them. A document goes to
`fnv1a64(restaurant_id) % S` (identical in C++ and Python), so a restaurant and all its dishes co-locate.

**Alternative.** Term partitioning: each node owns whole posting lists.

**Why not term partitioning.**
- A multi-term query must ship all but the rarest posting list to a coordinator for intersection and scoring.
  Network cost grows with document frequency, which itself grows with the corpus.
- Food queries are Zipfian ("pizza", "chicken", "near me"), so the nodes owning hot terms become hot spots.
  The load imbalance (max/mean) grows with node count.
- Filter-aware retrieval needs postings, filter bitmaps and doc values in one place. Term partitioning splits
  them.

`dishcovery bench partition` models both schemes on the actual corpus and query log:
nodes contacted per query, bytes moved per query, and per-node load imbalance.

**Cost we accept.** Every query touches every shard (fan-out S), so the tail is the slowest shard. This is
why replicas, hedging and per-shard p95 tracking exist (§7).

**Possible next step.** Geo-sharding by metro would let a query touch one shard. It risks hot cities and makes
cross-metro "radius" queries awkward. Restaurant-hash sharding is the safe default; geo-sharding is a follow-up
experiment.

## 2. Immutable segments + tombstones + background merges

This is the Lucene model, implemented from scratch.

- **Segments never change.** That makes mmap safe and caches keyed by segment uid trivially correct, and lets
  readers run lock-free on a snapshot.
- **Deletes are tombstone bitmaps** held in the *snapshot*, not in the segment. They are copy-on-write, so a
  point-in-time view stays consistent.
- **Merges stream postings** (k-way merge of term dictionaries, doc-id remapping around deletes) instead of
  re-analysing documents. Deletes that arrive *during* a merge are carried over through the doc maps when the
  merged segment is installed.
- **Obsolete segments** delete their files only when the last snapshot that references them is released.

**The merge-policy trade-off.**
- Merging eagerly rewrites the same bytes many times (write amplification, I/O, CPU).
- Merging lazily leaves many segments, and every query pays a per-segment cost (term lookups, iterator setup,
  filter materialisation): read amplification that hurts p99.

`bench merge` streams the corpus with 5% updates and 2% deletes under three policies and reports write
amplification, final segment count, and query p50/p99 over time:
- **tiered** (Lucene TieredMergePolicy: tier budget, skew-scored windows)
- **log** (leveled LogByteSize)
- **none**

## 3. Near-real-time indexing

1. Every write goes to a WAL and an in-memory buffer.
2. Every `refresh_ms` (200 ms) the buffer is built into a RAM segment (same format, `MemoryStorage`) and
   published. That is what makes a new document searchable in under 1 s; `bench nrt` measures it end to end
   through the gateway on all replicas.
3. On a schedule, or under memory pressure, the RAM segments are merged into one disk segment.
4. A commit point is written atomically (tmp + fsync + rename), and the WAL is rotated.

On a crash, recovery opens the last commit and replays the WALs.

**Why not search the mutable buffer directly?** Concurrent readers would need locks inside postings iteration.
A 200 ms refresh is the same trade Elasticsearch makes with its 1 s default.

## 4. Postings format and compression

- **Blocks of 128 doc-id deltas plus term frequencies,** each compressed with the segment's codec:
  - **raw** is the speed ceiling.
  - **VByte** is byte-aligned and has a simple decode.
  - **PForDelta (OptPFD)** chooses the bit width per block that minimises size and stores outliers as exceptions.
- **Roaring bitmaps** hold filter sets (types, diets, H3 cells, tombstones): set algebra and run containers win
  there.
- **Multi-level skip list** (fan-out 8) over block last-doc ids, so `advance()` touches O(log₈ B) entries.
- **Positions in a separate file,** with per-block offsets, so BM25-only queries never touch them.

`bench codecs` re-encodes the *real* postings of a built shard. It reports bits per posting against decode
throughput, and AND-intersection time three ways:
- decode everything + linear merge
- skip-list `advance()`
- Roaring AND

## 5. Filter-aware retrieval

The naive approach scores every text match and then discards the ones that are closed, too far or the wrong
diet. In delivery search the filter is usually far more selective than the text.

1. **Materialise the filter once per segment** as a Roaring bitmap:
   - keyword sets (`OR` within a field, `AND` across fields)
   - the H3 delivery cover of the user's cell and its ancestors
   - numeric ranges and open-hours checks over doc values, applied only to the surviving set

   The result is cached per (segment, filter key). Geo keys are quantised to the H3 cell, so nearby users share
   cache entries.
2. **Its iterator joins the BM25 leapfrog.** The candidate doc is advanced to the next admissible doc *before*
   anything is scored. MaxScore keeps working on top: only clauses whose upper bound can still beat the heap
   threshold are evaluated.

`bench filter` compares three modes on one shard, reporting p50/p99, documents scored, and recall loss for
`post_filter_topk`:
- `filter_aware`
- `post_filter` (exhaustive: score everything, then filter)
- `post_filter_topk` (over-fetch 10k, then filter, which loses recall)

## 6. Delivery zones as compacted H3 covers

Zones are polygons, not radii.

- **Indexing.** Each polygon is filled at `cover_res`, then **compacted**: a complete set of 7 children becomes
  its parent. A 5 km zone costs ~10–20 keys instead of ~100.
- **Query.** "Who delivers to me?" is the union of the bitmaps for the user's cell and its ancestors: at most
  `cover_res + 1` lookups.
- **Radius queries** use a k-ring over a point index at `point_res`, plus an exact distance check.
- **Choosing the resolution.** `dishcovery build --steps geostat` tabulates, for each resolution:
  - docs per cell (p50/p99)
  - cells in a 3 km disk
  - cover size before and after compaction

  The defaults (cover 7, point 8) come from that table.

## 7. Tail latency: replicas, hedging, budgets

- **Replica choice:** power of two choices over `(outstanding + 1) × EWMA latency`.
- **Hedging:** if the primary hasn't answered by the shard's rolling p95, a second replica gets the same request
  and the first answer wins. A token bucket refilled at 5% of requests caps hedges, so hedging can't amplify an
  overload.
- **Failover:** a failed request is retried once on another replica. Three consecutive failures take a replica
  out for a cool-down period.
- **Partial results:** if every replica of a shard fails, the response is served from the remaining shards and
  marked `partial`.

`bench hedging` injects +60 ms on 10% of requests to one replica of each shard (`/admin/fault`) and compares
p99 with hedging on and off.

**Connection-pool sizing.** This was found the hard way. The shard's HTTP server (cpp-httplib) dedicates one
thread to each keep-alive connection. When the broker's pool per replica (`broker.conns_per_shard` × gateway
workers) exceeds `serving.shard_threads`, new connections queue behind *idle* keep-alive connections. p99 then
explodes into seconds while CPU sits idle. The configs keep `shard_threads ≥ gateway_workers × conns_per_shard`
plus headroom for admin traffic. An event-loop server (epoll) would remove this constraint.

## 8. Two-phase query-then-fetch with point-in-time contexts

- **Phase 1** returns (segment, doc, score) per path, plus a context id. The shard keeps that snapshot alive for
  60 s.
- **Phase 2** asks *the same replica* for features of the globally fused candidates only.

Without the context, a refresh or merge between the phases would renumber documents. Pinning to the replica
matters because writable replicas merge independently.

**Why two phases instead of features in phase 1?** Feature extraction (positions for proximity, polygons,
vectors) costs ~10× retrieval per document. Computing it for every shard's local top-k would multiply the work
by the fan-out.

## 9. Caching and invalidation

| Cache | Where | Key | Invalidated by |
|---|---|---|---|
| filter bitmaps | shard | segment uid + canonical filter (geo quantised to an H3 cell) | the segment retiring (explicit prefix invalidation) |
| decoded postings (hot, long lists) | shard | segment uid + field + term | the segment retiring |
| query results | broker | system + normalised query + location cell + time bucket + params | the shards' **generation vector** changing |

Shard generations rise on refresh, delete, flush and merge. Writes that go through the gateway update the
vector immediately, and a 1 s poller catches writes that bypass it. A cached result is served only while every
shard is still at the generation it was computed under.

## 10. Ranking design

- **Fusion before learning.** RRF needs no score calibration across BM25, cosine and geo priors. It becomes a
  feature (`rrf_score`, plus per-path ranks) and is the no-learning baseline.
- **Feature groups:**
  - textual: per-field BM25, proximity, exact/phrase, coverage
  - semantic: cosine
  - entity: rating, reviews, price, popularity
  - geo: distance, inside zone, estimated delivery time
  - temporal: open now, prep time, time-of-day match against the dish's meal affinity
  - QU agreement: intent/type, chain, cuisine, diet
  - behavioural: CTR, CVR, doc CTR
- **Labels and leakage.**
  - The model trains on weak labels or IPW-debiased simulated clicks.
  - Behavioural features come from a *disjoint* simulated history.
  - Evaluation queries are excluded from the training query log.
  - Reported NDCG uses hand labels only.
- **Backends.** LightGBM `lambdarank` is the default. A from-scratch numpy LambdaMART (vectorised LambdaRank
  gradients, Newton leaf values via weighted regression trees) is the fallback, and documents the algorithm.

## 11. Query understanding choices

- **SymSpell over a Levenshtein automaton** for correcting whole words: O(1)-ish lookups. The FST's Levenshtein
  search powers typo-tolerant *prefix* autocomplete in the suggest service.
- **Corrections need evidence.** A token is corrected only when it is rare and a candidate is ≥10× more
  frequent. Attribute words, entities and protected ingredient terms are never corrected.
- **Hard filters only from unambiguous signals.**
  - A chain alias becomes a hard brand filter only when the alias is unique: aliases that other restaurant
    names share as a prefix (for example "golden") are rejected.
  - Diet words become hard filters.
  - Cuisine intent only boosts fields.

## 12. What would change at production scale

- A binary RPC (gRPC/flatbuffers) instead of JSON between the broker and shards. Phase-2 JSON is the largest
  serialisation cost.
- Block-max WAND / block-max MaxScore. Block-level score bounds are a small format extension.
- Product quantisation for vectors at 10M+ docs.
- An offline/online split of query understanding, with a learned spell model and a learned segmenter.
- Real click logs with interleaving experiments instead of simulated PBM clicks.
