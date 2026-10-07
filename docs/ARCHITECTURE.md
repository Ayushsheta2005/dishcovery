# Architecture

## Processes

```
                         ┌──────────────────────┐
     HTTP clients ─────▶ │ gateway (aiohttp)    │  N worker processes (SO_REUSEPORT)
                         │  broker:             │
                         │   QU · embed · RRF   │──────────────▶ suggest service (C++, FST)
                         │   LTR · MMR · cache  │                 /suggest /fuzzy
                         └─────────┬────────────┘
                 phase 1 /search   │   phase 2 /features (pinned)
          ┌────────────────────────┼─────────────────────────┐
          ▼                        ▼                         ▼
   shard 0: r0  r1          shard 1: r0  r1   …       shard S-1: r0  r1      (C++ dishcovery-shard)
   each replica: ShardIndex (segments, WAL, NRT) + Searcher (filter cache, postings cache)
```

## Life of a query

Example: "cheap vegan near me open now" sent from Philadelphia at Saturday 13:00.

1. **Gateway / broker.**
   - Normalize: unchanged.
   - Spell-check and segmentation: nothing to correct.
   - Extraction: `diet=[vegan]`, `price_tier ≤ 2`, `open_slot=(sat,13:00)`, geo = user, no content tokens.
   - Intent rule: diet present and no content, so intent = `dish`.
   - Paths: BM25 is skipped (no text). Vector runs on the corrected text, and the geo path runs.
2. **Phase 1, scatter.** For each shard, P2C picks a replica and the request is hedged after that shard's p95.
3. **Shard.** For each segment, the filter is materialised:
   1. the `diet:vegan` bitmap
   2. ∩ the delivery cover of the user's H3 cell and its ancestors
   3. then doc-value checks on the survivors (`price_tier ≤ 2`, open at slot 278)

   The result is cached. The vector path runs filtered HNSW, or a brute-force scan if few documents are
   admissible. The geo path scores admissible documents by popularity × distance decay. Each path returns its
   top-k, along with a point-in-time context.
4. **Broker fusion.** Each path is merged across shards by score, then RRF combines the paths. The broker keeps
   the top `rerank_depth` candidates.
5. **Phase 2.** Each shard receives `/features` for its candidates, pinned to the replica that answered phase 1.
   It returns ~33 signals per document plus stored summaries.
6. **Ranking.** The broker adds QU-agreement, temporal and behavioural features, scores with LambdaMART (in a
   thread), applies MMR with a per-restaurant cap, computes facets, returns the top 20, and caches the result
   under the shard generation vector.

The response carries `timings_ms` for every stage, plus the maximum per-shard time for each shard stage.

## Life of a write

1. `POST /index` arrives at the gateway with a document.
2. The broker embeds the document (if no vector is given), routes it to `fnv1a64(restaurant_id) % S`, and
   broadcasts it to every replica of that shard.
3. Each replica:
   1. appends the document to its WAL and buffer
   2. tombstones any older version in existing segments
   3. within ≤ `refresh_ms`, builds a RAM segment and publishes a new snapshot (the generation rises)
   4. on a schedule, merges RAM segments into a disk segment, writes the commit point and rotates the WAL
   5. lets the background merge policy compact disk segments
4. The broker's result cache stops serving entries computed under the old generation.

## On-disk layout of one shard

```
shard_0/
  COMMIT                      generation, live segment list, tombstone files (atomic rename)
  wal.log  wal.<gen>.*.log    write-ahead log (rotated at flush)
  del_<uid>_<gen>.rbm         tombstones (Roaring) per segment
  seg_00000012/
    meta.json                 doc count, codec, field length sums, flags
    terms.dict                front-coded term dictionary (blocks of 32 + in-memory block index)
    postings.bin              doc/freq blocks + multi-level skip lists
    positions.bin             vbyte position deltas
    filters.dict filters.bin  keyword + H3 keys → Roaring bitmaps
    norms_<f>.bin             u16 field lengths
    numeric_<f>.bin           f32 doc values
    latlon.bin hours.bin meal.bin type.bin
    stored.bin ids.bin polygons.bin   var-length columns (offsets at the end)
    idmap.bin                 sorted (hash(id), doc) for delete-by-id
    vectors.bin hnsw.bin      unit vectors; HNSW (level 0 mmap'd in place)
```
