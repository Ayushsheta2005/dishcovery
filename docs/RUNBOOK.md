# Runbook: full experiments on external compute

## Hardware

| Run | Docs | CPU | RAM | Disk | GPU |
|---|---|---|---|---|---|
| `configs/yelp.yaml` | ~2M (Yelp + menus) | 16+ cores | 32 GB | 50 GB SSD | recommended (embeddings) |
| `configs/1m.yaml` | 1M | 16 cores | 32 GB | 30 GB | optional |
| `configs/10m.yaml` | 10M | 32 cores | 64–128 GB | 200 GB SSD | strongly recommended |

Index size is dominated by the 384-d float vectors plus HNSW: about 1.7 KB per doc, so roughly 17 GB at 10M.
Leave room for Elasticsearch (8 GB heap) if it shares the box.

## 0. Setup (once)

```bash
git clone https://github.com/Ayushsheta2005/dishcovery && cd dishcovery
bash scripts/setup.sh            # apt toolchain, venv with LightGBM + sentence-transformers, build, all tests
```

## 1. Data

1. Download the **Yelp Open Dataset** (https://www.yelp.com/dataset). Accept the terms and extract
   `yelp_academic_dataset_business.json` and `yelp_academic_dataset_review.json` into `data/raw/yelp/`.
2. Optional: densify with OpenStreetMap POIs. Then set `data.osm_path: data/raw/osm/pois.jsonl` in
   `configs/yelp.yaml`.

   ```bash
   .venv/bin/python -m dishcovery.cli --config configs/yelp.yaml data osm
   ```

## 2. Elasticsearch baseline

```bash
docker compose -f docker/docker-compose.yml up -d
```

## 3. Build, train, evaluate, benchmark

```bash
bash scripts/run_full.sh configs/yelp.yaml
```

This runs each step below (each is also a CLI command and is idempotent):

| Step | Command | Typical time (16 cores + 1 GPU, ~2M docs) |
|---|---|---|
| Corpus: Yelp businesses + review aggregation + synthetic menus | `build --steps prepare` | 10–20 min (review file read in parallel byte ranges) |
| Embeddings (MiniLM, batch 1024) | `build --steps embed` | 10–15 min on GPU (hours on CPU) |
| Index per shard (parallel segment builds + HNSW + force-merge) | `build --steps index` | 10–30 min |
| QU artifacts (lexicon, intent model), suggest FST | `build --steps qu,suggest` | 2–5 min |
| H3 resolution study | `build --steps geostat` | 1–3 min |
| Cluster: S shards × R replicas + suggest + gateway | `cluster up` | seconds |
| LambdaMART: query log → candidates → weak labels → clicks → train | `train` | 10–30 min |
| ES index | `es index` | 10–20 min |
| Quality evaluation (all systems + ES) | `eval --labels weak` (until hand labels exist) | minutes |
| Ablations: filter, codecs, merge, HNSW, partition, NRT, load, cache, hedging | `ablations` | 30–60 min |

## 4. Hand labels (the one manual step)

```bash
DC=".venv/bin/python -m dishcovery.cli --config configs/yelp.yaml"
$DC eval --labels weak                     # produces runs for every system
$DC labels pool --depth 10                 # TREC-style pool: union of top-10s, unjudged only
$DC labels hand                            # interactive; resumable; writes data/eval/hand_labels.jsonl
#   or: $DC labels export-csv → grade in a spreadsheet → $DC labels import-csv --csv graded.csv
$DC eval --labels hand                     # the reported numbers
$DC report
```

Grading follows `data/eval/LABELING_GUIDELINES.md`. 50 queries × 6 documents (about 300 pairs) is the minimum;
depth-10 pools give about 25 documents per query and much tighter confidence intervals. Commit
`data/eval/hand_labels.jsonl`: it is the test set.

## 5. Scale points (latency)

```bash
bash scripts/run_full.sh configs/1m.yaml latency
bash scripts/run_full.sh configs/10m.yaml latency
```

Both reuse `work/yelp/corpus/docs.base.jsonl`, subsampled to 1M or cloned to 10M in shifted synthetic metros,
and re-embed and re-index it. For latency-only runs at 10M without a GPU, set `embedding.model: hashing` in
`configs/10m.yaml`. Latency is unaffected by embedding quality; recall/NDCG must not be reported from that run.

## 6. Reading the results

`work/<name>/results/RESULTS.md` contains every table, and the same directory holds the `*.png` plots.
Copy the headline numbers into the README's [N] placeholders.

## Operations

- **Logs:** `work/<name>/logs/` (one per shard replica, suggest, gateway).
- **Stats:**
  - shard: `curl :7101/stats` (segments, merges, write amplification, cache hit rates, per-endpoint latency)
  - gateway: `curl :8080/stats`, `curl :8080/metrics`
- **Fault injection:** `curl -XPOST :7101/admin/fault -d '{"delay_ms":50,"delay_prob":0.1}'` (send `{}` to clear).
- **Manual maintenance:** `POST /admin/refresh`, `/admin/flush`, `/admin/merge?max_segments=1`.
- **Shutdown:** `dishcovery cluster down`, or SIGTERM. Shards flush RAM segments on shutdown; the WAL covers
  crashes.
- **Replica disk use:** writable replicas (the default, needed for NRT indexing) each get their own copy of the
  shard index (`shard_<s>_r<r>`), created on the first `cluster up`. For read-only latency runs at 10M, set
  `serving.read_only: true`: replicas then share one copy and the WAL and merges are off.
- **Ports:** shard *s* replica *r* listens on `shard_base_port + 10·s + r`, suggest on `suggest_port`, gateway on
  `gateway_port`.
