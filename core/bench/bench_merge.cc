// bench_merge: merge-policy trade-off under a streaming write workload.
//
//   bench_merge --docs docs.jsonl --policy '{"type":"tiered"}' [--limit 200000]
//               [--batch 2000] [--flush-every 5] [--update-rate 0.05] [--delete-rate 0.02]
//               [--queries queries.txt] [--work /tmp/bm]
//
// Streams documents into a fresh ShardIndex: each batch is added (with some
// updates and deletes of earlier docs), refreshed, every --flush-every batches
// flushed, then the merge policy runs. After every flush a fixed query sample
// is executed (BM25 path) to measure read cost vs. segment count. Reports a
// JSON timeline plus write amplification (bytes written by flush+merge / bytes
// flushed) and query latency percentiles.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include "index/shard_index.h"
#include "search/searcher.h"
#include "util/args.h"
#include "util/histogram.h"
#include "util/timer.h"

using namespace dc;

int main(int argc, char** argv) {
  Args a(argc, argv);
  if (!a.has("docs")) {
    std::cerr << "usage: bench_merge --docs FILE [--policy JSON] [--limit N] [--batch 2000] [--flush-every 5]\n"
                 "                   [--update-rate 0.05] [--delete-rate 0.02] [--queries FILE] [--work DIR]\n";
    return 2;
  }
  index::Schema schema = a.has("schema") ? index::Schema::load(a.str("schema")) : index::Schema::default_schema();
  schema.vector_dim = 0;  // merges of text/filters/columns; HNSW rebuild cost is measured by the indexer
  std::string work = a.str("work", (std::filesystem::temp_directory_path() / "dc_bench_merge").string());
  std::filesystem::remove_all(work);
  index::IndexOptions io;
  io.background = false;
  io.wal = false;
  io.merge_policy = json::parse(a.str("policy", "{\"type\":\"tiered\"}"));
  index::ShardIndex idx(work, schema, io);
  idx.open();
  search::Searcher searcher(idx.schema(), {});

  std::vector<std::string> queries;
  if (a.has("queries")) {
    std::ifstream qf(a.str("queries"));
    for (std::string l; std::getline(qf, l);)
      if (!l.empty()) queries.push_back(l);
  }
  if (queries.empty()) queries = {"chicken biryani", "pizza", "pad thai", "vegan bowl", "burger", "tacos",
                                  "ramen", "paneer tikka", "fish and chips", "spicy noodles"};

  const size_t limit = static_cast<size_t>(a.integer("limit", 200000));
  const size_t batch = static_cast<size_t>(a.integer("batch", 2000));
  const int flush_every = static_cast<int>(a.integer("flush-every", 5));
  const double upd = a.real("update-rate", 0.05), del = a.real("delete-rate", 0.02);
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<double> U(0, 1);

  std::ifstream in(a.str("docs"));
  std::vector<std::string> ids;
  std::vector<json> recent;
  Histogram qlat;
  size_t added = 0;
  int batches = 0;
  Stopwatch total;
  std::string line;
  while (added < limit && std::getline(in, line)) {
    json j = json::parse(line);
    idx.add(index::parse_document(j, schema), "");
    ids.push_back(j["id"].get<std::string>());
    if (recent.size() < 5000) recent.push_back(j);
    else recent[rng() % recent.size()] = j;
    ++added;
    if (added % batch) continue;
    // Updates re-add a recent doc (delete + add); deletes remove a random earlier id.
    for (size_t k = 0; k < static_cast<size_t>(batch * upd); ++k) {
      const json& r = recent[rng() % recent.size()];
      idx.add(index::parse_document(r, schema), "");
    }
    for (size_t k = 0; k < static_cast<size_t>(batch * del); ++k) idx.remove(ids[rng() % ids.size()]);
    idx.refresh();
    if (++batches % flush_every == 0) {
      idx.flush();
      idx.maybe_merge();
      auto snap = idx.snapshot();
      Histogram h;
      for (const auto& q : queries) {
        search::SearchRequest req;
        int g = 0;
        for (auto& t : analysis::get_analyzer("plain").terms(q)) req.terms.push_back({t, 1.0f, g++, true});
        for (auto& f : schema.text_fields) req.field_weights.push_back(f.weight);
        req.bm25_k = 100;
        Stopwatch sw;
        searcher.search(*snap, req);
        h.record(sw.elapsed_us());
        qlat.record(sw.elapsed_us());
      }
      json st = idx.stats();
      std::cout << json{{"bench", "merge_timeline"},
                        {"docs", added},
                        {"segments", st["num_segments"]},
                        {"disk_bytes", st["disk_bytes"]},
                        {"bytes_flushed", st["bytes_flushed"]},
                        {"bytes_merge_written", st["bytes_merge_written"]},
                        {"write_amplification", st["write_amplification"]},
                        {"query_p50_us", h.percentile(0.5)},
                        {"query_p99_us", h.percentile(0.99)}}
                       .dump()
                << std::endl;
    }
  }
  idx.flush();
  json st = idx.stats();
  std::cout << json{{"bench", "merge_summary"},
                    {"policy", io.merge_policy},
                    {"docs", added},
                    {"seconds", total.elapsed_s()},
                    {"segments", st["num_segments"]},
                    {"merges", st["merges"]},
                    {"bytes_flushed", st["bytes_flushed"]},
                    {"bytes_merge_written", st["bytes_merge_written"]},
                    {"write_amplification", st["write_amplification"]},
                    {"query_p50_us", qlat.percentile(0.5)},
                    {"query_p95_us", qlat.percentile(0.95)},
                    {"query_p99_us", qlat.percentile(0.99)},
                    {"merge_us", st["merge_us"]}}
                   .dump()
            << std::endl;
  return 0;
}
