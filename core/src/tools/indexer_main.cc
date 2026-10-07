// dishcovery-indexer: build one shard's index from docs.jsonl (+ embeddings).
//
//   dishcovery-indexer --docs data/docs.jsonl --vectors data/emb.f32 \
//       --schema configs/schema.json --out data/index/shard_0 \
//       --shard 0 --num-shards 4 --threads 8 --segment-docs 250000 --max-segments 4
//
// Documents are routed by fnv1a64(restaurant_id) % num_shards (identical to
// dishcovery/hashing.py). Worker threads parse JSON and fill independent
// segment builders in parallel; each full builder is written as an on-disk
// segment (with its HNSW graph). Finally the segments are committed and
// force-merged down to --max-segments.
//
// --vectors is a raw little-endian float32 matrix with one row per line of
// --docs (row i = line i, before shard filtering).
#include <atomic>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include "index/shard_index.h"
#include "util/args.h"
#include "util/hash.h"
#include "util/log.h"
#include "util/timer.h"

using namespace dc;

namespace {

struct Chunk {
  uint64_t first_line = 0;
  std::vector<std::string> lines;
};

class ChunkQueue {
 public:
  explicit ChunkQueue(size_t cap) : cap_(cap) {}
  void push(Chunk c) {
    std::unique_lock<std::mutex> lk(mu_);
    not_full_.wait(lk, [&] { return q_.size() < cap_; });
    q_.push_back(std::move(c));
    not_empty_.notify_one();
  }
  bool pop(Chunk& out) {
    std::unique_lock<std::mutex> lk(mu_);
    not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    not_full_.notify_one();
    return true;
  }
  void close() {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    not_empty_.notify_all();
  }

 private:
  size_t cap_;
  std::deque<Chunk> q_;
  bool closed_ = false;
  std::mutex mu_;
  std::condition_variable not_full_, not_empty_;
};

}  // namespace

int main(int argc, char** argv) {
  Args a(argc, argv);
  if (a.has("help") || !a.has("docs") || !a.has("out")) {
    std::cerr << "usage: dishcovery-indexer --docs FILE --out DIR [--vectors FILE.f32] [--schema FILE]\n"
                 "         [--shard 0 --num-shards 1] [--threads N] [--segment-docs 250000]\n"
                 "         [--max-segments 0 (no force merge)] [--codec pfor|vbyte|raw]\n"
                 "         [--hnsw-m 16] [--hnsw-efc 200] [--hnsw-min-docs 2000] [--limit N]\n";
    return 2;
  }
  try {
    Stopwatch total;
    index::Schema schema =
        a.has("schema") ? index::Schema::load(a.str("schema")) : index::Schema::default_schema();
    const uint32_t shard = static_cast<uint32_t>(a.integer("shard", 0));
    const uint32_t num_shards = static_cast<uint32_t>(a.integer("num-shards", 1));
    const unsigned threads = static_cast<unsigned>(a.integer("threads", std::max(1u, std::thread::hardware_concurrency())));
    const uint32_t segment_docs = static_cast<uint32_t>(a.integer("segment-docs", 250000));
    const int max_segments = static_cast<int>(a.integer("max-segments", 0));
    const uint64_t limit = static_cast<uint64_t>(a.integer("limit", 0));
    const std::string out = a.required("out");

    index::IndexOptions io;
    io.codec = codec::codec_from_name(a.str("codec", "pfor"));
    io.background = false;
    io.wal = false;
    io.hnsw.M = static_cast<uint32_t>(a.integer("hnsw-m", 16));
    io.hnsw.ef_construction = static_cast<uint32_t>(a.integer("hnsw-efc", 200));
    io.hnsw_min_docs = static_cast<uint32_t>(a.integer("hnsw-min-docs", 2000));
    io.build_threads = threads;
    if (a.has("merge-policy")) io.merge_policy = json::parse(a.str("merge-policy"));
    if (path_exists(join_path(out, "COMMIT"))) fail(out + " already contains an index; remove it first");
    index::ShardIndex idx(out, schema, io);
    idx.open();

    std::unique_ptr<MMapFile> vec_file;
    const float* vectors = nullptr;
    uint64_t vec_rows = 0;
    if (a.has("vectors") && schema.vector_dim > 0) {
      vec_file = std::make_unique<MMapFile>(a.str("vectors"));
      vectors = reinterpret_cast<const float*>(vec_file->span().data);
      vec_rows = vec_file->size() / (4ull * schema.vector_dim);
      DC_CHECK(vec_file->size() % (4ull * schema.vector_dim) == 0, "vectors file size is not a multiple of dim*4");
      LOG_INFO("vectors: " << vec_rows << " rows x " << schema.vector_dim);
    }

    // Segment builds hold one HNSW build per worker, so give each 1 thread.
    index::SegmentWriteOptions wo;
    wo.codec = io.codec;
    wo.build_hnsw = true;
    wo.hnsw = io.hnsw;
    wo.hnsw_min_docs = io.hnsw_min_docs;
    wo.threads = 1;
    wo.source = "build";

    ChunkQueue queue(threads * 4);
    std::atomic<uint64_t> seen{0}, indexed{0}, skipped{0}, bad{0};
    std::mutex uids_mu;
    std::vector<std::string> uids;

    auto write_segment = [&](index::SegmentBuilder& b) {
      if (b.num_docs() == 0) return;
      std::string uid = idx.new_segment_uid();
      DirStorage storage(join_path(out, uid), true);
      auto opts = wo;
      opts.uid = uid;
      b.write(storage, opts);
      std::lock_guard<std::mutex> lk(uids_mu);
      uids.push_back(uid);
    };

    std::vector<std::thread> workers;
    for (unsigned t = 0; t < threads; ++t) {
      workers.emplace_back([&] {
        auto builder = std::make_unique<index::SegmentBuilder>(schema);
        Chunk c;
        while (queue.pop(c)) {
          for (size_t i = 0; i < c.lines.size(); ++i) {
            uint64_t line_no = c.first_line + i;
            try {
              json j = json::parse(c.lines[i]);
              std::string rid = j.value("restaurant_id", j.value("id", std::string()));
              if (num_shards > 1 && shard_of(rid, num_shards) != shard) {
                skipped.fetch_add(1);
                continue;
              }
              const float* v = nullptr;
              if (vectors) {
                DC_CHECK(line_no < vec_rows, "vectors file has fewer rows than docs");
                v = vectors + line_no * schema.vector_dim;
              }
              builder->add(index::parse_document(j, schema, v));
              indexed.fetch_add(1);
            } catch (const std::exception& e) {
              if (bad.fetch_add(1) < 10) LOG_WARN("line " << line_no << ": " << e.what());
            }
            if (builder->num_docs() >= segment_docs) {
              write_segment(*builder);
              builder = std::make_unique<index::SegmentBuilder>(schema);
            }
          }
        }
        write_segment(*builder);
      });
    }

    std::ifstream in(a.required("docs"));
    DC_CHECK(in.good(), "cannot open " + a.str("docs"));
    Chunk chunk;
    std::string line;
    uint64_t line_no = 0;
    Stopwatch progress;
    while (std::getline(in, line)) {
      if (limit && line_no >= limit) break;
      if (chunk.lines.empty()) chunk.first_line = line_no;
      chunk.lines.push_back(std::move(line));
      ++line_no;
      if (chunk.lines.size() == 1024) {
        queue.push(std::move(chunk));
        chunk = Chunk();
      }
      if (progress.elapsed_s() > 10) {
        LOG_INFO("read " << line_no << " lines, indexed " << indexed.load() << " ("
                         << indexed.load() / std::max(1e-9, total.elapsed_s()) << " docs/s)");
        progress.reset();
      }
    }
    if (!chunk.lines.empty()) queue.push(std::move(chunk));
    queue.close();
    for (auto& w : workers) w.join();
    seen = line_no;
    double build_s = total.elapsed_s();

    std::sort(uids.begin(), uids.end());
    for (auto& u : uids) idx.adopt_segment(u);
    idx.commit();
    LOG_INFO("built " << uids.size() << " segments, " << indexed.load() << " docs in " << build_s << " s");
    Stopwatch merge_sw;
    if (max_segments > 0) idx.force_merge(max_segments);
    double merge_s = merge_sw.elapsed_s();

    json st = idx.stats();
    json report = {{"shard", shard},
                   {"num_shards", num_shards},
                   {"lines", seen.load()},
                   {"indexed", indexed.load()},
                   {"skipped_other_shards", skipped.load()},
                   {"bad_lines", bad.load()},
                   {"build_seconds", build_s},
                   {"merge_seconds", merge_s},
                   {"total_seconds", total.elapsed_s()},
                   {"docs_per_second", indexed.load() / std::max(build_s, 1e-9)},
                   {"index_bytes", dir_size_bytes(out)},
                   {"segments", st["num_segments"]},
                   {"codec", codec::codec_name(io.codec)}};
    write_file_atomic(join_path(out, "build_report.json"), report.dump(2));
    std::cout << report.dump() << std::endl;
  } catch (const std::exception& e) {
    LOG_ERROR(e.what());
    return 1;
  }
  return 0;
}
