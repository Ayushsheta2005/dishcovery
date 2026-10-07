// HTTP front-end of one shard replica.
//
//   GET  /health                     liveness + generation + doc counts
//   POST /search                     phase 1: retrieval paths → hits + context id
//   POST /features                   phase 2: ranking signals for (seg, doc) pairs of a context
//   POST /index[?refresh=true]       add/update documents (JSON object, array, or JSON lines)
//   POST /delete                     {"ids": [...]}
//   GET  /doc?id=...                 stored fields of a live document (visibility probes)
//   GET  /stats                      index, cache and per-endpoint latency statistics
//   POST /admin/{refresh,flush,merge,maybe_merge,cache/clear,fault}
//
// Phase-2 requests reference a point-in-time context created by phase 1, so
// (segment, doc) addresses stay valid even if a refresh or merge lands in
// between. Contexts expire after ctx_ttl_ms.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <unordered_map>

#include "index/shard_index.h"
#include "search/searcher.h"
#include "util/histogram.h"

namespace httplib {
class Server;
}

namespace dc::server {

struct ShardServerOptions {
  std::string host = "0.0.0.0";
  int port = 7001;
  int threads = 64;  // cpp-httplib holds one thread per keep-alive connection
  uint32_t shard_id = 0;
  std::string replica = "r0";
  uint32_t ctx_ttl_ms = 60000;
  size_t max_contexts = 50000;
};

class ShardServer {
 public:
  ShardServer(index::ShardIndex& idx, search::Searcher& searcher, ShardServerOptions opt);
  ~ShardServer();
  void run();   // blocks until stop()
  void stop();

 private:
  struct Context {
    std::shared_ptr<const index::Snapshot> snap;
    uint64_t expires_ns;
  };
  uint64_t put_context(std::shared_ptr<const index::Snapshot> snap);
  std::shared_ptr<const index::Snapshot> get_context(uint64_t id);
  void retire_dropped_segments(const index::Snapshot& snap);
  // Returns false (and the caller replies 503) when an injected fault fires.
  bool maybe_inject_fault();

  index::ShardIndex& idx_;
  search::Searcher& searcher_;
  ShardServerOptions opt_;
  std::unique_ptr<httplib::Server> http_;

  std::mutex ctx_mu_;
  std::unordered_map<uint64_t, Context> contexts_;
  std::atomic<uint64_t> next_ctx_{1};

  std::mutex retire_mu_;
  uint64_t seen_generation_ = 0;
  std::set<std::string> seen_uids_;

  std::mutex fault_mu_;
  double fault_delay_ms_ = 0, fault_delay_prob_ = 0, fault_error_prob_ = 0;
  std::mt19937_64 rng_{12345};

  Histogram search_us_, features_us_, index_us_;
  std::atomic<uint64_t> searches_{0}, feature_calls_{0}, ctx_misses_{0}, errors_{0};
};

}  // namespace dc::server
