// Query execution over a point-in-time snapshot of one shard.
//
// Retrieval paths (each returns its own top-k; the broker fuses them with RRF):
//   bm25    weighted multi-field BM25 over (field, term) clauses with MaxScore
//           dynamic pruning. In filter-aware mode the filter bitmap's iterator
//           participates in the leapfrog, so only admissible docs are scored;
//           post_filter scores every match and filters afterwards (the naive
//           baseline); post_filter_topk over-fetches k*oversample unfiltered.
//   exact   conjunction of all query tokens on the un-stemmed name field,
//           ranked by popularity (the exact-match baseline).
//   vector  HNSW over unit embeddings, filtered; brute force when the filter
//           admits fewer than brute_force_max docs (or for RAM segments).
//   geo     admissible docs near the user ranked by popularity × distance decay.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "index/shard_index.h"
#include "search/filter.h"
#include "search/request.h"
#include "util/lru_cache.h"

namespace dc::search {

struct SearcherOptions {
  size_t filter_cache_bytes = 256ull << 20;
  size_t postings_cache_bytes = 256ull << 20;
  uint32_t postings_cache_min_df = 4096;  // only cache long (expensive to decode) lists
};

struct DecodedPostings {
  std::vector<uint32_t> docs;
  std::vector<uint32_t> freqs;
  size_t bytes() const { return (docs.size() + freqs.size()) * 4 + 64; }
};

// Iterates either a live (block-decoding) postings stream or a cached,
// fully decoded list.
class TermCursor {
 public:
  void init_stream(index::PostingsIterator it) {
    stream_ = std::move(it);
    cached_.reset();
    doc_ = kUnpositioned;
  }
  void init_cached(std::shared_ptr<const DecodedPostings> p) {
    cached_ = std::move(p);
    idx_ = 0;
    doc_ = kUnpositioned;
  }
  DocId doc() const { return cached_ ? doc_ : stream_.doc(); }
  DocId next();
  DocId advance(DocId target);
  uint32_t freq() { return cached_ ? cached_->freqs[idx_] : stream_.freq(); }
  uint64_t cost() const { return cached_ ? cached_->docs.size() : stream_.cost(); }

 private:
  index::PostingsIterator stream_;
  std::shared_ptr<const DecodedPostings> cached_;
  size_t idx_ = 0;
  DocId doc_ = kUnpositioned;
};

// One (field, analysed term) scoring clause with snapshot-global statistics.
struct Clause {
  uint8_t field = 0;
  std::string term;
  float weight = 0;     // query weight × field weight
  int group = 0;
  bool primary = true;
  uint64_t df = 0;      // summed over segments
  float idf = 0;
  float avgdl = 0;
};

class Searcher {
 public:
  Searcher(const index::Schema& schema, SearcherOptions opt);

  SearchResponse search(const index::Snapshot& snap, const SearchRequest& req);
  FeatureResponse features(const index::Snapshot& snap, const FeatureRequest& req);

  // Analyse query terms into clauses. `merge_duplicates` sums weights of
  // repeated (field, term) pairs (retrieval); features keep them per group.
  std::vector<Clause> compile(const index::Snapshot& snap, const std::vector<QueryTerm>& terms,
                              const std::vector<float>& field_weights, bool merge_duplicates,
                              bool include_zero_weight) const;

  void retire_segment(const std::string& uid);
  json cache_stats() const;
  FilterEngine& filters() { return filters_; }

 private:
  void run_bm25(const index::Snapshot& snap, const SearchRequest& req,
                const std::vector<std::shared_ptr<const RoaringBitmap>>& fsets, PathResult& out);
  void run_exact(const index::Snapshot& snap, const SearchRequest& req,
                 const std::vector<std::shared_ptr<const RoaringBitmap>>& fsets, PathResult& out);
  void run_vector(const index::Snapshot& snap, const SearchRequest& req,
                  const std::vector<std::shared_ptr<const RoaringBitmap>>& fsets, PathResult& out);
  void run_geo(const index::Snapshot& snap, const SearchRequest& req,
               const std::vector<std::shared_ptr<const RoaringBitmap>>& fsets, PathResult& out);
  bool open_cursor(const index::SegmentReader& seg, const Clause& c, bool use_cache, TermCursor& cur,
                   index::TermInfo* info_out);

  const index::Schema& schema_;
  SearcherOptions opt_;
  FilterEngine filters_;
  mutable LruCache<DecodedPostings> postings_cache_;
  int f_popularity_, f_rating_, f_name_exact_, f_name_;
};

}  // namespace dc::search
