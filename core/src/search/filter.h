// Filter materialisation: turns a FilterSpec into a per-segment Roaring bitmap
// of admissible docs (tombstones are applied separately, at collection time).
//
// Keyword filters and H3 geo keys are bitmap lookups + set algebra; numeric
// ranges and "open at slot" are evaluated over doc values, restricted to the
// bitmap built so far. Results for immutable on-disk segments are cached
// (the "filter cache"); a segment's entries are dropped when it is retired.
#pragma once

#include <memory>
#include <string>

#include "codec/roaring.h"
#include "index/segment.h"
#include "search/request.h"
#include "util/lru_cache.h"

namespace dc::search {

class FilterEngine {
 public:
  explicit FilterEngine(size_t cache_bytes) : cache_(cache_bytes) {}

  // nullptr means "no filter" (all docs admissible).
  std::shared_ptr<const RoaringBitmap> materialize(const index::SegmentReader& seg, const FilterSpec& f,
                                                   const std::string& cache_key, bool use_cache, bool* cache_hit);

  void invalidate_segment(const std::string& uid) { cache_.invalidate_prefix(uid + "|"); }
  LruCache<RoaringBitmap>& cache() { return cache_; }

 private:
  RoaringBitmap compute(const index::SegmentReader& seg, const FilterSpec& f);
  LruCache<RoaringBitmap> cache_;
};

// Union of the bitmaps for a list of filter-dictionary keys.
RoaringBitmap union_keys(const index::SegmentReader& seg, const std::vector<std::string>& keys);

}  // namespace dc::search
