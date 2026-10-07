// Merge policies decide which immutable segments to merge.
//
// The trade-off: merging too eagerly rewrites the same bytes many times
// (write amplification, I/O); merging too lazily leaves many small segments
// and every query pays per-segment overhead (read amplification → p99).
//
//   tiered   Lucene-style TieredMergePolicy: allow ~segments_per_tier segments
//            per size tier; when over budget pick the cheapest, least-skewed
//            run of up to max_merge_at_once similarly-sized segments.
//   log      leveled LogByteSizeMergePolicy: level = floor(log_mf(size/min));
//            merge merge_factor segments as soon as a level fills up.
//   none     never merge (worst-case read amplification baseline).
#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "util/json.h"

namespace dc::index {

struct SegmentStat {
  std::string uid;
  uint64_t size_bytes = 0;
  uint32_t num_docs = 0;
  uint32_t del_docs = 0;
  double live_ratio() const { return num_docs ? 1.0 - static_cast<double>(del_docs) / num_docs : 1.0; }
  double effective_bytes() const { return size_bytes * live_ratio(); }
};

struct MergeSpec {
  std::vector<std::string> uids;
  std::string reason;
};

class MergePolicy {
 public:
  virtual ~MergePolicy() = default;
  // Segments are in creation order (oldest first). `merging` holds uids
  // already participating in a running merge.
  virtual std::vector<MergeSpec> find_merges(const std::vector<SegmentStat>& segs,
                                             const std::set<std::string>& merging) const = 0;
  // Merge down to at most max_segments segments.
  std::vector<MergeSpec> find_forced_merges(const std::vector<SegmentStat>& segs, int max_segments,
                                            uint32_t max_merge_at_once = 50) const;
  virtual json describe() const = 0;
};

class TieredMergePolicy : public MergePolicy {
 public:
  struct Params {
    uint32_t max_merge_at_once = 10;
    double segments_per_tier = 10.0;
    uint64_t floor_segment_bytes = 2ull << 20;
    uint64_t max_merged_segment_bytes = 5ull << 30;
    double deletes_pct_allowed = 20.0;
  };
  explicit TieredMergePolicy(Params p) : p_(p) {}
  std::vector<MergeSpec> find_merges(const std::vector<SegmentStat>& segs,
                                     const std::set<std::string>& merging) const override;
  json describe() const override;

 private:
  Params p_;
};

class LogMergePolicy : public MergePolicy {
 public:
  struct Params {
    uint32_t merge_factor = 10;
    uint64_t min_merge_bytes = 1ull << 20;
    uint64_t max_merge_bytes = 5ull << 30;
  };
  explicit LogMergePolicy(Params p) : p_(p) {}
  std::vector<MergeSpec> find_merges(const std::vector<SegmentStat>& segs,
                                     const std::set<std::string>& merging) const override;
  json describe() const override;

 private:
  Params p_;
};

class NoMergePolicy : public MergePolicy {
 public:
  std::vector<MergeSpec> find_merges(const std::vector<SegmentStat>&, const std::set<std::string>&) const override {
    return {};
  }
  json describe() const override { return {{"type", "none"}}; }
};

// {"type": "tiered"|"log"|"none", ...params}
std::unique_ptr<MergePolicy> make_merge_policy(const json& cfg);

}  // namespace dc::index
