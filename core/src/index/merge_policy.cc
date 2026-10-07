#include "index/merge_policy.h"

#include <algorithm>
#include <cmath>
#include <map>

#include "util/common.h"

namespace dc::index {

std::vector<MergeSpec> MergePolicy::find_forced_merges(const std::vector<SegmentStat>& segs, int max_segments,
                                                       uint32_t max_merge_at_once) const {
  std::vector<MergeSpec> out;
  if (max_segments < 1) max_segments = 1;
  if (static_cast<int>(segs.size()) <= max_segments) {
    // Still expunge heavy deletes on a forced merge.
    for (auto& s : segs)
      if (s.del_docs > 0 && s.live_ratio() < 0.9) out.push_back({{s.uid}, "forced-expunge"});
    return out;
  }
  // Merge the smallest segments together so that the result count fits.
  std::vector<SegmentStat> sorted = segs;
  std::sort(sorted.begin(), sorted.end(),
            [](const SegmentStat& a, const SegmentStat& b) { return a.effective_bytes() < b.effective_bytes(); });
  size_t excess = segs.size() - static_cast<size_t>(max_segments) + 1;
  MergeSpec spec;
  spec.reason = "forced";
  for (size_t i = 0; i < excess && i < sorted.size(); ++i) {
    spec.uids.push_back(sorted[i].uid);
    if (spec.uids.size() == max_merge_at_once) {
      out.push_back(spec);
      spec.uids.clear();
    }
  }
  if (spec.uids.size() > 1) out.push_back(spec);
  return out;
}

std::vector<MergeSpec> TieredMergePolicy::find_merges(const std::vector<SegmentStat>& segs,
                                                      const std::set<std::string>& merging) const {
  std::vector<SegmentStat> eligible;
  uint64_t total = 0;
  uint64_t min_size = UINT64_MAX;
  for (auto& s : segs) {
    if (merging.count(s.uid)) continue;
    eligible.push_back(s);
    total += static_cast<uint64_t>(s.effective_bytes());
    min_size = std::min<uint64_t>(min_size, static_cast<uint64_t>(s.effective_bytes()));
  }
  if (eligible.size() < 2) {
    // Lone segment with too many deletes: rewrite it.
    if (eligible.size() == 1 && eligible[0].del_docs &&
        100.0 * (1.0 - eligible[0].live_ratio()) > p_.deletes_pct_allowed)
      return {{{eligible[0].uid}, "expunge-deletes"}};
    return {};
  }
  std::sort(eligible.begin(), eligible.end(), [](const SegmentStat& a, const SegmentStat& b) {
    return a.effective_bytes() > b.effective_bytes();
  });

  // Segment budget: segments_per_tier per tier, tiers growing by max_merge_at_once.
  double level_size = std::max<double>(static_cast<double>(min_size), static_cast<double>(p_.floor_segment_bytes));
  double bytes_left = static_cast<double>(total);
  double allowed = 0;
  while (true) {
    double count = bytes_left / level_size;
    if (count < p_.segments_per_tier) {
      allowed += std::ceil(count);
      break;
    }
    allowed += p_.segments_per_tier;
    bytes_left -= p_.segments_per_tier * level_size;
    level_size *= p_.max_merge_at_once;
  }
  allowed = std::max(allowed, p_.segments_per_tier);

  // Segments too large to be merged further are excluded from candidates.
  std::vector<SegmentStat> candidates;
  for (auto& s : eligible)
    if (s.effective_bytes() < p_.max_merged_segment_bytes / 2.0) candidates.push_back(s);

  if (static_cast<double>(eligible.size()) <= allowed) {
    // Within budget; still reclaim segments dominated by deletes.
    for (auto& s : eligible)
      if (s.del_docs && 100.0 * (1.0 - s.live_ratio()) > p_.deletes_pct_allowed)
        return {{{s.uid}, "expunge-deletes"}};
    return {};
  }

  // Score every window of up to max_merge_at_once consecutive (size-sorted)
  // segments; lower is better. Skew penalises merging one big segment with
  // tiny ones (pure write amplification); the size exponent mildly prefers
  // smaller merges; reclaiming deletes is rewarded.
  double best_score = 1e300;
  size_t best_start = 0, best_len = 0;
  for (size_t start = 0; start + 1 < candidates.size(); ++start) {
    double sum = 0, max_seg = 0, live = 0, raw = 0;
    size_t len = 0;
    for (size_t i = start; i < candidates.size() && len < p_.max_merge_at_once; ++i) {
      double eb = candidates[i].effective_bytes();
      if (sum + eb > p_.max_merged_segment_bytes) break;
      sum += eb;
      max_seg = std::max(max_seg, eb);
      live += eb;
      raw += static_cast<double>(candidates[i].size_bytes);
      ++len;
    }
    if (len < 2) continue;
    double floor_sum = std::max(sum, static_cast<double>(p_.floor_segment_bytes));
    double skew = len == p_.max_merge_at_once ? 1.0 / p_.max_merge_at_once
                                               : std::max(max_seg, static_cast<double>(p_.floor_segment_bytes)) / floor_sum;
    double del_factor = raw > 0 ? std::pow(live / raw, 2.0) : 1.0;
    double score = skew * std::pow(floor_sum, 0.05) * del_factor;
    if (score < best_score) {
      best_score = score;
      best_start = start;
      best_len = len;
    }
  }
  if (best_len < 2) return {};
  MergeSpec spec;
  spec.reason = "tiered";
  for (size_t i = best_start; i < best_start + best_len; ++i) spec.uids.push_back(candidates[i].uid);
  return {spec};
}

json TieredMergePolicy::describe() const {
  return {{"type", "tiered"},
          {"max_merge_at_once", p_.max_merge_at_once},
          {"segments_per_tier", p_.segments_per_tier},
          {"floor_segment_bytes", p_.floor_segment_bytes},
          {"max_merged_segment_bytes", p_.max_merged_segment_bytes},
          {"deletes_pct_allowed", p_.deletes_pct_allowed}};
}

std::vector<MergeSpec> LogMergePolicy::find_merges(const std::vector<SegmentStat>& segs,
                                                   const std::set<std::string>& merging) const {
  // Assign levels; segments keep creation order. Merge the first run of
  // merge_factor adjacent, non-merging segments that share a level.
  auto level_of = [&](const SegmentStat& s) {
    double sz = std::max<double>(s.effective_bytes(), static_cast<double>(p_.min_merge_bytes));
    return static_cast<int>(std::floor(std::log(sz / p_.min_merge_bytes) / std::log(p_.merge_factor)));
  };
  std::map<int, std::vector<const SegmentStat*>> by_level;
  for (auto& s : segs) {
    if (merging.count(s.uid)) continue;
    if (s.effective_bytes() >= p_.max_merge_bytes) continue;
    by_level[level_of(s)].push_back(&s);
  }
  std::vector<MergeSpec> out;
  for (auto& [lvl, v] : by_level) {
    if (v.size() >= p_.merge_factor) {
      MergeSpec spec;
      spec.reason = "log-level-" + std::to_string(lvl);
      for (size_t i = 0; i < p_.merge_factor; ++i) spec.uids.push_back(v[i]->uid);
      out.push_back(std::move(spec));
      break;  // one merge per call; the caller re-evaluates afterwards
    }
  }
  return out;
}

json LogMergePolicy::describe() const {
  return {{"type", "log"},
          {"merge_factor", p_.merge_factor},
          {"min_merge_bytes", p_.min_merge_bytes},
          {"max_merge_bytes", p_.max_merge_bytes}};
}

std::unique_ptr<MergePolicy> make_merge_policy(const json& cfg) {
  std::string type = cfg.value("type", std::string("tiered"));
  if (type == "tiered") {
    TieredMergePolicy::Params p;
    p.max_merge_at_once = cfg.value("max_merge_at_once", p.max_merge_at_once);
    p.segments_per_tier = cfg.value("segments_per_tier", p.segments_per_tier);
    p.floor_segment_bytes = cfg.value("floor_segment_bytes", p.floor_segment_bytes);
    p.max_merged_segment_bytes = cfg.value("max_merged_segment_bytes", p.max_merged_segment_bytes);
    p.deletes_pct_allowed = cfg.value("deletes_pct_allowed", p.deletes_pct_allowed);
    return std::make_unique<TieredMergePolicy>(p);
  }
  if (type == "log") {
    LogMergePolicy::Params p;
    p.merge_factor = cfg.value("merge_factor", p.merge_factor);
    p.min_merge_bytes = cfg.value("min_merge_bytes", p.min_merge_bytes);
    p.max_merge_bytes = cfg.value("max_merge_bytes", p.max_merge_bytes);
    return std::make_unique<LogMergePolicy>(p);
  }
  if (type == "none") return std::make_unique<NoMergePolicy>();
  fail("unknown merge policy: " + type);
}

}  // namespace dc::index
