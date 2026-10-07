#include "search/filter.h"

#include <algorithm>

#include "geo/geo.h"
#include "index/document.h"

namespace dc::search {

RoaringBitmap union_keys(const index::SegmentReader& seg, const std::vector<std::string>& keys) {
  std::vector<RoaringBitmap> parts;
  for (const auto& k : keys)
    if (auto ti = seg.filter_entry(k)) parts.push_back(seg.filter_bitmap(*ti));
  if (parts.empty()) return {};
  if (parts.size() == 1) return std::move(parts[0]);
  std::vector<const RoaringBitmap*> ptrs;
  for (auto& p : parts) ptrs.push_back(&p);
  return RoaringBitmap::or_many(ptrs);
}

RoaringBitmap FilterEngine::compute(const index::SegmentReader& seg, const FilterSpec& f) {
  const auto& schema = seg.schema();
  std::vector<RoaringBitmap> sets;

  for (const auto& [field, values] : f.terms) {
    std::vector<std::string> keys;
    for (const auto& v : values) keys.push_back(index::make_key(static_cast<uint8_t>(field), v));
    sets.push_back(union_keys(seg, keys));
    if (sets.back().empty()) return {};
  }

  bool radius_check = false;
  if (f.geo.mode == GeoFilter::Mode::kDelivery) {
    // Covers are compacted, so a restaurant delivering to the user may be keyed
    // on any ancestor of the user's cell.
    uint64_t cell = geo::cell_of(f.geo.lat, f.geo.lon, schema.geo.cover_res);
    std::vector<std::string> keys;
    for (int r = 0; r <= schema.geo.cover_res && cell; ++r)
      keys.push_back(index::make_cell_key(index::kGeoCoverField, geo::parent_of(cell, r)));
    sets.push_back(union_keys(seg, keys));
    if (sets.back().empty()) return {};
  } else if (f.geo.mode == GeoFilter::Mode::kRadius) {
    std::vector<std::string> keys;
    for (uint64_t c : geo::disk_cells(f.geo.lat, f.geo.lon, f.geo.radius_km, schema.geo.point_res))
      keys.push_back(index::make_cell_key(index::kGeoPointField, c));
    sets.push_back(union_keys(seg, keys));
    if (sets.back().empty()) return {};
    radius_check = true;
  }

  // Intersect smallest-first.
  std::sort(sets.begin(), sets.end(),
            [](const RoaringBitmap& a, const RoaringBitmap& b) { return a.cardinality() < b.cardinality(); });
  RoaringBitmap acc;
  bool have = false;
  for (auto& s : sets) {
    if (!have) {
      acc = std::move(s);
      have = true;
    } else {
      acc.and_inplace(s);
    }
    if (acc.empty()) return acc;
  }

  // Doc-value predicates.
  bool need_scan = radius_check || !f.ranges.empty() || f.open_slot >= 0;
  if (need_scan) {
    auto pass = [&](uint32_t d) {
      for (const auto& r : f.ranges) {
        float v = seg.numeric(static_cast<uint32_t>(r.field), d);
        if (v < r.lo || v > r.hi) return false;
      }
      if (f.open_slot >= 0 && !index::hours_open(seg.hours(d), f.open_slot)) return false;
      if (radius_check && geo::haversine_km(f.geo.lat, f.geo.lon, seg.lat(d), seg.lon(d)) > f.geo.radius_km)
        return false;
      return true;
    };
    std::vector<uint32_t> keep;
    if (have) {
      acc.for_each([&](uint32_t d) {
        if (pass(d)) keep.push_back(d);
      });
    } else {
      for (uint32_t d = 0; d < seg.num_docs(); ++d)
        if (pass(d)) keep.push_back(d);
    }
    acc = RoaringBitmap::from_sorted(keep.data(), keep.size());
    have = true;
  }

  if (!have) acc = RoaringBitmap::from_range(0, seg.num_docs());
  for (const auto& [field, values] : f.exclude) {
    std::vector<std::string> keys;
    for (const auto& v : values) keys.push_back(index::make_key(static_cast<uint8_t>(field), v));
    acc.andnot_inplace(union_keys(seg, keys));
  }
  acc.run_optimize();
  return acc;
}

std::shared_ptr<const RoaringBitmap> FilterEngine::materialize(const index::SegmentReader& seg, const FilterSpec& f,
                                                               const std::string& cache_key, bool use_cache,
                                                               bool* cache_hit) {
  if (cache_hit) *cache_hit = false;
  if (f.empty()) return nullptr;
  std::string key = seg.uid() + "|" + cache_key;
  bool cacheable = use_cache && seg.on_disk();
  if (cacheable) {
    if (auto v = cache_.get(key)) {
      if (cache_hit) *cache_hit = true;
      return v;
    }
  }
  auto bm = std::make_shared<const RoaringBitmap>(compute(seg, f));
  if (cacheable) cache_.put(key, bm, bm->memory_bytes() + key.size() + 64);
  return bm;
}

}  // namespace dc::search
