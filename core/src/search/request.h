// Shard-level search request/response types and their JSON wire format.
//
// Phase 1 (/search):   retrieval paths (bm25, exact, vector, geo) → per-path
//                      top-k (segment, doc, score) under a point-in-time context.
// Phase 2 (/features): ranking signals for the candidates the broker kept.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "index/schema.h"
#include "util/common.h"
#include "util/json.h"

namespace dc::search {

enum class FilterMode { kFilterAware, kPostFilter, kPostFilterTopK };

const char* filter_mode_name(FilterMode m);
FilterMode filter_mode_from_name(const std::string& s);

struct QueryTerm {
  std::string text;      // surface form; analysed per field on the shard
  float weight = 1.0f;
  int group = 0;         // original query position this term belongs to
  bool primary = true;   // false for synonym / spelling-variant expansions
};

struct GeoFilter {
  enum class Mode { kNone, kDelivery, kRadius };
  Mode mode = Mode::kNone;
  double lat = 0, lon = 0;
  double radius_km = 0;
};

struct FilterSpec {
  // keyword field id → allowed normalised values (OR within a field, AND across fields)
  std::vector<std::pair<int, std::vector<std::string>>> terms;
  std::vector<std::pair<int, std::vector<std::string>>> exclude;
  struct Range {
    int field;
    float lo;
    float hi;
  };
  std::vector<Range> ranges;  // numeric doc-value ranges (inclusive)
  int open_slot = -1;         // half-hour slot (0..335) the doc must be open at
  GeoFilter geo;

  bool empty() const {
    return terms.empty() && exclude.empty() && ranges.empty() && open_slot < 0 && geo.mode == GeoFilter::Mode::kNone;
  }
  // Canonical string used as the filter-cache key. Geo coordinates are
  // quantised to the H3 cell so nearby users share cache entries.
  std::string cache_key(const index::Schema& schema) const;
};

struct SearchRequest {
  std::vector<QueryTerm> terms;
  std::vector<float> field_weights;  // per text field; empty = schema weights
  FilterSpec filter;
  FilterMode filter_mode = FilterMode::kFilterAware;
  uint32_t oversample = 10;          // post_filter_topk: retrieve k*oversample, then filter

  bool bm25 = true;
  uint32_t bm25_k = 1000;
  bool exact = false;
  uint32_t exact_k = 100;
  bool vector = false;
  uint32_t vector_k = 500;
  uint32_t ef = 128;
  std::vector<float> qvec;
  uint32_t brute_force_max = 20000;  // filtered vector search scans when |filter| is below this
  bool geo = false;
  uint32_t geo_k = 500;
  float geo_tau_km = 3.0f;

  bool use_filter_cache = true;
  bool use_postings_cache = true;
  bool debug = false;

  static SearchRequest from_json(const json& j, const index::Schema& schema);
};

struct Hit {
  uint32_t seg;
  DocId doc;
  float score;
};

struct PathResult {
  std::vector<Hit> hits;
  uint64_t scored = 0;    // documents whose score was computed
  uint64_t matched = 0;   // documents that matched (where known)
  uint64_t time_us = 0;
};

struct SearchResponse {
  uint64_t generation = 0;
  PathResult bm25, exact, vector, geo;
  uint64_t filter_us = 0;
  uint64_t filter_cardinality = 0;  // docs passing the filter (summed over segments)
  uint64_t filter_cache_hits = 0;
  uint64_t analyze_us = 0;
  uint64_t total_us = 0;
  uint32_t segments = 0;
};

struct FeatureRequest {
  std::vector<QueryTerm> terms;
  std::vector<float> field_weights;
  std::vector<float> qvec;
  double user_lat = NAN, user_lon = NAN;
  int slot = -1;
  std::vector<std::pair<uint32_t, DocId>> docs;  // (segment ordinal in the context snapshot, doc)
  bool include_stored = true;

  static FeatureRequest from_json(const json& j, const index::Schema& schema);
};

struct FeatureResponse {
  std::vector<std::string> names;
  std::vector<std::vector<float>> rows;     // aligned with request docs
  std::vector<std::string_view> ids;
  std::vector<std::string_view> stored;     // raw JSON, borrowed from the snapshot
  uint64_t time_us = 0;
};

std::vector<QueryTerm> parse_terms(const json& j);
std::vector<float> parse_field_weights(const json& j, const index::Schema& schema);

}  // namespace dc::search
