#include "search/request.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "analysis/analyzer.h"
#include "geo/geo.h"
#include "util/base64.h"

namespace dc::search {

const char* filter_mode_name(FilterMode m) {
  switch (m) {
    case FilterMode::kFilterAware: return "filter_aware";
    case FilterMode::kPostFilter: return "post_filter";
    case FilterMode::kPostFilterTopK: return "post_filter_topk";
  }
  return "?";
}

FilterMode filter_mode_from_name(const std::string& s) {
  if (s == "filter_aware") return FilterMode::kFilterAware;
  if (s == "post_filter") return FilterMode::kPostFilter;
  if (s == "post_filter_topk") return FilterMode::kPostFilterTopK;
  fail("unknown filter_mode: " + s);
}

std::string FilterSpec::cache_key(const index::Schema& schema) const {
  std::ostringstream os;
  auto sorted_terms = terms;
  std::sort(sorted_terms.begin(), sorted_terms.end());
  for (auto& [f, vals] : sorted_terms) {
    auto v = vals;
    std::sort(v.begin(), v.end());
    os << "t" << f << "=";
    for (auto& x : v) os << x << ",";
    os << ";";
  }
  auto sorted_ex = exclude;
  std::sort(sorted_ex.begin(), sorted_ex.end());
  for (auto& [f, vals] : sorted_ex) {
    auto v = vals;
    std::sort(v.begin(), v.end());
    os << "x" << f << "=";
    for (auto& x : v) os << x << ",";
    os << ";";
  }
  for (auto& r : ranges) os << "r" << r.field << "=" << r.lo << ":" << r.hi << ";";
  if (open_slot >= 0) os << "o" << open_slot << ";";
  if (geo.mode == GeoFilter::Mode::kDelivery) {
    os << "gd" << std::hex << geo::cell_of(geo.lat, geo.lon, schema.geo.cover_res) << std::dec << ";";
  } else if (geo.mode == GeoFilter::Mode::kRadius) {
    // Radius filters check exact distances, so key on the precise point.
    os << "gr" << geo.lat << "," << geo.lon << "," << geo.radius_km << ";";
  }
  return os.str();
}

std::vector<QueryTerm> parse_terms(const json& j) {
  std::vector<QueryTerm> out;
  if (!j.is_array()) return out;
  int auto_group = 0;
  for (const auto& t : j) {
    QueryTerm q;
    if (t.is_string()) {
      q.text = t.get<std::string>();
      q.group = auto_group++;
    } else {
      q.text = t.at("text").get<std::string>();
      q.weight = t.value("weight", 1.0f);
      q.group = t.value("group", auto_group);
      q.primary = t.value("primary", t.value("source", std::string("query")) != "synonym");
      auto_group = std::max(auto_group, q.group + 1);
    }
    if (!q.text.empty() && q.weight > 0) out.push_back(std::move(q));
  }
  return out;
}

std::vector<float> parse_field_weights(const json& j, const index::Schema& schema) {
  std::vector<float> w;
  for (auto& f : schema.text_fields) w.push_back(f.weight);
  if (j.is_object()) {
    for (auto it = j.begin(); it != j.end(); ++it) {
      int f = schema.text_field(it.key());
      if (f >= 0) w[f] = it.value().get<float>();
    }
  }
  return w;
}

static FilterSpec parse_filter(const json& j, const index::Schema& schema) {
  FilterSpec f;
  if (!j.is_object()) return f;
  const auto& kw = analysis::get_analyzer("keyword");
  auto parse_values = [&](const json& v) {
    std::vector<std::string> out;
    auto add = [&](const json& x) {
      std::string s = x.is_string() ? x.get<std::string>() : x.dump();
      auto t = kw.terms(s);
      if (!t.empty()) out.push_back(t[0]);
    };
    if (v.is_array())
      for (auto& x : v) add(x);
    else
      add(v);
    return out;
  };
  if (j.contains("terms"))
    for (auto it = j["terms"].begin(); it != j["terms"].end(); ++it) {
      int fid = schema.keyword_field(it.key());
      DC_CHECK(fid >= 0, "unknown keyword filter field: " + it.key());
      f.terms.emplace_back(fid, parse_values(it.value()));
    }
  if (j.contains("exclude"))
    for (auto it = j["exclude"].begin(); it != j["exclude"].end(); ++it) {
      int fid = schema.keyword_field(it.key());
      DC_CHECK(fid >= 0, "unknown keyword filter field: " + it.key());
      f.exclude.emplace_back(fid, parse_values(it.value()));
    }
  if (j.contains("range"))
    for (auto it = j["range"].begin(); it != j["range"].end(); ++it) {
      int fid = schema.numeric_field(it.key());
      DC_CHECK(fid >= 0, "unknown numeric filter field: " + it.key());
      FilterSpec::Range r{fid, -INFINITY, INFINITY};
      const auto& v = it.value();
      if (v.contains("gte")) r.lo = v["gte"].get<float>();
      if (v.contains("gt")) r.lo = std::nextafter(v["gt"].get<float>(), INFINITY);
      if (v.contains("lte")) r.hi = v["lte"].get<float>();
      if (v.contains("lt")) r.hi = std::nextafter(v["lt"].get<float>(), -INFINITY);
      f.ranges.push_back(r);
    }
  f.open_slot = j.value("open_slot", -1);
  if (j.contains("geo") && j["geo"].is_object()) {
    const auto& g = j["geo"];
    std::string mode = g.value("mode", std::string("none"));
    if (mode == "delivery") f.geo.mode = GeoFilter::Mode::kDelivery;
    else if (mode == "radius") f.geo.mode = GeoFilter::Mode::kRadius;
    if (f.geo.mode != GeoFilter::Mode::kNone) {
      f.geo.lat = g.at("lat").get<double>();
      f.geo.lon = g.at("lon").get<double>();
      f.geo.radius_km = g.value("radius_km", 5.0);
    }
  }
  return f;
}

SearchRequest SearchRequest::from_json(const json& j, const index::Schema& schema) {
  SearchRequest r;
  r.terms = parse_terms(j.value("terms", json::array()));
  r.field_weights = parse_field_weights(j.value("field_weights", json::object()), schema);
  r.filter = parse_filter(j.value("filter", json::object()), schema);
  r.filter_mode = filter_mode_from_name(j.value("filter_mode", std::string("filter_aware")));
  r.oversample = j.value("oversample", 10u);
  const json paths = j.value("paths", json::object());
  if (paths.contains("bm25")) {
    r.bm25 = paths["bm25"].value("enabled", true);
    r.bm25_k = paths["bm25"].value("k", 1000u);
  } else {
    r.bm25 = !paths.is_object() || paths.empty();
  }
  if (paths.contains("exact")) {
    r.exact = paths["exact"].value("enabled", true);
    r.exact_k = paths["exact"].value("k", 100u);
  }
  if (paths.contains("vector")) {
    const auto& v = paths["vector"];
    r.vector = v.value("enabled", true);
    r.vector_k = v.value("k", 500u);
    r.ef = v.value("ef", 128u);
    r.brute_force_max = v.value("brute_force_max", 20000u);
    if (v.contains("vector_b64")) r.qvec = decode_f32_base64(v["vector_b64"].get<std::string>());
    else if (v.contains("vector")) r.qvec = v["vector"].get<std::vector<float>>();
    if (r.qvec.empty()) r.vector = false;
  }
  if (paths.contains("geo")) {
    r.geo = paths["geo"].value("enabled", true);
    r.geo_k = paths["geo"].value("k", 500u);
    r.geo_tau_km = paths["geo"].value("tau_km", 3.0f);
  }
  r.use_filter_cache = j.value("filter_cache", true);
  r.use_postings_cache = j.value("postings_cache", true);
  r.debug = j.value("debug", false);
  return r;
}

FeatureRequest FeatureRequest::from_json(const json& j, const index::Schema& schema) {
  FeatureRequest r;
  r.terms = parse_terms(j.value("terms", json::array()));
  r.field_weights = parse_field_weights(j.value("field_weights", json::object()), schema);
  if (j.contains("vector_b64")) r.qvec = decode_f32_base64(j["vector_b64"].get<std::string>());
  if (j.contains("user")) {
    const auto& u = j["user"];
    if (u.contains("lat") && u["lat"].is_number()) r.user_lat = u["lat"].get<double>();
    if (u.contains("lon") && u["lon"].is_number()) r.user_lon = u["lon"].get<double>();
    r.slot = u.value("slot", -1);
  }
  for (const auto& d : j.at("docs")) r.docs.emplace_back(d.at(0).get<uint32_t>(), d.at(1).get<uint32_t>());
  r.include_stored = j.value("include_stored", true);
  return r;
}

}  // namespace dc::search
