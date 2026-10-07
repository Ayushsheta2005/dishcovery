#include "index/schema.h"

#include "analysis/analyzer.h"
#include "util/common.h"
#include "util/fs.h"
#include "util/hash.h"

namespace dc::index {

Schema Schema::from_json(const json& j) {
  Schema s;
  for (const auto& f : j.at("text_fields")) {
    TextFieldDef d;
    d.name = f.at("name").get<std::string>();
    d.source = f.value("source", d.name);
    d.analyzer = f.value("analyzer", std::string("standard"));
    d.weight = f.value("weight", 1.0f);
    d.positions = f.value("positions", false);
    analysis::get_analyzer(d.analyzer);  // validate
    s.text_fields.push_back(std::move(d));
  }
  for (const auto& f : j.value("keyword_fields", json::array())) {
    KeywordFieldDef d;
    d.name = f.at("name").get<std::string>();
    d.source = f.value("source", d.name);
    s.keyword_fields.push_back(std::move(d));
  }
  for (const auto& f : j.value("numeric_fields", json::array())) {
    NumericFieldDef d;
    d.name = f.at("name").get<std::string>();
    d.source = f.value("source", d.name);
    d.default_value = f.value("default", 0.0f);
    s.numeric_fields.push_back(std::move(d));
  }
  for (const auto& f : j.value("stored_fields", json::array())) s.stored_fields.push_back(f.get<std::string>());
  s.vector_dim = j.value("vector_dim", 0u);
  if (j.contains("geo")) {
    s.geo.cover_res = j["geo"].value("cover_res", 7);
    s.geo.point_res = j["geo"].value("point_res", 8);
  }
  if (j.contains("bm25")) {
    s.bm25.k1 = j["bm25"].value("k1", 1.2f);
    s.bm25.b = j["bm25"].value("b", 0.75f);
  }
  DC_CHECK(s.text_fields.size() < 64, "too many text fields");
  DC_CHECK(s.keyword_fields.size() < 200, "too many keyword fields");
  return s;
}

Schema Schema::load(const std::string& path) { return from_json(json::parse(read_file(path))); }

Schema Schema::default_schema() {
  return from_json(json::parse(R"({
    "text_fields": [
      {"name": "name", "analyzer": "minimal", "weight": 3.0, "positions": true},
      {"name": "name_exact", "source": "name", "analyzer": "plain", "weight": 0.0, "positions": true},
      {"name": "restaurant_name", "analyzer": "minimal", "weight": 1.5, "positions": true},
      {"name": "cuisines", "analyzer": "standard", "weight": 2.0},
      {"name": "categories", "analyzer": "standard", "weight": 1.0},
      {"name": "tags", "analyzer": "standard", "weight": 1.2},
      {"name": "description", "analyzer": "standard", "weight": 0.8, "positions": true},
      {"name": "reviews", "analyzer": "standard", "weight": 0.4}
    ],
    "keyword_fields": [
      {"name": "type"}, {"name": "diet"}, {"name": "price_tier"},
      {"name": "cuisine", "source": "cuisines"}, {"name": "chain"}, {"name": "city"},
      {"name": "canonical_dish"}
    ],
    "numeric_fields": [
      {"name": "rating"}, {"name": "review_count"}, {"name": "price"}, {"name": "price_tier"},
      {"name": "popularity"}, {"name": "prep_time_min", "default": 20},
      {"name": "delivery_radius_km", "default": 5}
    ],
    "stored_fields": ["id", "type", "name", "restaurant_name", "restaurant_id", "chain", "canonical_dish",
                      "cuisines", "diet", "price", "price_tier", "rating", "review_count", "city", "lat", "lon"],
    "vector_dim": 384,
    "geo": {"cover_res": 7, "point_res": 8},
    "bm25": {"k1": 1.2, "b": 0.75}
  })"));
}

json Schema::to_json() const {
  json j;
  j["text_fields"] = json::array();
  for (auto& f : text_fields)
    j["text_fields"].push_back(
        {{"name", f.name}, {"source", f.source}, {"analyzer", f.analyzer}, {"weight", f.weight}, {"positions", f.positions}});
  j["keyword_fields"] = json::array();
  for (auto& f : keyword_fields) j["keyword_fields"].push_back({{"name", f.name}, {"source", f.source}});
  j["numeric_fields"] = json::array();
  for (auto& f : numeric_fields)
    j["numeric_fields"].push_back({{"name", f.name}, {"source", f.source}, {"default", f.default_value}});
  j["stored_fields"] = stored_fields;
  j["vector_dim"] = vector_dim;
  j["geo"] = {{"cover_res", geo.cover_res}, {"point_res", geo.point_res}};
  j["bm25"] = {{"k1", bm25.k1}, {"b", bm25.b}};
  return j;
}

int Schema::text_field(std::string_view name) const {
  for (size_t i = 0; i < text_fields.size(); ++i)
    if (text_fields[i].name == name) return static_cast<int>(i);
  return -1;
}
int Schema::keyword_field(std::string_view name) const {
  for (size_t i = 0; i < keyword_fields.size(); ++i)
    if (keyword_fields[i].name == name) return static_cast<int>(i);
  return -1;
}
int Schema::numeric_field(std::string_view name) const {
  for (size_t i = 0; i < numeric_fields.size(); ++i)
    if (numeric_fields[i].name == name) return static_cast<int>(i);
  return -1;
}

uint64_t Schema::fingerprint() const {
  // Only structural properties: changing a weight must not invalidate an index.
  std::string sig;
  for (auto& f : text_fields) sig += f.name + "|" + f.source + "|" + f.analyzer + "|" + (f.positions ? "p" : "") + ";";
  for (auto& f : keyword_fields) sig += f.name + "|" + f.source + ";";
  for (auto& f : numeric_fields) sig += f.name + "|" + f.source + ";";
  sig += std::to_string(vector_dim) + ";" + std::to_string(geo.cover_res) + ";" + std::to_string(geo.point_res);
  return fnv1a64(sig);
}

}  // namespace dc::index
