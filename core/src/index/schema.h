// Index schema: which document fields are analysed for full-text search, which
// become filter bitmaps, which are columnar doc values, and how BM25 weighs
// each field. Loaded from configs/schema.json.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "util/json.h"

namespace dc::index {

struct TextFieldDef {
  std::string name;
  std::string source;     // document key to read (defaults to name)
  std::string analyzer;   // see analysis/analyzer.h
  float weight = 1.0f;    // BM25F-style field weight used by the retrieval scorer
  bool positions = false; // store positions (phrase / proximity features)
};

struct KeywordFieldDef {
  std::string name;
  std::string source;
};

struct NumericFieldDef {
  std::string name;
  std::string source;
  float default_value = 0.0f;
};

struct GeoConfig {
  int cover_res = 7;   // H3 resolution for delivery-zone covers (compacted)
  int point_res = 8;   // H3 resolution of the restaurant-location index (radius queries)
};

struct Bm25Params {
  float k1 = 1.2f;
  float b = 0.75f;
};

// Reserved filter-dictionary field ids for geo keys (keyword fields use 0..N-1).
inline constexpr uint8_t kGeoCoverField = 250;
inline constexpr uint8_t kGeoPointField = 251;

struct Schema {
  std::vector<TextFieldDef> text_fields;
  std::vector<KeywordFieldDef> keyword_fields;
  std::vector<NumericFieldDef> numeric_fields;
  std::vector<std::string> stored_fields;
  uint32_t vector_dim = 0;
  GeoConfig geo;
  Bm25Params bm25;

  static Schema from_json(const json& j);
  static Schema load(const std::string& path);
  static Schema default_schema();
  json to_json() const;

  int text_field(std::string_view name) const;     // -1 if absent
  int keyword_field(std::string_view name) const;
  int numeric_field(std::string_view name) const;
  uint64_t fingerprint() const;                     // detects schema/index mismatch
};

}  // namespace dc::index
