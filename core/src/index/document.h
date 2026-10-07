// A document parsed from the docs.jsonl contract (dishcovery/data/schema.py)
// into the shape the segment builder consumes.
#pragma once

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "index/schema.h"
#include "util/json.h"

namespace dc::index {

inline constexpr size_t kHoursBytes = 42;  // 336 half-hour slots, Monday 00:00 = bit 0
inline constexpr size_t kMealBuckets = 5;  // breakfast, lunch, snack, dinner, late_night
inline constexpr int kTypeRestaurant = 0;
inline constexpr int kTypeDish = 1;

struct Document {
  std::string id;
  std::string restaurant_id;
  uint8_t type_code = kTypeRestaurant;
  std::vector<std::vector<std::string>> text;      // [text field][value]
  std::vector<std::vector<std::string>> keywords;  // [keyword field][normalised value]
  std::vector<float> numerics;                     // [numeric field]
  float lat = NAN, lon = NAN;
  std::array<uint8_t, kHoursBytes> hours{};
  std::array<uint8_t, kMealBuckets> meal{};        // affinity quantised to 0..255
  std::vector<float> polygon;                      // lat0, lon0, lat1, lon1, ...
  std::vector<float> vec;                          // empty or schema.vector_dim floats
  std::string stored;                              // compact JSON of stored fields
};

// Parse one docs.jsonl object. If `vector` is non-null it overrides any
// embedding inside the JSON ("embedding" array or "embedding_b64").
Document parse_document(const json& j, const Schema& schema, const float* vector = nullptr);

bool hours_open(const uint8_t* hours, int slot);
std::array<uint8_t, kHoursBytes> parse_hours_hex(const std::string& hex);

}  // namespace dc::index
