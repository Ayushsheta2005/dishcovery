#include "index/document.h"

#include <algorithm>
#include <cstring>

#include "analysis/analyzer.h"
#include "util/base64.h"
#include "util/common.h"
#include "vector/distance.h"

namespace dc::index {

namespace {

int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return 0;
}

// Text value(s) of a key: strings are taken as-is, arrays are flattened,
// numbers stringified.
void collect_strings(const json& v, std::vector<std::string>& out) {
  if (v.is_string()) {
    const auto& s = v.get_ref<const std::string&>();
    if (!s.empty()) out.push_back(s);
  } else if (v.is_array()) {
    for (const auto& x : v) collect_strings(x, out);
  } else if (v.is_number_integer()) {
    out.push_back(std::to_string(v.get<int64_t>()));
  } else if (v.is_number()) {
    double d = v.get<double>();
    if (d == std::floor(d)) out.push_back(std::to_string(static_cast<int64_t>(d)));
    else out.push_back(std::to_string(d));
  } else if (v.is_boolean()) {
    out.push_back(v.get<bool>() ? "true" : "false");
  }
}

}  // namespace

std::array<uint8_t, kHoursBytes> parse_hours_hex(const std::string& hex) {
  std::array<uint8_t, kHoursBytes> out{};
  for (size_t k = 0; k < kHoursBytes && 2 * k + 1 < hex.size(); ++k)
    out[k] = static_cast<uint8_t>(hexval(hex[2 * k]) | (hexval(hex[2 * k + 1]) << 4));
  return out;
}

bool hours_open(const uint8_t* hours, int slot) {
  if (slot < 0) return true;
  slot %= 336;
  return (hours[slot >> 3] >> (slot & 7)) & 1;
}

Document parse_document(const json& j, const Schema& schema, const float* vector) {
  Document d;
  d.id = j.at("id").get<std::string>();
  d.restaurant_id = j.value("restaurant_id", d.id);
  d.type_code = j.value("type", std::string("restaurant")) == "dish" ? kTypeDish : kTypeRestaurant;

  d.text.resize(schema.text_fields.size());
  for (size_t f = 0; f < schema.text_fields.size(); ++f) {
    auto it = j.find(schema.text_fields[f].source);
    if (it != j.end()) collect_strings(*it, d.text[f]);
  }

  const auto& kw = analysis::get_analyzer("keyword");
  d.keywords.resize(schema.keyword_fields.size());
  for (size_t f = 0; f < schema.keyword_fields.size(); ++f) {
    auto it = j.find(schema.keyword_fields[f].source);
    if (it == j.end()) continue;
    std::vector<std::string> raw;
    collect_strings(*it, raw);
    for (auto& r : raw) {
      auto t = kw.terms(r);
      if (!t.empty()) d.keywords[f].push_back(std::move(t[0]));
    }
    std::sort(d.keywords[f].begin(), d.keywords[f].end());
    d.keywords[f].erase(std::unique(d.keywords[f].begin(), d.keywords[f].end()), d.keywords[f].end());
  }

  d.numerics.resize(schema.numeric_fields.size());
  for (size_t f = 0; f < schema.numeric_fields.size(); ++f) {
    auto it = j.find(schema.numeric_fields[f].source);
    d.numerics[f] = (it != j.end() && it->is_number()) ? it->get<float>() : schema.numeric_fields[f].default_value;
  }

  if (j.contains("lat") && j["lat"].is_number() && j.contains("lon") && j["lon"].is_number()) {
    d.lat = j["lat"].get<float>();
    d.lon = j["lon"].get<float>();
  }
  if (auto it = j.find("hours"); it != j.end() && it->is_string()) {
    d.hours = parse_hours_hex(it->get<std::string>());
  } else {
    d.hours.fill(0xFF);  // unknown hours: treat as always open
  }
  if (auto it = j.find("meal_affinity"); it != j.end() && it->is_array()) {
    for (size_t i = 0; i < kMealBuckets && i < it->size(); ++i) {
      double v = (*it)[i].is_number() ? (*it)[i].get<double>() : 0.0;
      d.meal[i] = static_cast<uint8_t>(std::clamp(v, 0.0, 1.0) * 255.0 + 0.5);
    }
  }
  if (auto it = j.find("delivery_polygon"); it != j.end() && it->is_array()) {
    for (const auto& p : *it) {
      if (p.is_array() && p.size() >= 2) {
        d.polygon.push_back(p[0].get<float>());
        d.polygon.push_back(p[1].get<float>());
      }
    }
  }

  if (schema.vector_dim > 0) {
    if (vector) {
      d.vec.assign(vector, vector + schema.vector_dim);
    } else if (auto it = j.find("embedding_b64"); it != j.end() && it->is_string()) {
      d.vec = decode_f32_base64(it->get<std::string>());
    } else if (auto it2 = j.find("embedding"); it2 != j.end() && it2->is_array()) {
      d.vec = it2->get<std::vector<float>>();
    }
    if (!d.vec.empty()) {
      DC_CHECK(d.vec.size() == schema.vector_dim,
               "embedding dim " + std::to_string(d.vec.size()) + " != schema " + std::to_string(schema.vector_dim));
      vec::normalize(d.vec.data(), d.vec.size());
    }
  }

  json stored = json::object();
  for (const auto& k : schema.stored_fields) {
    auto it = j.find(k);
    if (it != j.end()) stored[k] = *it;
  }
  d.stored = stored.dump();
  return d;
}

}  // namespace dc::index
