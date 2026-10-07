// JSON: nlohmann for parsing requests/configs, plus a small append-only writer
// for large responses (thousands of candidates) where building a DOM is too
// slow.
#pragma once

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace dc {

using json = nlohmann::json;

class JsonWriter {
 public:
  explicit JsonWriter(size_t reserve = 4096) { s_.reserve(reserve); }

  JsonWriter& begin_object() { return sep().raw('{'), push(); }
  JsonWriter& end_object() { return pop().raw('}'); }
  JsonWriter& begin_array() { return sep().raw('['), push(); }
  JsonWriter& end_array() { return pop().raw(']'); }

  JsonWriter& key(std::string_view k) {
    sep();
    write_string(k);
    s_.push_back(':');
    after_key_ = true;
    return *this;
  }

  JsonWriter& value(std::string_view v) { return sep().write_string(v); }
  JsonWriter& value(const char* v) { return value(std::string_view(v)); }
  JsonWriter& value(bool v) { return sep().raw_str(v ? "true" : "false"); }
  JsonWriter& value(int64_t v) { return sep().num(v); }
  JsonWriter& value(uint64_t v) { return sep().num(v); }
  JsonWriter& value(int v) { return sep().num(static_cast<int64_t>(v)); }
  JsonWriter& value(unsigned v) { return sep().num(static_cast<uint64_t>(v)); }
  JsonWriter& value(double v) {
    sep();
    if (!std::isfinite(v)) return raw_str("null");
    char buf[32];
    auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::general, 7);
    s_.append(buf, r.ptr);
    return *this;
  }
  JsonWriter& value(float v) { return value(static_cast<double>(v)); }
  JsonWriter& null() { return sep().raw_str("null"); }
  // Insert pre-serialised JSON (e.g. a stored document) verbatim.
  JsonWriter& raw_json(std::string_view j) {
    sep();
    s_.append(j.data(), j.size());
    return *this;
  }

  template <class T>
  JsonWriter& kv(std::string_view k, const T& v) {
    key(k);
    return value(v);
  }

  const std::string& str() const { return s_; }
  std::string take() { return std::move(s_); }

 private:
  JsonWriter& raw(char c) {
    s_.push_back(c);
    return *this;
  }
  JsonWriter& raw_str(std::string_view v) {
    s_.append(v.data(), v.size());
    return *this;
  }
  template <class I>
  JsonWriter& num(I v) {
    char buf[24];
    auto r = std::to_chars(buf, buf + sizeof buf, v);
    s_.append(buf, r.ptr);
    return *this;
  }
  JsonWriter& push() {
    first_ = true;
    ++depth_;
    return *this;
  }
  JsonWriter& pop() {
    --depth_;
    first_ = false;
    after_key_ = false;
    return *this;
  }
  // Emit a comma unless this is the first element of the container or follows a key.
  JsonWriter& sep() {
    if (after_key_) {
      after_key_ = false;
      return *this;
    }
    if (!first_) s_.push_back(',');
    first_ = false;
    return *this;
  }
  JsonWriter& write_string(std::string_view v) {
    s_.push_back('"');
    for (char c : v) {
      switch (c) {
        case '"': s_ += "\\\""; break;
        case '\\': s_ += "\\\\"; break;
        case '\n': s_ += "\\n"; break;
        case '\r': s_ += "\\r"; break;
        case '\t': s_ += "\\t"; break;
        default:
          if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            s_ += buf;
          } else {
            s_.push_back(c);
          }
      }
    }
    s_.push_back('"');
    return *this;
  }

  std::string s_;
  bool first_ = true;
  bool after_key_ = false;
  int depth_ = 0;
};

}  // namespace dc
