#include "util/base64.h"

#include <cstring>

#include "util/common.h"

namespace dc {

static const char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(const uint8_t* data, size_t n) {
  std::string out;
  out.reserve((n + 2) / 3 * 4);
  size_t i = 0;
  for (; i + 2 < n; i += 3) {
    uint32_t v = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) | data[i + 2];
    out.push_back(kAlphabet[(v >> 18) & 63]);
    out.push_back(kAlphabet[(v >> 12) & 63]);
    out.push_back(kAlphabet[(v >> 6) & 63]);
    out.push_back(kAlphabet[v & 63]);
  }
  if (i < n) {
    uint32_t v = uint32_t(data[i]) << 16;
    if (i + 1 < n) v |= uint32_t(data[i + 1]) << 8;
    out.push_back(kAlphabet[(v >> 18) & 63]);
    out.push_back(kAlphabet[(v >> 12) & 63]);
    out.push_back(i + 1 < n ? kAlphabet[(v >> 6) & 63] : '=');
    out.push_back('=');
  }
  return out;
}

std::vector<uint8_t> base64_decode(std::string_view s) {
  static int8_t table[256];
  static bool init = [] {
    std::memset(table, -1, sizeof table);
    for (int i = 0; i < 64; ++i) table[static_cast<uint8_t>(kAlphabet[i])] = static_cast<int8_t>(i);
    table[static_cast<uint8_t>('-')] = 62;  // accept url-safe alphabet too
    table[static_cast<uint8_t>('_')] = 63;
    return true;
  }();
  (void)init;
  std::vector<uint8_t> out;
  out.reserve(s.size() * 3 / 4);
  uint32_t acc = 0;
  int bits = 0;
  for (char c : s) {
    if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
    int8_t v = table[static_cast<uint8_t>(c)];
    if (v < 0) fail("base64: invalid character");
    acc = (acc << 6) | static_cast<uint32_t>(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<uint8_t>((acc >> bits) & 0xFF));
    }
  }
  return out;
}

std::vector<float> decode_f32_base64(std::string_view s) {
  auto bytes = base64_decode(s);
  if (bytes.size() % 4) fail("base64 float array: length not a multiple of 4");
  std::vector<float> v(bytes.size() / 4);
  std::memcpy(v.data(), bytes.data(), bytes.size());
  return v;
}

std::string encode_f32_base64(const float* v, size_t n) {
  return base64_encode(reinterpret_cast<const uint8_t*>(v), n * sizeof(float));
}

}  // namespace dc
