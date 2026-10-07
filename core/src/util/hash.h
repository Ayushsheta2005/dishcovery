#pragma once

#include <cstdint>
#include <string_view>

namespace dc {

// FNV-1a 64-bit. Used for shard routing and MUST stay identical to
// dishcovery/hashing.py::fnv1a64.
inline uint64_t fnv1a64(std::string_view s) {
  uint64_t h = 0xCBF29CE484222325ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 0x100000001B3ull;
  }
  return h;
}

inline uint32_t shard_of(std::string_view restaurant_id, uint32_t num_shards) {
  return static_cast<uint32_t>(fnv1a64(restaurant_id) % num_shards);
}

// Finalizer from splitmix64; good avalanche for integer keys.
inline uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

inline uint64_t hash_combine(uint64_t a, uint64_t b) { return mix64(a ^ (b + 0x9E3779B97F4A7C15ull + (a << 6) + (a >> 2))); }

}  // namespace dc
