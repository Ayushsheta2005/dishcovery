#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace dc {

std::string base64_encode(const uint8_t* data, size_t n);
std::vector<uint8_t> base64_decode(std::string_view s);

// Little-endian float32 arrays travel between broker and shards as base64.
std::vector<float> decode_f32_base64(std::string_view s);
std::string encode_f32_base64(const float* v, size_t n);

}  // namespace dc
