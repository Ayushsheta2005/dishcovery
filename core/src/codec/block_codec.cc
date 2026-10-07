#include "codec/block_codec.h"

#include <bit>
#include <cstring>
#include <string>

#include "util/common.h"

namespace dc::codec {

static_assert(std::endian::native == std::endian::little,
              "raw blocks are memcpy'd u32s; a big-endian port needs byte swaps");

const char* codec_name(CodecId id) {
  switch (id) {
    case CodecId::kRaw:
      return "raw";
    case CodecId::kVByte:
      return "vbyte";
    case CodecId::kPFor:
      return "pfor";
  }
  return "unknown";
}

CodecId codec_from_name(std::string_view name) {
  if (name == "raw") return CodecId::kRaw;
  if (name == "vbyte") return CodecId::kVByte;
  if (name == "pfor") return CodecId::kPFor;
  fail("unknown block codec: '" + std::string(name) + "' (expected raw, vbyte or pfor)");
}

void encode_block(CodecId id, const uint32_t* in, size_t n, ByteWriter& out) {
  switch (id) {
    case CodecId::kRaw:
      out.put_bytes(in, n * sizeof(uint32_t));
      return;
    case CodecId::kVByte:
      vbyte_encode(in, n, out);
      return;
    case CodecId::kPFor:
      pfor_encode(in, n, out);
      return;
  }
  fail("encode_block: invalid codec id " + std::to_string(static_cast<int>(id)));
}

const uint8_t* decode_block(CodecId id, const uint8_t* in, size_t n, uint32_t* out) {
  switch (id) {
    case CodecId::kRaw:
      std::memcpy(out, in, n * sizeof(uint32_t));
      return in + n * sizeof(uint32_t);
    case CodecId::kVByte:
      return vbyte_decode(in, n, out);
    case CodecId::kPFor:
      return pfor_decode(in, n, out);
  }
  fail("decode_block: invalid codec id " + std::to_string(static_cast<int>(id)));
}

void delta_encode(uint32_t* vals, size_t n, uint32_t base) {
  // Walk backwards so each element still sees its original predecessor;
  // the loop body has no carried dependency and vectorizes.
  for (size_t i = n; i-- > 1;) vals[i] -= vals[i - 1];
  if (n) vals[0] -= base;
}

void prefix_sum(uint32_t* vals, size_t n, uint32_t base) {
  uint32_t acc = base;
  for (size_t i = 0; i < n; ++i) {
    acc += vals[i];
    vals[i] = acc;
  }
}

}  // namespace dc::codec
