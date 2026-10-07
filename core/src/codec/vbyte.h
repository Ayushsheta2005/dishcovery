// Variable-byte integer coding (LEB128 order): each uint32 is split into
// 7-bit groups, least-significant group first; every byte carries one group
// in its low 7 bits and sets the high bit iff more bytes follow. A value takes
// 1..5 bytes (1 byte for < 2^7, 2 for < 2^14, ... 5 for >= 2^28). This is the
// same encoding as ByteWriter::put_varint for 32-bit inputs.
//
// The decoders know only n, not the byte length, and they never read past the
// last byte of the n-th value: 8-byte loads are used only while at least 8
// values remain, which guarantees at least 8 bytes remain.
#pragma once

#include <cstddef>
#include <cstdint>

#include "util/bytes.h"

namespace dc::codec {

// Encoded size of one value in bytes (1..5).
inline constexpr size_t vbyte_size(uint32_t v) {
  return 1 + (v >= (1u << 7)) + (v >= (1u << 14)) + (v >= (1u << 21)) + (v >= (1u << 28));
}

// Append the encoding of in[0..n) to out.
void vbyte_encode(const uint32_t* in, size_t n, ByteWriter& out);

// Encode into a raw buffer with room for at least 5 * n bytes; returns the
// pointer one past the last byte written.
uint8_t* vbyte_encode_to(const uint32_t* in, size_t n, uint8_t* out);

// Decode n values; returns the pointer one past the last byte of the n-th
// value. Throws DcError on a value longer than 5 bytes (corrupt input).
const uint8_t* vbyte_decode(const uint8_t* in, size_t n, uint32_t* out);

// Skip n encoded values without materialising them (counts terminator bytes).
// Returns the same pointer vbyte_decode would.
const uint8_t* vbyte_skip(const uint8_t* in, size_t n);

}  // namespace dc::codec
