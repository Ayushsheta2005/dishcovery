// Interchangeable posting-block codecs plus delta helpers.
//
// A posting list is a sequence of blocks of up to kBlockSize uint32 values
// (docID deltas, term frequencies or position deltas). Each block is encoded
// with one of the codecs below and blocks are simply concatenated; the reader
// tracks n per block (it is not stored by the codecs) and every decoder
// returns the pointer just past the block it consumed. Decoders never read a
// byte beyond the end of their block, so the last block may sit flush against
// the end of an mmap'd file.
//
//   raw    n little-endian u32 (4n bytes)
//   vbyte  see vbyte.h
//   pfor   see pfor.h (OptPFD)
//
// Including this header pulls in the whole codec API.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "codec/bitpack.h"
#include "codec/pfor.h"
#include "codec/vbyte.h"
#include "util/bytes.h"

namespace dc::codec {

enum class CodecId : uint8_t { kRaw = 0, kVByte = 1, kPFor = 2 };

// "raw" | "vbyte" | "pfor" ("unknown" for an out-of-range id).
const char* codec_name(CodecId id);

// Inverse of codec_name; throws DcError on an unknown name.
CodecId codec_from_name(std::string_view name);

// Append the encoding of in[0..n) to out. n <= kBlockSize for kPFor.
void encode_block(CodecId id, const uint32_t* in, size_t n, ByteWriter& out);

// Decode n values; returns the pointer one past the block.
const uint8_t* decode_block(CodecId id, const uint8_t* in, size_t n, uint32_t* out);

// In-place gap encoding: vals[i] -= (i ? original vals[i-1] : base).
// Intended for non-decreasing input; arithmetic wraps mod 2^32 so any input
// round-trips through prefix_sum.
void delta_encode(uint32_t* vals, size_t n, uint32_t base);

// Inverse of delta_encode: vals[i] += (i ? new vals[i-1] : base).
void prefix_sum(uint32_t* vals, size_t n, uint32_t base);

}  // namespace dc::codec
