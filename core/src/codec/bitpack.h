// Fixed-width bit packing of uint32 arrays: the building block of the PFor
// codec and of any other structure that stores small integers densely.
//
// Byte layout of bitpack(in, n, bits, out) — always exactly
// bitpack_bytes(n, bits) = ceil(n * bits / 8) bytes, little-endian:
//
//  * n == kBlockSize (128): "vertical" 4-lane layout (SIMD-BP128 style).
//    Value i belongs to lane i % 4 at position i / 4. Each lane packs its 32
//    values LSB-first into `bits` 32-bit words, and the words of the four
//    lanes are interleaved: word k of lane l is the (4k + l)-th u32 of the
//    output. Every lane therefore uses the same shift amounts, so the
//    unpacker is a plain loop the compiler turns into 128-bit vector shifts
//    (SSE2 on x86-64, NEON on arm64). 128 * bits / 8 = 16 * bits bytes, so the
//    unpacker can use whole-word loads without ever leaving the block.
//
//  * any other n: "horizontal" LSB-first bit stream. Value i occupies bits
//    [i*bits, (i+1)*bits) of the byte stream (bit 0 = LSB of byte 0).
//    Groups of 32 values (= 4*bits bytes) are unpacked with unrolled
//    word-sized kernels; the < 32 value tail is read byte by byte.
//
// The two layouts differ, so a buffer packed with a given n must be unpacked
// with the same n (the posting format always knows n). Only the low `bits`
// bits of each input value are stored; higher bits are silently dropped.
//
// bitunpack never reads beyond the bitpack_bytes(n, bits) bytes of its input
// and never writes beyond out[n - 1].
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>

namespace dc::codec {

// Number of integers in a full posting block.
inline constexpr size_t kBlockSize = 128;

// Minimum number of bits that can represent max_value (0 for 0, 32 for
// values >= 2^31).
inline unsigned bits_needed(uint32_t max_value) {
  return static_cast<unsigned>(std::bit_width(max_value));
}

// Exact packed size in bytes: ceil(n * bits / 8).
inline constexpr size_t bitpack_bytes(size_t n, unsigned bits) { return (n * bits + 7) / 8; }

// Pack the low `bits` bits (0..32) of in[0..n) into out, which must have room
// for bitpack_bytes(n, bits) bytes. Every one of those bytes is written.
void bitpack(const uint32_t* in, size_t n, unsigned bits, uint8_t* out);

// Inverse of bitpack. Reads exactly bitpack_bytes(n, bits) bytes.
void bitunpack(const uint8_t* in, size_t n, unsigned bits, uint32_t* out);

}  // namespace dc::codec
