// PForDelta block codec, OptPFD flavour.
//
// A block of n (<= 128) values is stored as n fixed-width b-bit slots holding
// the low b bits of every value; values >= 2^b are "exceptions" whose high
// parts (value >> b) are stored separately and OR-ed back after unpacking.
// The encoder picks b (and the exception encoding) that minimise the exact
// encoded size of the block.
//
// Block layout:
//
//   u8   header   bits 0..5  b, the slot width (0..32)
//                 bit  6     1 = block has exceptions (count byte follows)
//                 bit  7     high parts are vbyte-coded (else bit-packed)
//   [u8  e]       exception count, 1..n          (only if bit 6 is set)
//   ceil(n*b/8)   low b bits of all n values, bitpack(n, b) layout
//   --- only if e > 0 ---
//   e x u8        exception positions (indexes 0..n-1, ascending)
//   high parts, one of:
//     bit-packed:  u8 hb (= max exception width - b), then bitpack(e, hb)
//     vbyte:       e vbyte-coded values
//
// Exception high parts: neither encoding dominates. Bit packing is better
// when there are many exceptions of similar magnitude; vbyte is better for a
// handful of outliers (no width byte, and a moderate outlier does not pay for
// the largest one). Both sizes are computed exactly for every candidate b and
// the smaller one is used, so the choice is per block.
//
// OptPFD selection in O(n + 33): one pass builds a histogram of bit widths;
// above[t] = #values wider than t bits is a suffix sum. For a candidate b:
//   exceptions        e        = above[b]
//   bit-packed highs  1 + ceil(e * (maxw - b) / 8)
//   vbyte highs       sum_{k>=0} above[b + 7k]
// (a high part of w - b bits needs ceil((w - b) / 7) vbyte bytes, i.e. one
// byte for every k with b + 7k < w). Ties prefer the larger b, which means
// fewer exceptions to patch at decode time.
//
// Exception-free blocks cost one header byte plus the packed slots, so PFor is
// never more than 1 byte larger than plain bit packing at the block's max width.
#pragma once

#include <cstddef>
#include <cstdint>

#include "util/bytes.h"

namespace dc::codec {

// Append the PFor encoding of in[0..n) to out. 0 <= n <= kBlockSize.
void pfor_encode(const uint32_t* in, size_t n, ByteWriter& out);

// Decode a block written by pfor_encode with the same n; returns the pointer
// one past its last byte. Throws DcError on structurally corrupt blocks.
const uint8_t* pfor_decode(const uint8_t* in, size_t n, uint32_t* out);

}  // namespace dc::codec
