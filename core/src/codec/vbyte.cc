// Variable-byte coding. The decoder works on 8-byte windows:
//
//  * all 8 bytes are terminators (8 one-byte values): emit them directly.
//    This branch is well predicted for small gaps and term frequencies.
//  * otherwise: every complete value in the window is decoded branch-free.
//    The 8 continuation bits index a 256-entry table giving each value's bit
//    offset and length in the window with the continuation bits squeezed
//    out; all 8 output slots are written unconditionally (slots past the
//    window's value count hold scratch that later iterations overwrite), so
//    mixed value lengths cost no branch mispredictions.
//
// Both paths run only while >= 8 values remain: then >= 8 bytes remain in the
// block (each value is >= 1 byte) and >= 8 output slots remain. The last < 8
// values are decoded one byte at a time.
#include "codec/vbyte.h"

#include <array>
#include <bit>

#include "util/common.h"

namespace dc::codec {

namespace {

constexpr uint64_t kHighBits = 0x8080808080808080ull;
constexpr unsigned kMaxLen = 5;  // bytes in the longest u32 encoding

// Layout of the complete values inside an 8-byte window, for one pattern of
// continuation bits (bit k of the pattern = high bit of byte k). Offsets and
// lengths are in bits of the *compacted* window (7 data bits per byte).
struct Window {
  uint8_t count;       // complete values in the window
  bool valid;          // false: a value longer than 5 bytes (corrupt input)
  uint8_t shift[8];    // 7 * first byte of value s
  uint8_t len_bits[8]; // 7 * byte length of value s
};

constexpr std::array<Window, 256> make_windows() {
  std::array<Window, 256> table{};
  for (unsigned m = 0; m < 256; ++m) {
    Window w{};
    w.valid = true;
    for (unsigned s = 0; s < 8; ++s) {
      w.shift[s] = 0;  // unused slot: re-decodes group 0, overwritten later
      w.len_bits[s] = 7;
    }
    unsigned start = 0;
    for (unsigned k = 0; k < 8; ++k) {
      if (m & (1u << k)) continue;  // byte k continues a value
      const unsigned len = k - start + 1;
      if (len > kMaxLen) w.valid = false;
      w.shift[w.count] = static_cast<uint8_t>(7 * start);
      w.len_bits[w.count] = static_cast<uint8_t>(7 * len);
      ++w.count;
      start = k + 1;
    }
    // The first value must end within 5 bytes (this also rejects m == 0xFF).
    if (w.count == 0 || w.len_bits[0] > 7 * kMaxLen) w.valid = false;
    table[m] = w;
  }
  return table;
}

constexpr std::array<Window, 256> kWindows = make_windows();

// Gather the high bit of every byte of w into one byte (bit k <- byte k).
// Each high bit is multiplied into a distinct position, so no carries occur.
inline unsigned continuation_mask(uint64_t w) {
  return static_cast<unsigned>(((w & kHighBits) * 0x0002040810204081ull) >> 56);
}

// Drop the continuation bits: the 8 7-bit groups of w become 56 contiguous
// bits, so every value of the window is a plain bit field.
inline uint64_t compact_window(uint64_t w) {
  uint64_t x = w & 0x7F7F7F7F7F7F7F7Full;
  x = (x & 0x007F007F007F007Full) | ((x & 0x7F007F007F007F00ull) >> 1);
  x = (x & 0x00003FFF00003FFFull) | ((x & 0x3FFF00003FFF0000ull) >> 2);
  x = (x & 0x000000000FFFFFFFull) | ((x & 0x0FFFFFFF00000000ull) >> 4);
  return x;
}

[[noreturn]] void corrupt() { fail("vbyte: value longer than 5 bytes (corrupt block?)"); }

// Byte-at-a-time decode of one value; touches only the value's own bytes.
inline uint32_t decode_one_bytewise(const uint8_t*& p) {
  uint32_t b = *p++;
  uint32_t v = b & 0x7F;
  if (b < 0x80) return v;
  b = *p++;
  v |= (b & 0x7F) << 7;
  if (b < 0x80) return v;
  b = *p++;
  v |= (b & 0x7F) << 14;
  if (b < 0x80) return v;
  b = *p++;
  v |= (b & 0x7F) << 21;
  if (b < 0x80) return v;
  b = *p++;
  if (DC_UNLIKELY(b >= 0x80)) corrupt();
  return v | (b << 28);
}

}  // namespace

uint8_t* vbyte_encode_to(const uint32_t* in, size_t n, uint8_t* out) {
  for (size_t i = 0; i < n; ++i) {
    uint32_t v = in[i];
    while (v >= 0x80) {
      *out++ = static_cast<uint8_t>(v | 0x80);
      v >>= 7;
    }
    *out++ = static_cast<uint8_t>(v);
  }
  return out;
}

void vbyte_encode(const uint32_t* in, size_t n, ByteWriter& out) {
  const size_t start = out.size();
  uint8_t* begin = out.extend(5 * n);  // worst case, trimmed below
  uint8_t* end = vbyte_encode_to(in, n, begin);
  out.truncate(start + static_cast<size_t>(end - begin));
}

const uint8_t* vbyte_decode(const uint8_t* in, size_t n, uint32_t* out) {
  const uint8_t* p = in;
  size_t i = 0;
  while (n - i >= 8) {
    const uint64_t w = load_u64(p);
    const uint64_t stops = ~w & kHighBits;  // terminator bytes
    if (stops == kHighBits) {
      for (unsigned k = 0; k < 8; ++k) out[i + k] = static_cast<uint8_t>(w >> (8 * k));
      i += 8;
      p += 8;
      continue;
    }
    const Window& win = kWindows[continuation_mask(w)];
    if (DC_UNLIKELY(!win.valid)) corrupt();
    const uint64_t x = compact_window(w);
    for (unsigned s = 0; s < 8; ++s)
      out[i + s] = static_cast<uint32_t>((x >> win.shift[s]) & ((uint64_t{1} << win.len_bits[s]) - 1));
    i += win.count;
    // Advance past the last terminator. Computed from w rather than the table
    // to keep the loop-carried p -> load -> p chain short.
    p += static_cast<unsigned>(64 - std::countl_zero(stops)) >> 3;
  }
  for (; i < n; ++i) out[i] = decode_one_bytewise(p);
  return p;
}

const uint8_t* vbyte_skip(const uint8_t* in, size_t n) {
  const uint8_t* p = in;
  size_t left = n;  // terminator bytes still to pass; >= bytes needed, so
                    // an 8-byte load is in bounds while left >= 8.
  while (left >= 8) {
    left -= static_cast<size_t>(std::popcount(~load_u64(p) & kHighBits));
    p += 8;
  }
  while (left) {
    if (*p++ < 0x80) --left;
  }
  return p;
}

}  // namespace dc::codec
