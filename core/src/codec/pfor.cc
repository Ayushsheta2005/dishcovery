#include "codec/pfor.h"

#include <bit>

#include "codec/bitpack.h"
#include "codec/vbyte.h"
#include "util/common.h"

namespace dc::codec {

namespace {

constexpr uint8_t kWidthMask = 0x3F;
constexpr uint8_t kHasExceptions = 0x40;
constexpr uint8_t kVByteHighs = 0x80;

struct Plan {
  unsigned b = 0;           // slot width
  unsigned e = 0;           // exception count
  unsigned high_bits = 0;   // width of bit-packed high parts
  bool vbyte_highs = false;
  size_t bytes = 0;         // exact encoded size
};

// OptPFD: evaluate every b in [0, maxw] from a bit-width histogram.
Plan plan_block(const uint32_t* in, size_t n) {
  uint32_t hist[33] = {};
  for (size_t i = 0; i < n; ++i) ++hist[std::bit_width(in[i])];

  unsigned maxw = 32;
  while (maxw > 0 && hist[maxw] == 0) --maxw;

  // above[t] = number of values needing more than t bits.
  uint32_t above[33];
  above[32] = 0;
  for (int t = 31; t >= 0; --t) above[t] = above[t + 1] + hist[t + 1];

  Plan best;
  best.bytes = SIZE_MAX;
  for (unsigned b = 0; b <= maxw; ++b) {
    const unsigned e = above[b];
    size_t bytes = 1 + bitpack_bytes(n, b);
    size_t packed_highs = 0, vbyte_highs = 0;
    if (e) {
      packed_highs = 1 + bitpack_bytes(e, maxw - b);
      for (unsigned t = b; t < maxw; t += 7) vbyte_highs += above[t];
      bytes += 1 + e + (vbyte_highs < packed_highs ? vbyte_highs : packed_highs);
    }
    if (bytes <= best.bytes) {  // '<=': on ties prefer larger b (fewer exceptions)
      best.b = b;
      best.e = e;
      best.high_bits = maxw - b;
      best.vbyte_highs = e && vbyte_highs < packed_highs;
      best.bytes = bytes;
    }
  }
  return best;
}

}  // namespace

void pfor_encode(const uint32_t* in, size_t n, ByteWriter& out) {
  DC_CHECK(n <= kBlockSize, "pfor_encode: block larger than kBlockSize");
  const Plan plan = plan_block(in, n);

  uint8_t* p = out.extend(plan.bytes);
  uint8_t* const end = p + plan.bytes;

  *p++ = static_cast<uint8_t>(plan.b | (plan.e ? kHasExceptions : 0) |
                              (plan.vbyte_highs ? kVByteHighs : 0));
  if (plan.e) *p++ = static_cast<uint8_t>(plan.e);
  bitpack(in, n, plan.b, p);  // stores the low b bits only
  p += bitpack_bytes(n, plan.b);

  if (plan.e) {
    // b < maxw <= 32 here, so the shift is well-defined.
    uint32_t highs[kBlockSize];
    unsigned k = 0;
    for (size_t i = 0; i < n; ++i) {
      const uint32_t hi = in[i] >> plan.b;
      if (hi) {
        p[k] = static_cast<uint8_t>(i);
        highs[k++] = hi;
      }
    }
    p += plan.e;
    if (plan.vbyte_highs) {
      p = vbyte_encode_to(highs, plan.e, p);
    } else {
      *p++ = static_cast<uint8_t>(plan.high_bits);
      bitpack(highs, plan.e, plan.high_bits, p);
      p += bitpack_bytes(plan.e, plan.high_bits);
    }
  }
  DC_CHECK(p == end, "pfor_encode: size plan mismatch");
}

const uint8_t* pfor_decode(const uint8_t* in, size_t n, uint32_t* out) {
  const uint8_t header = *in++;
  const unsigned b = header & kWidthMask;
  DC_CHECK(b <= 32 && n <= kBlockSize, "pfor_decode: corrupt header");

  if (!(header & kHasExceptions)) {
    bitunpack(in, n, b, out);
    return in + bitpack_bytes(n, b);
  }

  const unsigned e = *in++;
  DC_CHECK(b < 32 && e >= 1 && e <= n, "pfor_decode: corrupt exception count");
  bitunpack(in, n, b, out);
  in += bitpack_bytes(n, b);
  const uint8_t* positions = in;
  in += e;

  uint32_t highs[kBlockSize];
  if (header & kVByteHighs) {
    in = vbyte_decode(in, e, highs);
  } else {
    const unsigned hb = *in++;
    DC_CHECK(hb <= 32, "pfor_decode: corrupt exception width");
    bitunpack(in, e, hb, highs);
    in += bitpack_bytes(e, hb);
  }
  for (unsigned k = 0; k < e; ++k) {
    const unsigned i = positions[k];
    DC_CHECK(i < n, "pfor_decode: exception position out of range");
    out[i] |= highs[k] << b;
  }
  return in;
}

}  // namespace dc::codec
