// Tests for the posting-block integer codecs (bitpack, vbyte, pfor, raw).
//
// Every decode reads from a heap buffer sized exactly to the encoded bytes,
// so an over-read past the end of a block is reported when the tests run
// under AddressSanitizer.
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "codec/block_codec.h"
#include "testing.h"
#include "util/bytes.h"
#include "util/common.h"

using namespace dc;
using namespace dc::codec;

namespace {

// Deterministic, platform-independent PRNG.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  uint32_t u32() { return static_cast<uint32_t>(next() >> 32); }
  uint32_t below(uint32_t n) { return static_cast<uint32_t>((next() >> 32) % n); }
};

enum class Dist { kZeros, kEqual, kSmall, kFull32, kZipfGaps, kDocIdDeltas, kAllMax };
constexpr Dist kAllDists[] = {Dist::kZeros,    Dist::kEqual,       Dist::kSmall, Dist::kFull32,
                              Dist::kZipfGaps, Dist::kDocIdDeltas, Dist::kAllMax};
constexpr size_t kSizes[] = {1, 2, 7, 31, 32, 33, 127, 128};
constexpr CodecId kAllCodecs[] = {CodecId::kRaw, CodecId::kVByte, CodecId::kPFor};

const char* dist_name(Dist d) {
  switch (d) {
    case Dist::kZeros: return "zeros";
    case Dist::kEqual: return "equal";
    case Dist::kSmall: return "small";
    case Dist::kFull32: return "full32";
    case Dist::kZipfGaps: return "zipf_gaps";
    case Dist::kDocIdDeltas: return "docid_deltas";
    case Dist::kAllMax: return "all_max";
  }
  return "?";
}

std::vector<uint32_t> generate(Dist d, size_t n, uint64_t seed) {
  Rng rng(seed * 7919 + n);
  std::vector<uint32_t> v(n);
  switch (d) {
    case Dist::kZeros:
      break;
    case Dist::kEqual: {
      const uint32_t x = rng.u32() | 1;
      std::fill(v.begin(), v.end(), x);
      break;
    }
    case Dist::kSmall:
      for (auto& x : v) x = rng.below(16);
      break;
    case Dist::kFull32:
      for (auto& x : v) x = rng.u32();
      v[rng.below(static_cast<uint32_t>(n))] = 0xFFFFFFFFu;
      v[rng.below(static_cast<uint32_t>(n))] = 0;
      break;
    case Dist::kZipfGaps:
      // Mostly tiny gaps with rare huge outliers: the PFor exception path.
      for (auto& x : v) {
        const uint32_t r = rng.below(100);
        if (r < 4) {
          x = rng.u32() >> rng.below(16);  // outlier: up to 32 bits
        } else if (r < 20) {
          x = 8 + rng.below(56);
        } else {
          x = 1 + rng.below(7);
        }
      }
      if (n >= 3) v[n - 1] = 0xFFFFFFFFu;
      break;
    case Dist::kDocIdDeltas: {
      // Strictly increasing docIDs, then gap-encoded against the previous
      // block's last docID.
      const uint32_t base = rng.below(1u << 20);
      uint32_t doc = base;
      for (auto& x : v) x = doc += 1 + rng.below(20);
      delta_encode(v.data(), n, base);
      break;
    }
    case Dist::kAllMax:
      std::fill(v.begin(), v.end(), 0xFFFFFFFFu);
      break;
  }
  return v;
}

std::string ctx(CodecId id, Dist d, size_t n) {
  std::ostringstream os;
  os << codec_name(id) << "/" << dist_name(d) << "/n=" << n;
  return os.str();
}

#define CHECK_CTX(cond, context)                                                       \
  do {                                                                                 \
    if (!(cond)) ::dctest::report_failure(__FILE__, __LINE__, std::string(context) +   \
                                                                  ": expected " #cond); \
  } while (0)

// A heap copy whose size is exactly the encoded size (no slack for ASan).
struct ExactBuffer {
  explicit ExactBuffer(const ByteWriter& w)
      : size(w.size()), data(static_cast<uint8_t*>(std::malloc(std::max<size_t>(size, 1)))) {
    if (size) std::memcpy(data, w.data(), size);
  }
  ~ExactBuffer() { std::free(data); }
  ExactBuffer(const ExactBuffer&) = delete;
  ExactBuffer& operator=(const ExactBuffer&) = delete;
  size_t size;
  uint8_t* data;
  const uint8_t* end() const { return data + size; }
};

constexpr uint32_t kGuard = 0xCAFEBABEu;

// Decodes `buf` at `p` and checks the values, the returned pointer and that
// nothing beyond out[n - 1] was written.
const uint8_t* decode_and_check(CodecId id, const uint8_t* p, const std::vector<uint32_t>& want,
                                const std::string& context) {
  const size_t n = want.size();
  std::vector<uint32_t> out(n + 8, kGuard);
  const uint8_t* end = decode_block(id, p, n, out.data());
  for (size_t i = 0; i < n; ++i) {
    if (out[i] != want[i]) {
      std::ostringstream os;
      os << context << ": value " << i << " decoded as " << out[i] << ", want " << want[i];
      ::dctest::report_failure(__FILE__, __LINE__, os.str());
      break;
    }
  }
  for (size_t i = n; i < out.size(); ++i) CHECK_CTX(out[i] == kGuard, context + " wrote past out[n-1]");
  return end;
}

void roundtrip(CodecId id, const std::vector<uint32_t>& v, const std::string& context) {
  ByteWriter w;
  encode_block(id, v.data(), v.size(), w);
  ExactBuffer buf(w);
  const uint8_t* end = decode_and_check(id, buf.data, v, context);
  CHECK_CTX(end == buf.end(), context + " returned pointer");
}

// Reference implementations of the documented bitpack layouts.
std::vector<uint8_t> reference_pack(const std::vector<uint32_t>& v, unsigned bits) {
  const size_t n = v.size();
  std::vector<uint8_t> out(bitpack_bytes(n, bits), 0);
  for (size_t i = 0; i < n; ++i) {
    for (unsigned k = 0; k < bits; ++k) {
      if (!((v[i] >> k) & 1)) continue;
      size_t bit;
      if (n == kBlockSize) {
        // vertical: lane i%4, position i/4; word w of lane l is u32 #(4w + l)
        const size_t lane = i % 4, lane_bit = (i / 4) * bits + k;
        bit = (4 * (lane_bit / 32) + lane) * 32 + lane_bit % 32;
      } else {
        bit = i * bits + k;  // horizontal LSB-first stream
      }
      out[bit / 8] |= static_cast<uint8_t>(1u << (bit % 8));
    }
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// bit packing
// ---------------------------------------------------------------------------

TEST(Codec, BitsNeeded) {
  EXPECT_EQ(bits_needed(0), 0u);
  EXPECT_EQ(bits_needed(1), 1u);
  EXPECT_EQ(bits_needed(2), 2u);
  EXPECT_EQ(bits_needed(3), 2u);
  EXPECT_EQ(bits_needed(255), 8u);
  EXPECT_EQ(bits_needed(256), 9u);
  EXPECT_EQ(bits_needed(0x7FFFFFFFu), 31u);
  EXPECT_EQ(bits_needed(0x80000000u), 32u);
  EXPECT_EQ(bits_needed(0xFFFFFFFFu), 32u);
}

TEST(Codec, BitpackBytes) {
  EXPECT_EQ(bitpack_bytes(0, 32), 0u);
  EXPECT_EQ(bitpack_bytes(128, 0), 0u);
  EXPECT_EQ(bitpack_bytes(1, 1), 1u);
  EXPECT_EQ(bitpack_bytes(3, 3), 2u);
  EXPECT_EQ(bitpack_bytes(8, 3), 3u);
  EXPECT_EQ(bitpack_bytes(128, 5), 80u);
  EXPECT_EQ(bitpack_bytes(127, 32), 508u);
}

TEST(Codec, BitpackAllWidths) {
  const size_t sizes[] = {128, 1, 3, 31, 32, 33, 77, 127, 129, 255, 256};
  Rng rng(42);
  for (unsigned bits = 0; bits <= 32; ++bits) {
    const uint32_t mask = bits == 32 ? 0xFFFFFFFFu : (1u << bits) - 1;
    for (size_t n : sizes) {
      const std::string c = "bits=" + std::to_string(bits) + " n=" + std::to_string(n);
      // Unmasked random input (bitpack must drop the high bits) with the
      // extremes of the width forced in.
      std::vector<uint32_t> in(n), want(n);
      for (auto& x : in) x = rng.u32();
      in[0] = mask;
      if (n > 1) in[n - 1] = 0xFFFFFFFFu;
      if (n > 2) in[n / 2] = 0;
      for (size_t i = 0; i < n; ++i) want[i] = in[i] & mask;

      // Pack into a buffer with guard bytes: exactly bitpack_bytes are written.
      const size_t bytes = bitpack_bytes(n, bits);
      std::vector<uint8_t> packed(bytes + 16, 0x5A);
      bitpack(in.data(), n, bits, packed.data());
      for (size_t i = bytes; i < packed.size(); ++i) CHECK_CTX(packed[i] == 0x5A, c + " pack overrun");
      packed.resize(bytes);
      CHECK_CTX(packed == reference_pack(want, bits), c + " layout matches reference");

      // Unpack from an exact-size heap buffer.
      std::unique_ptr<uint8_t[]> exact(new uint8_t[std::max<size_t>(bytes, 1)]);
      if (bytes) std::memcpy(exact.get(), packed.data(), bytes);
      std::vector<uint32_t> out(n + 4, kGuard);
      bitunpack(exact.get(), n, bits, out.data());
      bool ok = std::equal(want.begin(), want.end(), out.begin());
      CHECK_CTX(ok, c + " roundtrip");
      for (size_t i = n; i < out.size(); ++i) CHECK_CTX(out[i] == kGuard, c + " unpack overrun");
    }
  }
}

TEST(Codec, BitpackRejectsBadWidth) {
  uint32_t in[4] = {};
  uint8_t buf[64] = {};
  EXPECT_THROW(bitpack(in, 4, 33, buf));
  EXPECT_THROW(bitunpack(buf, 4, 40, in));
}

// ---------------------------------------------------------------------------
// round trips through every codec
// ---------------------------------------------------------------------------

TEST(Codec, RoundTripAllCodecsSizesDistributions) {
  for (CodecId id : kAllCodecs)
    for (Dist d : kAllDists)
      for (size_t n : kSizes)
        for (uint64_t seed = 1; seed <= 3; ++seed) roundtrip(id, generate(d, n, seed), ctx(id, d, n));
}

TEST(Codec, RoundTripRandomFuzz) {
  Rng rng(2024);
  for (int iter = 0; iter < 3000; ++iter) {
    const size_t n = 1 + rng.below(kBlockSize);
    std::vector<uint32_t> v(n);
    // Random bit width per block, with occasional wider outliers.
    const unsigned width = rng.below(33);
    for (auto& x : v) {
      x = width == 0 ? 0 : rng.u32() >> (32 - width);
      if (rng.below(50) == 0) x = rng.u32() >> rng.below(32);
    }
    for (CodecId id : kAllCodecs) roundtrip(id, v, std::string(codec_name(id)) + "/fuzz#" + std::to_string(iter));
  }
}

TEST(Codec, ConcatenatedBlocksDecodeWithCorrectPointers) {
  for (CodecId id : kAllCodecs) {
    ByteWriter w;
    std::vector<std::vector<uint32_t>> blocks;
    std::vector<size_t> offsets;
    uint64_t seed = 11;
    for (Dist d : kAllDists) {
      for (size_t n : kSizes) {
        offsets.push_back(w.size());
        blocks.push_back(generate(d, n, ++seed));
        encode_block(id, blocks.back().data(), n, w);
      }
    }
    offsets.push_back(w.size());

    ExactBuffer buf(w);
    const uint8_t* p = buf.data;
    for (size_t b = 0; b < blocks.size(); ++b) {
      const std::string c = std::string(codec_name(id)) + "/block#" + std::to_string(b);
      CHECK_CTX(p == buf.data + offsets[b], c + " start");
      p = decode_and_check(id, p, blocks[b], c);
    }
    CHECK_CTX(p == buf.end(), std::string(codec_name(id)) + " stream end");
  }
}

TEST(Codec, MixedCodecStream) {
  // Blocks of different codecs can share a stream as long as the reader
  // knows (id, n) for each.
  ByteWriter w;
  struct Item {
    CodecId id;
    std::vector<uint32_t> v;
  };
  std::vector<Item> items;
  Rng rng(5);
  for (int i = 0; i < 60; ++i) {
    Item it{kAllCodecs[i % 3], generate(kAllDists[rng.below(7)], kSizes[rng.below(8)], 100 + i)};
    encode_block(it.id, it.v.data(), it.v.size(), w);
    items.push_back(std::move(it));
  }
  ExactBuffer buf(w);
  const uint8_t* p = buf.data;
  for (size_t i = 0; i < items.size(); ++i)
    p = decode_and_check(items[i].id, p, items[i].v, "mixed#" + std::to_string(i));
  EXPECT_TRUE(p == buf.end());
}

TEST(Codec, DecodeAtExactEndOfMallocBuffer) {
  // The block is the last thing in a malloc'd region whose size is exactly the
  // encoded size; under ASan any read past the end aborts the test binary.
  for (CodecId id : kAllCodecs) {
    for (Dist d : kAllDists) {
      for (size_t n : kSizes) {
        const auto v = generate(d, n, 77);
        ByteWriter w;
        w.put_u32(0x12345678);  // something in front, like a previous block
        const size_t start = w.size();
        encode_block(id, v.data(), n, w);
        const size_t len = w.size() - start;
        uint8_t* raw = static_cast<uint8_t*>(std::malloc(len));
        std::memcpy(raw, w.data() + start, len);
        const uint8_t* end = decode_and_check(id, raw, v, ctx(id, d, n));
        CHECK_CTX(end == raw + len, ctx(id, d, n) + " end pointer");
        if (id == CodecId::kVByte) CHECK_CTX(vbyte_skip(raw, n) == raw + len, ctx(id, d, n) + " skip");
        std::free(raw);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// vbyte
// ---------------------------------------------------------------------------

TEST(Codec, VByteMatchesVarintFormat) {
  const uint32_t vals[] = {0,          1,          127,        128,        300,        16383,
                           16384,      2097151,    2097152,    268435455,  268435456,  0xFFFFFFFFu};
  for (uint32_t v : vals) {
    ByteWriter a, b;
    vbyte_encode(&v, 1, a);
    b.put_varint(v);
    EXPECT_TRUE(a.buffer() == b.buffer());
    EXPECT_EQ(a.size(), vbyte_size(v));
  }
}

TEST(Codec, VByteSkipMatchesDecode) {
  for (Dist d : kAllDists) {
    for (size_t n : kSizes) {
      const auto v = generate(d, n, 9);
      ByteWriter w;
      vbyte_encode(v.data(), n, w);
      ExactBuffer buf(w);
      std::vector<uint32_t> out(n);
      const uint8_t* dec = vbyte_decode(buf.data, n, out.data());
      const uint8_t* skp = vbyte_skip(buf.data, n);
      CHECK_CTX(dec == skp, ctx(CodecId::kVByte, d, n) + " skip==decode");
      CHECK_CTX(skp == buf.end(), ctx(CodecId::kVByte, d, n) + " skip end");
    }
  }
  // Skipping a prefix of a stream lands on the next block, for every split.
  Rng rng(3);
  std::vector<uint32_t> v(300);
  for (auto& x : v) x = rng.u32() >> rng.below(32);
  ByteWriter w;
  vbyte_encode(v.data(), v.size(), w);
  ExactBuffer buf(w);
  for (size_t k = 0; k <= v.size(); ++k) {
    const uint8_t* p = vbyte_skip(buf.data, k);
    std::vector<uint32_t> rest(v.size() - k);
    const uint8_t* end = vbyte_decode(p, rest.size(), rest.data());
    CHECK_CTX(std::equal(rest.begin(), rest.end(), v.begin() + static_cast<long>(k)),
              "skip " + std::to_string(k) + " then decode");
    CHECK_CTX(end == buf.end(), "skip " + std::to_string(k) + " end");
  }
}

TEST(Codec, VByteFastPathPatterns) {
  // Runs that hit the 8x1-byte fast path and the table-driven path with
  // longer values, including transitions near the 8-values-left boundary.
  std::vector<uint32_t> v;
  for (int i = 0; i < 16; ++i) v.push_back(i);
  for (int i = 0; i < 12; ++i) v.push_back(200 + 1000 * i);
  for (int i = 0; i < 9; ++i) v.push_back(1u << (3 * i + 3));
  v.push_back(0xFFFFFFFFu);
  for (int i = 0; i < 9; ++i) v.push_back(300 + i);
  for (size_t n = 1; n <= v.size(); ++n) {
    std::vector<uint32_t> pre(v.begin(), v.begin() + static_cast<long>(n));
    roundtrip(CodecId::kVByte, pre, "vbyte/pattern/n=" + std::to_string(n));
  }
}

TEST(Codec, VByteRejectsOverlongValue) {
  // Six continuation bytes: not a valid u32 varint.
  const uint8_t bad[16] = {0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  uint32_t out[9];
  EXPECT_THROW(vbyte_decode(bad, 1, out));  // byte-wise path
  EXPECT_THROW(vbyte_decode(bad, 9, out));  // word path
}

// ---------------------------------------------------------------------------
// PFor
// ---------------------------------------------------------------------------

namespace {

// Brute-force O(33 n) size of the optimal OptPFD encoding (layout in pfor.h).
size_t brute_force_pfor_size(const std::vector<uint32_t>& v) {
  size_t best = SIZE_MAX;
  for (unsigned b = 0; b <= 32; ++b) {
    size_t e = 0, vb = 0;
    unsigned hb = 0;
    for (uint32_t x : v) {
      const uint32_t hi = b == 32 ? 0 : x >> b;
      if (!hi) continue;
      ++e;
      vb += vbyte_size(hi);
      hb = std::max(hb, bits_needed(hi));
    }
    size_t size = 1 + bitpack_bytes(v.size(), b);
    if (e) size += 1 + e + std::min(vb, 1 + bitpack_bytes(e, hb));
    best = std::min(best, size);
  }
  return best;
}

}  // namespace

TEST(Codec, PForChoosesOptimalWidth) {
  for (Dist d : kAllDists) {
    for (size_t n : kSizes) {
      for (uint64_t seed = 1; seed <= 4; ++seed) {
        const auto v = generate(d, n, seed);
        ByteWriter w;
        pfor_encode(v.data(), n, w);
        CHECK_CTX(w.size() == brute_force_pfor_size(v), ctx(CodecId::kPFor, d, n) + " optimal size");
        uint32_t maxv = 0;
        for (uint32_t x : v) maxv = std::max(maxv, x);
        CHECK_CTX(w.size() <= 1 + bitpack_bytes(n, bits_needed(maxv)),
                  ctx(CodecId::kPFor, d, n) + " never worse than plain bitpack + 1");
      }
    }
  }
}

TEST(Codec, PForExceptionPaths) {
  Rng rng(8);
  // One moderate outlier: exception with vbyte-coded high part.
  std::vector<uint32_t> a(128);
  for (auto& x : a) x = rng.below(4);
  a[77] = 300;
  // Half the values ~2^20: many exceptions of one width, bit-packed highs.
  std::vector<uint32_t> b(128);
  for (size_t i = 0; i < b.size(); ++i) b[i] = i % 2 ? (1u << 20) + rng.below(1u << 20) : rng.below(4);
  // Outliers at the first and last positions, including 0xFFFFFFFF.
  std::vector<uint32_t> c(128);
  for (auto& x : c) x = 1 + rng.below(7);
  c[0] = 0xFFFFFFFFu;
  c[127] = 0x80000000u;

  for (auto* v : {&a, &b, &c}) {
    ByteWriter w;
    pfor_encode(v->data(), v->size(), w);
    EXPECT_TRUE(w.data()[0] & 0x40);  // has exceptions
    roundtrip(CodecId::kPFor, *v, "pfor/exceptions");
  }
  ByteWriter wa, wb;
  pfor_encode(a.data(), a.size(), wa);
  pfor_encode(b.data(), b.size(), wb);
  EXPECT_TRUE(wa.data()[0] & 0x80);     // vbyte highs
  EXPECT_FALSE(wb.data()[0] & 0x80);    // bit-packed highs
  EXPECT_EQ(wa.data()[0] & 0x3F, 2);    // b = 2 for the single-outlier block
  EXPECT_EQ(wa.data()[1], 1);           // one exception

  // Zipf-like gaps actually use the exception path and beat plain packing.
  const auto z = generate(Dist::kZipfGaps, 128, 1);
  ByteWriter wz;
  pfor_encode(z.data(), z.size(), wz);
  EXPECT_TRUE(wz.data()[0] & 0x40);
  EXPECT_LT(wz.size(), bitpack_bytes(128, 32) / 2);
}

TEST(Codec, PForEmptyBlock) {
  ByteWriter w;
  pfor_encode(nullptr, 0, w);
  EXPECT_EQ(w.size(), 1u);
  uint32_t out[1] = {kGuard};
  EXPECT_TRUE(pfor_decode(w.data(), 0, out) == w.data() + 1);
  EXPECT_EQ(out[0], kGuard);
}

TEST(Codec, PForRejectsCorruptBlocks) {
  uint32_t out[kBlockSize];
  std::vector<uint8_t> buf(1024, 0);
  buf[0] = 40;  // width 40
  EXPECT_THROW(pfor_decode(buf.data(), 16, out));

  // Valid block, then corrupt the exception position.
  std::vector<uint32_t> v(16, 1);
  v[3] = 1000;
  ByteWriter w;
  pfor_encode(v.data(), v.size(), w);
  std::vector<uint8_t> bad = w.buffer();
  ASSERT_TRUE(bad[0] & 0x40);
  const size_t pos_at = 2 + bitpack_bytes(16, bad[0] & 0x3F);
  EXPECT_EQ(bad[pos_at], 3);
  bad[pos_at] = 200;
  EXPECT_THROW(pfor_decode(bad.data(), 16, out));

  std::vector<uint32_t> big(130, 1);
  ByteWriter w2;
  EXPECT_THROW(pfor_encode(big.data(), big.size(), w2));
}

// ---------------------------------------------------------------------------
// delta helpers and codec ids
// ---------------------------------------------------------------------------

TEST(Codec, DeltaEncodePrefixSumRoundTrip) {
  Rng rng(17);
  for (size_t n : {size_t{0}, size_t{1}, size_t{2}, size_t{7}, size_t{128}, size_t{1000}}) {
    for (bool weak : {false, true}) {
      const uint32_t base = rng.below(1000);
      std::vector<uint32_t> docs(n);
      uint32_t d = base;
      for (auto& x : docs) x = d += (weak ? rng.below(3) : 1 + rng.below(50));
      std::vector<uint32_t> v = docs;
      delta_encode(v.data(), n, base);
      if (n) {
        EXPECT_EQ(v[0], docs[0] - base);
        for (size_t i = 1; i < n; ++i) {
          if (v[i] != docs[i] - docs[i - 1]) {
            EXPECT_EQ(v[i], docs[i] - docs[i - 1]);
            break;
          }
        }
      }
      prefix_sum(v.data(), n, base);
      EXPECT_TRUE(v == docs);
    }
  }
  // Wrap-around: arbitrary (non-monotone) input still round-trips.
  std::vector<uint32_t> any = {5, 0xFFFFFFFFu, 3, 3, 0, 0x80000000u};
  std::vector<uint32_t> v = any;
  delta_encode(v.data(), v.size(), 0xFFFFFFF0u);
  prefix_sum(v.data(), v.size(), 0xFFFFFFF0u);
  EXPECT_TRUE(v == any);
}

TEST(Codec, CodecNames) {
  for (CodecId id : kAllCodecs) EXPECT_TRUE(codec_from_name(codec_name(id)) == id);
  EXPECT_EQ(std::string(codec_name(CodecId::kRaw)), "raw");
  EXPECT_EQ(std::string(codec_name(CodecId::kVByte)), "vbyte");
  EXPECT_EQ(std::string(codec_name(CodecId::kPFor)), "pfor");
  EXPECT_EQ(std::string(codec_name(static_cast<CodecId>(9))), "unknown");
  EXPECT_THROW(codec_from_name("roaring"));
  EXPECT_THROW(codec_from_name("PFOR"));
  EXPECT_THROW(codec_from_name(""));
  bool threw_dc = false;
  try {
    codec_from_name("lz4");
  } catch (const DcError&) {
    threw_dc = true;
  }
  EXPECT_TRUE(threw_dc);
  ByteWriter w;
  uint32_t x = 1;
  EXPECT_THROW(encode_block(static_cast<CodecId>(7), &x, 1, w));
}
