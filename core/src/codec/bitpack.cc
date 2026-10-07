// Bit packing kernels. See bitpack.h for the byte layouts.
//
// Every bit width gets its own template instantiation with all shifts and
// word offsets known at compile time (the per-value steps are expanded with a
// fold over std::index_sequence, so there are no runtime loops over values).
// The 128-value kernels operate on four interleaved lanes with identical
// shifts; the inner `for (l < 4)` loops are what the compiler vectorizes.
#include "codec/bitpack.h"

#include <array>
#include <bit>
#include <cstring>
#include <utility>

#include "util/bytes.h"
#include "util/common.h"

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#define DC_RESTRICT __restrict
#else
#define DC_RESTRICT
#endif

namespace dc::codec {

static_assert(std::endian::native == std::endian::little,
              "packed words are copied with memcpy; a big-endian port needs byte swaps");

namespace {

template <unsigned B>
constexpr uint32_t low_mask() {
  if constexpr (B >= 32) {
    return 0xFFFFFFFFu;
  } else {
    return (uint32_t{1} << B) - 1;
  }
}

// Position of value J (within a 32-value lane or group) in the packed words.
template <unsigned B, unsigned J>
struct Slot {
  static constexpr unsigned kBit = J * B;
  static constexpr unsigned kWord = kBit / 32;
  static constexpr unsigned kShift = kBit % 32;
  // The value straddles two words.
  static constexpr bool kSpills = kShift + B > 32;
  // The right shift alone already isolates the value (no mask needed).
  static constexpr bool kEndsAtWordTop = kShift + B == 32;
};

// ---------------------------------------------------------------------------
// 128-value vertical kernels (4 lanes x 32 values, words interleaved by lane)
// ---------------------------------------------------------------------------

template <unsigned B, unsigned J>
inline void pack128_step(const uint32_t* DC_RESTRICT in, uint32_t* DC_RESTRICT w) {
  using S = Slot<B, J>;
  for (unsigned l = 0; l < 4; ++l) {
    const uint32_t v = in[4 * J + l] & low_mask<B>();
    w[4 * S::kWord + l] |= v << S::kShift;
    if constexpr (S::kSpills) w[4 * (S::kWord + 1) + l] |= v >> (32 - S::kShift);
  }
}

template <unsigned B, size_t... J>
inline void pack128_all(const uint32_t* in, uint32_t* w, std::index_sequence<J...>) {
  (pack128_step<B, J>(in, w), ...);
}

template <unsigned B>
void pack128(const uint32_t* in, uint8_t* out) {
  if constexpr (B == 0) {
    (void)in;
    (void)out;
  } else if constexpr (B == 32) {
    std::memcpy(out, in, kBlockSize * 4);
  } else {
    uint32_t w[4 * B] = {};
    pack128_all<B>(in, w, std::make_index_sequence<32>{});
    std::memcpy(out, w, sizeof w);
  }
}

template <unsigned B, unsigned J>
inline void unpack128_step(const uint32_t* DC_RESTRICT w, uint32_t* DC_RESTRICT out) {
  using S = Slot<B, J>;
  for (unsigned l = 0; l < 4; ++l) {
    uint32_t v = w[4 * S::kWord + l] >> S::kShift;
    if constexpr (S::kSpills) v |= w[4 * (S::kWord + 1) + l] << (32 - S::kShift);
    if constexpr (!S::kEndsAtWordTop) v &= low_mask<B>();
    out[4 * J + l] = v;
  }
}

template <unsigned B, size_t... J>
inline void unpack128_all(const uint32_t* w, uint32_t* out, std::index_sequence<J...>) {
  (unpack128_step<B, J>(w, out), ...);
}

template <unsigned B>
void unpack128(const uint8_t* in, uint32_t* out) {
  if constexpr (B == 0) {
    (void)in;
    std::memset(out, 0, kBlockSize * 4);
  } else if constexpr (B == 32) {
    std::memcpy(out, in, kBlockSize * 4);
  } else {
    // Copying the words into a local array tells the compiler they cannot
    // alias `out` (a uint8_t* input may alias anything), which lets it keep
    // them in registers and vectorize across the four lanes.
    uint32_t w[4 * B];
    std::memcpy(w, in, sizeof w);
    unpack128_all<B>(w, out, std::make_index_sequence<32>{});
  }
}

// ---------------------------------------------------------------------------
// Horizontal kernels for arbitrary n: 32-value groups (= B words) plus a tail
// ---------------------------------------------------------------------------

template <unsigned B, unsigned J>
inline void pack32_step(const uint32_t* DC_RESTRICT in, uint32_t* DC_RESTRICT w) {
  using S = Slot<B, J>;
  const uint32_t v = in[J] & low_mask<B>();
  w[S::kWord] |= v << S::kShift;
  if constexpr (S::kSpills) w[S::kWord + 1] |= v >> (32 - S::kShift);
}

template <unsigned B, size_t... J>
inline void pack32_all(const uint32_t* in, uint32_t* w, std::index_sequence<J...>) {
  (pack32_step<B, J>(in, w), ...);
}

template <unsigned B, unsigned J>
inline void unpack32_step(const uint32_t* DC_RESTRICT w, uint32_t* DC_RESTRICT out) {
  using S = Slot<B, J>;
  uint32_t v = w[S::kWord] >> S::kShift;
  if constexpr (S::kSpills) v |= w[S::kWord + 1] << (32 - S::kShift);
  if constexpr (!S::kEndsAtWordTop) v &= low_mask<B>();
  out[J] = v;
}

template <unsigned B, size_t... J>
inline void unpack32_all(const uint32_t* w, uint32_t* out, std::index_sequence<J...>) {
  (unpack32_step<B, J>(w, out), ...);
}

// Tail of fewer than 32 values (any bit width), streamed through a 64-bit
// accumulator one byte at a time so that exactly ceil(n*bits/8) bytes are
// touched.
void pack_tail(const uint32_t* in, size_t n, unsigned bits, uint8_t* out) {
  if (bits == 0) return;
  const uint64_t mask = bits == 32 ? 0xFFFFFFFFull : (uint64_t{1} << bits) - 1;
  uint64_t acc = 0;
  unsigned have = 0;  // pending bits in acc; always < 8 between values
  for (size_t i = 0; i < n; ++i) {
    acc |= (in[i] & mask) << have;
    have += bits;
    while (have >= 8) {
      *out++ = static_cast<uint8_t>(acc);
      acc >>= 8;
      have -= 8;
    }
  }
  if (have) *out = static_cast<uint8_t>(acc);
}

void unpack_tail(const uint8_t* in, size_t n, unsigned bits, uint32_t* out) {
  if (bits == 0) {
    std::memset(out, 0, n * 4);
    return;
  }
  const uint64_t mask = bits == 32 ? 0xFFFFFFFFull : (uint64_t{1} << bits) - 1;
  uint64_t acc = 0;
  unsigned have = 0;  // valid bits in acc
  for (size_t i = 0; i < n; ++i) {
    while (have < bits) {  // load only the bytes this value actually needs
      acc |= static_cast<uint64_t>(*in++) << have;
      have += 8;
    }
    out[i] = static_cast<uint32_t>(acc & mask);
    acc >>= bits;
    have -= bits;
  }
}

template <unsigned B>
void pack_any(const uint32_t* in, size_t n, uint8_t* out) {
  if constexpr (B == 0) {
    (void)in;
    (void)n;
    (void)out;
  } else {
    const size_t groups = n / 32;
    for (size_t g = 0; g < groups; ++g, in += 32, out += 4 * B) {
      if constexpr (B == 32) {
        std::memcpy(out, in, 32 * 4);
      } else {
        uint32_t w[B] = {};
        pack32_all<B>(in, w, std::make_index_sequence<32>{});
        std::memcpy(out, w, sizeof w);
      }
    }
    pack_tail(in, n % 32, B, out);
  }
}

template <unsigned B>
void unpack_any(const uint8_t* in, size_t n, uint32_t* out) {
  if constexpr (B == 0) {
    (void)in;
    std::memset(out, 0, n * 4);
  } else {
    const size_t groups = n / 32;
    for (size_t g = 0; g < groups; ++g, in += 4 * B, out += 32) {
      if constexpr (B == 32) {
        std::memcpy(out, in, 32 * 4);
      } else {
        uint32_t w[B];  // local copy: see unpack128
        std::memcpy(w, in, sizeof w);
        unpack32_all<B>(w, out, std::make_index_sequence<32>{});
      }
    }
    unpack_tail(in, n % 32, B, out);
  }
}

// ---------------------------------------------------------------------------
// Dispatch tables indexed by bit width 0..32
// ---------------------------------------------------------------------------

using Pack128Fn = void (*)(const uint32_t*, uint8_t*);
using Unpack128Fn = void (*)(const uint8_t*, uint32_t*);
using PackAnyFn = void (*)(const uint32_t*, size_t, uint8_t*);
using UnpackAnyFn = void (*)(const uint8_t*, size_t, uint32_t*);

template <size_t... B>
constexpr std::array<Pack128Fn, 33> make_pack128(std::index_sequence<B...>) {
  return {&pack128<B>...};
}
template <size_t... B>
constexpr std::array<Unpack128Fn, 33> make_unpack128(std::index_sequence<B...>) {
  return {&unpack128<B>...};
}
template <size_t... B>
constexpr std::array<PackAnyFn, 33> make_pack_any(std::index_sequence<B...>) {
  return {&pack_any<B>...};
}
template <size_t... B>
constexpr std::array<UnpackAnyFn, 33> make_unpack_any(std::index_sequence<B...>) {
  return {&unpack_any<B>...};
}

constexpr auto kPack128 = make_pack128(std::make_index_sequence<33>{});
constexpr auto kUnpack128 = make_unpack128(std::make_index_sequence<33>{});
constexpr auto kPackAny = make_pack_any(std::make_index_sequence<33>{});
constexpr auto kUnpackAny = make_unpack_any(std::make_index_sequence<33>{});

}  // namespace

void bitpack(const uint32_t* in, size_t n, unsigned bits, uint8_t* out) {
  DC_CHECK(bits <= 32, "bitpack: bit width > 32");
  if (n == kBlockSize) {
    kPack128[bits](in, out);
  } else {
    kPackAny[bits](in, n, out);
  }
}

void bitunpack(const uint8_t* in, size_t n, unsigned bits, uint32_t* out) {
  DC_CHECK(bits <= 32, "bitunpack: bit width > 32 (corrupt block?)");
  if (n == kBlockSize) {
    kUnpack128[bits](in, out);
  } else {
    kUnpackAny[bits](in, n, out);
  }
}

}  // namespace dc::codec
