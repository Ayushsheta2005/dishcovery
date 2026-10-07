// Roaring bitmap implementation. See roaring.h for the layout and format.
//
// Organisation:
//   1. word / array / run search helpers
//   2. container construction, conversion and canonical form
//   3. single-container mutation and queries
//   4. pairwise container kernels (and / or / andnot / xor, counting, in-place)
//   5. RoaringBitmap methods
//   6. Iterator
//   7. serialization

#include "codec/roaring.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstring>
#include <utility>

namespace dc {

using roaring_detail::Container;
using roaring_detail::Kind;
using roaring_detail::kArrayMaxCard;
using roaring_detail::kBitmapWords;
using roaring_detail::lower_bound_u16;
using roaring_detail::Run;
using roaring_detail::run_search;

static_assert(std::endian::native == std::endian::little,
              "roaring payloads are bulk-copied as little-endian (as util/bytes.h assumes)");
static_assert(sizeof(Run) == 4, "Run must be two packed uint16s");

namespace {

constexpr uint32_t kMagic = 0x314D4252u;  // bytes 'R' 'B' 'M' '1'
constexpr uint32_t kFullCard = 65536;     // values per container
constexpr size_t kBitmapBytes = kBitmapWords * sizeof(uint64_t);
constexpr const char* kSentinelMsg = "values 0xFFFFFFFE/0xFFFFFFFF are reserved iterator sentinels";

inline uint32_t popcnt(uint64_t w) { return uint32_t(__builtin_popcountll(w)); }
inline uint32_t ctz64(uint64_t w) { return uint32_t(__builtin_ctzll(w)); }
inline uint32_t clz64(uint64_t w) { return uint32_t(__builtin_clzll(w)); }

template <class T>
void release(std::vector<T>& v) {
  std::vector<T>().swap(v);
}

// ===========================================================================
// 1. Search helpers

// Masks within one word: bits >= b and bits <= b (b taken mod 64).
inline uint64_t mask_from(uint32_t b) { return ~0ULL << (b & 63); }
inline uint64_t mask_upto(uint32_t b) { return ~0ULL >> (63 - (b & 63)); }

inline bool test_bit(const uint64_t* w, uint32_t x) { return (w[x >> 6] >> (x & 63)) & 1; }
inline void set_bit(uint64_t* w, uint32_t x) { w[x >> 6] |= 1ULL << (x & 63); }

// Range operations on a 1024-word bitmap; [lo, hi] is inclusive.
void set_range(uint64_t* w, uint32_t lo, uint32_t hi) {
  const uint32_t a = lo >> 6, b = hi >> 6;
  if (a == b) {
    w[a] |= mask_from(lo) & mask_upto(hi);
    return;
  }
  w[a] |= mask_from(lo);
  std::fill(w + a + 1, w + b, ~0ULL);
  w[b] |= mask_upto(hi);
}

void clear_range(uint64_t* w, uint32_t lo, uint32_t hi) {
  const uint32_t a = lo >> 6, b = hi >> 6;
  if (a == b) {
    w[a] &= ~(mask_from(lo) & mask_upto(hi));
    return;
  }
  w[a] &= ~mask_from(lo);
  std::fill(w + a + 1, w + b, 0ULL);
  w[b] &= ~mask_upto(hi);
}

uint32_t popcount_range(const uint64_t* w, uint32_t lo, uint32_t hi) {
  const uint32_t a = lo >> 6, b = hi >> 6;
  if (a == b) return popcnt(w[a] & mask_from(lo) & mask_upto(hi));
  uint32_t n = popcnt(w[a] & mask_from(lo));
  for (uint32_t i = a + 1; i < b; ++i) n += popcnt(w[i]);
  return n + popcnt(w[b] & mask_upto(hi));
}

uint32_t popcount_all(const uint64_t* w) {
  uint32_t n = 0;
  for (uint32_t i = 0; i < kBitmapWords; ++i) n += popcnt(w[i]);
  return n;
}

// First index i in [pos, n) with a[i] >= x, or n. Exponential probing from
// pos then binary search: O(log distance), which suits forward cursors and
// skewed intersections.
size_t gallop_u16(const uint16_t* a, size_t pos, size_t n, uint16_t x) {
  if (pos >= n || a[pos] >= x) return pos;
  size_t lo = pos, step = 1, hi = pos + 1;  // invariant: a[lo] < x
  while (hi < n && a[hi] < x) {
    lo = hi;
    step <<= 1;
    hi = lo + step;
  }
  if (hi > n) hi = n;
  // Answer lies in (lo, hi]; hi itself is either n or known to satisfy a[hi] >= x.
  return lo + 1 + lower_bound_u16(a + lo + 1, hi - lo - 1, x);
}

// First run index i in [pos, n) with r[i].end() >= x, or n (galloping).
size_t gallop_run(const Run* r, size_t pos, size_t n, uint32_t x) {
  if (pos >= n || r[pos].end() >= x) return pos;
  size_t lo = pos, step = 1, hi = pos + 1;  // invariant: r[lo].end() < x
  while (hi < n && r[hi].end() < x) {
    lo = hi;
    step <<= 1;
    hi = lo + step;
  }
  if (hi > n) hi = n;
  return run_search(r, lo + 1, hi, x);
}

// Membership against a run list for an increasing sequence of probes.
struct RunCursor {
  const Run* r;
  size_t n;
  size_t j = 0;
  bool contains(uint32_t v) {
    j = gallop_run(r, j, n, v);
    return j < n && r[j].start <= v;
  }
};

// ===========================================================================
// 2. Container construction, conversion, canonical form

// Serialized payload sizes, used to choose encodings.
inline size_t plain_bytes(uint32_t card) {
  return card <= kArrayMaxCard ? 2 * size_t(card) : kBitmapBytes;
}
inline size_t run_bytes(size_t nruns) { return 2 + 4 * nruns; }
// Runs are used only when strictly smaller than the array/bitmap encoding.
inline bool run_is_best(size_t nruns, uint32_t card) { return run_bytes(nruns) < plain_bytes(card); }

size_t payload_bytes(const Container& c) {
  switch (c.kind) {
    case Kind::kArray: return 2 * size_t(c.card);
    case Kind::kBitmap: return kBitmapBytes;
    case Kind::kRun: return run_bytes(c.runs.size());
  }
  return 0;
}

Container new_bitmap() {
  Container c;
  c.kind = Kind::kBitmap;
  c.bits.assign(kBitmapWords, 0);
  return c;
}

Container new_full() {
  Container c;
  c.kind = Kind::kRun;
  c.runs.push_back(Run{0, 0xFFFF});
  c.card = kFullCard;
  return c;
}

inline bool is_full(const Container& c) { return c.card == kFullCard; }

// Container holding exactly [lo, hi], in its cheapest encoding.
Container new_range(uint32_t lo, uint32_t hi) {
  const uint32_t n = hi - lo + 1;
  Container c;
  c.card = n;
  if (run_is_best(1, n)) {
    c.kind = Kind::kRun;
    c.runs.push_back(Run{uint16_t(lo), uint16_t(hi - lo)});
  } else {  // at most 3 values
    for (uint32_t v = lo; v <= hi; ++v) c.arr.push_back(uint16_t(v));
  }
  return c;
}

// Appends inclusive intervals with non-decreasing starts, coalescing
// overlapping and adjacent ones so the output is canonical.
struct RunWriter {
  std::vector<Run>& out;
  uint32_t card = 0;

  void add(uint32_t s, uint32_t e) {
    if (!out.empty()) {
      Run& b = out.back();
      const uint32_t be = b.end();
      if (s <= be + 1) {
        if (e > be) {
          card += e - be;
          b.len = uint16_t(e - b.start);
        }
        return;
      }
    }
    out.push_back(Run{uint16_t(s), uint16_t(e - s)});
    card += e - s + 1;
  }
};

void bitmap_to_array(Container& c) {
  std::vector<uint16_t> a(c.card);
  size_t k = 0;
  for (uint32_t i = 0; i < kBitmapWords; ++i)
    for (uint64_t w = c.bits[i]; w; w &= w - 1) a[k++] = uint16_t((i << 6) | ctz64(w));
  c.arr = std::move(a);
  release(c.bits);
  c.kind = Kind::kArray;
}

void array_to_bitmap(Container& c) {
  c.bits.assign(kBitmapWords, 0);
  for (uint16_t v : c.arr) set_bit(c.bits.data(), v);
  release(c.arr);
  c.kind = Kind::kBitmap;
}

// Run -> array or bitmap, chosen by cardinality.
void run_to_plain(Container& c) {
  if (c.card <= kArrayMaxCard) {
    std::vector<uint16_t> a;
    a.reserve(c.card);
    for (const Run& r : c.runs)
      for (uint32_t v = r.start; v <= r.end(); ++v) a.push_back(uint16_t(v));
    c.arr = std::move(a);
    c.kind = Kind::kArray;
  } else {
    c.bits.assign(kBitmapWords, 0);
    for (const Run& r : c.runs) set_range(c.bits.data(), r.start, r.end());
    c.kind = Kind::kBitmap;
  }
  release(c.runs);
}

// Number of maximal runs of consecutive values.
size_t count_runs(const Container& c) {
  switch (c.kind) {
    case Kind::kArray: {
      if (c.arr.empty()) return 0;
      size_t n = 1;
      for (size_t i = 1; i < c.arr.size(); ++i) n += c.arr[i] != c.arr[i - 1] + 1;
      return n;
    }
    case Kind::kBitmap: {
      // A run starts at every set bit whose predecessor bit is clear.
      size_t n = 0;
      uint64_t carry = 0;  // top bit of the previous word
      for (uint32_t i = 0; i < kBitmapWords; ++i) {
        const uint64_t w = c.bits[i];
        n += popcnt(w & ~((w << 1) | carry));
        carry = w >> 63;
      }
      return n;
    }
    case Kind::kRun:
      return c.runs.size();
  }
  return 0;
}

// Array/bitmap -> run encoding (nruns = count_runs(c), used to reserve).
void to_run(Container& c, size_t nruns) {
  std::vector<Run> runs;
  runs.reserve(nruns);
  if (c.kind == Kind::kArray) {
    RunWriter w{runs};
    for (uint16_t v : c.arr) w.add(v, v);
    release(c.arr);
  } else {
    // Find a run start with ctz, then its end with ctz of the complement,
    // possibly continuing through whole words of ones.
    const uint64_t* bits = c.bits.data();
    uint32_t i = 0;
    uint64_t cur = bits[0];
    for (;;) {
      while (cur == 0 && i + 1 < kBitmapWords) cur = bits[++i];
      if (cur == 0) break;
      const uint32_t start = (i << 6) | ctz64(cur);
      uint64_t ones = cur | (cur - 1);  // also fill the zeros below the run start
      while (ones == ~0ULL && i + 1 < kBitmapWords) ones = bits[++i];
      if (ones == ~0ULL) {  // run reaches the end of the container
        runs.push_back(Run{uint16_t(start), uint16_t(65535 - start)});
        break;
      }
      const uint32_t end = (i << 6) + ctz64(~ones);  // exclusive
      runs.push_back(Run{uint16_t(start), uint16_t(end - 1 - start)});
      cur = ones & (ones + 1);  // drop the trailing ones (this run)
    }
    release(c.bits);
  }
  c.runs = std::move(runs);
  c.kind = Kind::kRun;
}

// Restore the canonical array/bitmap encoding after the cardinality changed.
void fix_plain(Container& c) {
  if (c.kind == Kind::kBitmap && c.card <= kArrayMaxCard) {
    bitmap_to_array(c);
  } else if (c.kind == Kind::kArray && c.card > kArrayMaxCard) {
    array_to_bitmap(c);
  }
}

// Drop the run encoding once it stops being the smallest. (Adopting runs only
// happens in run_optimize() and add_range().)
void fix_run(Container& c) {
  if (c.kind == Kind::kRun && c.card > 0 && !run_is_best(c.runs.size(), c.card)) run_to_plain(c);
}

// Copy with a run encoding expanded to array/bitmap.
Container plain_copy(const Container& c) {
  Container r = c;
  if (r.kind == Kind::kRun) run_to_plain(r);
  return r;
}

// ===========================================================================
// 3. Single-container mutation and queries (arguments are low halves)

uint16_t c_min(const Container& c) {
  switch (c.kind) {
    case Kind::kArray: return c.arr.front();
    case Kind::kRun: return c.runs.front().start;
    case Kind::kBitmap:
      for (uint32_t i = 0; i < kBitmapWords; ++i)
        if (c.bits[i]) return uint16_t((i << 6) | ctz64(c.bits[i]));
      break;
  }
  return 0;
}

uint16_t c_max(const Container& c) {
  switch (c.kind) {
    case Kind::kArray: return c.arr.back();
    case Kind::kRun: return uint16_t(c.runs.back().end());
    case Kind::kBitmap:
      for (uint32_t i = kBitmapWords; i-- > 0;)
        if (c.bits[i]) return uint16_t((i << 6) | (63 - clz64(c.bits[i])));
      break;
  }
  return 0;
}

// Number of values <= x.
uint32_t c_rank(const Container& c, uint32_t x) {
  switch (c.kind) {
    case Kind::kArray: {
      const size_t i = lower_bound_u16(c.arr.data(), c.arr.size(), uint16_t(x));
      return uint32_t(i + (i < c.arr.size() && c.arr[i] == x));
    }
    case Kind::kBitmap: {
      uint32_t n = 0;
      const uint32_t w = x >> 6;
      for (uint32_t i = 0; i < w; ++i) n += popcnt(c.bits[i]);
      return n + popcnt(c.bits[w] & mask_upto(x));
    }
    case Kind::kRun: {
      uint32_t n = 0;
      for (const Run& r : c.runs) {
        if (r.end() < x) {
          n += uint32_t(r.len) + 1;
        } else {
          if (r.start <= x) n += x - r.start + 1;
          break;
        }
      }
      return n;
    }
  }
  return 0;
}

bool c_add(Container& c, uint16_t x) {
  switch (c.kind) {
    case Kind::kArray: {
      auto& a = c.arr;
      size_t i = a.size();
      if (!a.empty() && a.back() >= x) {  // not an append
        i = lower_bound_u16(a.data(), a.size(), x);
        if (a[i] == x) return false;
      }
      if (c.card == kArrayMaxCard) {  // the 4097th value: switch to a bitmap
        array_to_bitmap(c);
        set_bit(c.bits.data(), x);
      } else {
        a.insert(a.begin() + ptrdiff_t(i), x);
      }
      ++c.card;
      return true;
    }
    case Kind::kBitmap: {
      uint64_t& w = c.bits[x >> 6];
      const uint64_t m = 1ULL << (x & 63);
      if (w & m) return false;
      w |= m;
      ++c.card;
      return true;
    }
    case Kind::kRun: {
      auto& r = c.runs;
      const size_t n = r.size();
      const size_t i = run_search(r.data(), 0, n, x);  // first run ending at/after x
      if (i < n && r[i].start <= x) return false;      // already covered
      // x lies in the gap between r[i-1] and r[i].
      const bool join_prev = i > 0 && r[i - 1].end() + 1 == x;
      const bool join_next = i < n && uint32_t(x) + 1 == r[i].start;
      if (join_prev && join_next) {
        r[i - 1].len = uint16_t(r[i].end() - r[i - 1].start);
        r.erase(r.begin() + ptrdiff_t(i));
      } else if (join_prev) {
        ++r[i - 1].len;
      } else if (join_next) {
        --r[i].start;
        ++r[i].len;
      } else {
        r.insert(r.begin() + ptrdiff_t(i), Run{x, 0});
      }
      ++c.card;
      fix_run(c);
      return true;
    }
  }
  return false;
}

// May leave c empty (card == 0); the caller drops empty containers.
bool c_remove(Container& c, uint16_t x) {
  switch (c.kind) {
    case Kind::kArray: {
      auto& a = c.arr;
      const size_t i = lower_bound_u16(a.data(), a.size(), x);
      if (i == a.size() || a[i] != x) return false;
      a.erase(a.begin() + ptrdiff_t(i));
      --c.card;
      return true;
    }
    case Kind::kBitmap: {
      uint64_t& w = c.bits[x >> 6];
      const uint64_t m = 1ULL << (x & 63);
      if (!(w & m)) return false;
      w &= ~m;
      --c.card;
      fix_plain(c);
      return true;
    }
    case Kind::kRun: {
      auto& r = c.runs;
      const size_t i = run_search(r.data(), 0, r.size(), x);
      if (i == r.size() || r[i].start > x) return false;
      const uint32_t s = r[i].start, e = r[i].end();
      if (s == e) {
        r.erase(r.begin() + ptrdiff_t(i));
      } else if (x == s) {
        ++r[i].start;
        --r[i].len;
      } else if (x == e) {
        --r[i].len;
      } else {  // split [s, e] into [s, x-1] and [x+1, e]
        r[i].len = uint16_t(x - 1 - s);
        r.insert(r.begin() + ptrdiff_t(i) + 1, Run{uint16_t(x + 1), uint16_t(e - x - 1)});
      }
      --c.card;
      fix_run(c);
      return true;
    }
  }
  return false;
}

// Adds [lo, hi] (inclusive low halves). An empty container becomes a range.
void c_add_range(Container& c, uint32_t lo, uint32_t hi) {
  const uint32_t n = hi - lo + 1;
  if (c.card == 0 || n == kFullCard) {
    c = new_range(lo, hi);
    return;
  }
  switch (c.kind) {
    case Kind::kArray: {
      auto& a = c.arr;
      // a[i, j) are the existing values inside [lo, hi].
      const size_t i = lower_bound_u16(a.data(), a.size(), uint16_t(lo));
      const size_t j = hi >= 0xFFFF ? a.size() : lower_bound_u16(a.data(), a.size(), uint16_t(hi + 1));
      const uint32_t new_card = c.card - uint32_t(j - i) + n;
      if (new_card <= kArrayMaxCard) {
        a.insert(a.begin() + ptrdiff_t(j), n - (j - i), uint16_t(0));  // n >= j - i
        for (uint32_t k = 0; k < n; ++k) a[i + k] = uint16_t(lo + k);
      } else {
        array_to_bitmap(c);
        set_range(c.bits.data(), lo, hi);
      }
      c.card = new_card;
      return;
    }
    case Kind::kBitmap:
      c.card += n - popcount_range(c.bits.data(), lo, hi);
      set_range(c.bits.data(), lo, hi);
      return;
    case Kind::kRun: {
      auto& r = c.runs;
      // Runs [i, j) overlap or touch [lo, hi]; they collapse into one run.
      const size_t i = run_search(r.data(), 0, r.size(), lo == 0 ? 0 : lo - 1);
      const size_t j = size_t(
          std::upper_bound(r.begin() + ptrdiff_t(i), r.end(), hi + 1,
                           [](uint32_t v, const Run& run) { return v < run.start; }) -
          r.begin());
      uint32_t s = lo, e = hi;
      if (i < j) {
        s = std::min<uint32_t>(s, r[i].start);
        e = std::max(e, r[j - 1].end());
      }
      r.erase(r.begin() + ptrdiff_t(i), r.begin() + ptrdiff_t(j));
      r.insert(r.begin() + ptrdiff_t(i), Run{uint16_t(s), uint16_t(e - s)});
      uint32_t card = 0;
      for (const Run& x : r) card += uint32_t(x.len) + 1;
      c.card = card;
      fix_run(c);
      return;
    }
  }
}

// ===========================================================================
// 4. Pairwise container kernels
//
// Results are canonical: array/bitmap by cardinality, runs only when smallest.
// A result may be empty (card == 0); callers drop those.

constexpr Kind kA = Kind::kArray, kB = Kind::kBitmap, kR = Kind::kRun;
constexpr int pc(Kind a, Kind b) { return int(a) * 4 + int(b); }

enum class Mode { kWrite, kCount, kAny };

// Sorted-array intersection. kWrite stores into `out`, which may alias either
// input (the k-th output slot is never past the read position of either
// side). Gallops when one side is > 32x larger, otherwise merges linearly.
template <Mode M>
size_t intersect_u16(const uint16_t* a, size_t na, const uint16_t* b, size_t nb,
                     [[maybe_unused]] uint16_t* out) {
  if (na > nb) {
    std::swap(a, b);
    std::swap(na, nb);
  }
  size_t k = 0;
  if (nb > 32 * na) {
    size_t j = 0;
    for (size_t i = 0; i < na; ++i) {
      j = gallop_u16(b, j, nb, a[i]);
      if (j == nb) break;
      if (b[j] == a[i]) {
        if constexpr (M == Mode::kWrite) out[k] = a[i];
        if constexpr (M == Mode::kAny) return 1;
        ++k;
        ++j;
      }
    }
    return k;
  }
  size_t i = 0, j = 0;
  while (i < na && j < nb) {
    const uint16_t x = a[i], y = b[j];
    if (x == y) {
      if constexpr (M == Mode::kWrite) out[k] = x;
      if constexpr (M == Mode::kAny) return 1;
      ++k;
      ++i;
      ++j;
    } else {
      i += x < y;
      j += y < x;
    }
  }
  return k;
}

// Sorted-array difference a \ b; out may alias a.
size_t difference_u16(const uint16_t* a, size_t na, const uint16_t* b, size_t nb, uint16_t* out) {
  size_t k = 0, j = 0;
  const bool skewed = nb > 32 * na;
  for (size_t i = 0; i < na; ++i) {
    const uint16_t v = a[i];
    if (skewed) {
      j = gallop_u16(b, j, nb, v);
    } else {
      while (j < nb && b[j] < v) ++j;
    }
    out[k] = v;
    k += (j == nb || b[j] != v);
  }
  return k;
}

// Keeps the array values for which keep(v) holds (probes arrive in
// increasing order, so stateful predicates such as RunCursor work).
template <class Pred>
void filter_array(Container& c, Pred&& keep) {
  size_t k = 0;
  for (size_t i = 0, n = c.arr.size(); i < n; ++i) {
    const uint16_t v = c.arr[i];
    c.arr[k] = v;
    k += keep(v) ? 1 : 0;
  }
  c.arr.resize(k);
  c.card = uint32_t(k);
}

// Word-wise combination of two bitmaps. A popcount pass picks the output
// encoding first so the result is materialised exactly once.
template <class Op>
Container bitmap_combine(const uint64_t* x, const uint64_t* y, Op op) {
  uint32_t card = 0;
  for (uint32_t i = 0; i < kBitmapWords; ++i) card += popcnt(op(x[i], y[i]));
  Container r;
  r.card = card;
  if (card > kArrayMaxCard) {
    r.kind = Kind::kBitmap;
    r.bits.resize(kBitmapWords);
    for (uint32_t i = 0; i < kBitmapWords; ++i) r.bits[i] = op(x[i], y[i]);
  } else {
    r.arr.resize(card);
    size_t k = 0;
    for (uint32_t i = 0; i < kBitmapWords; ++i)
      for (uint64_t w = op(x[i], y[i]); w; w &= w - 1) r.arr[k++] = uint16_t((i << 6) | ctz64(w));
  }
  return r;
}

template <class Op>
void bitmap_combine_inplace(Container& a, const uint64_t* y, Op op) {
  uint64_t* x = a.bits.data();
  uint32_t card = 0;
  for (uint32_t i = 0; i < kBitmapWords; ++i) {
    x[i] = op(x[i], y[i]);
    card += popcnt(x[i]);
  }
  a.card = card;
  fix_plain(a);
}

constexpr auto kAndOp = [](uint64_t p, uint64_t q) { return p & q; };
constexpr auto kOrOp = [](uint64_t p, uint64_t q) { return p | q; };
constexpr auto kAndNotOp = [](uint64_t p, uint64_t q) { return p & ~q; };
constexpr auto kXorOp = [](uint64_t p, uint64_t q) { return p ^ q; };

// Bitmap-container updates that keep `card` exact without a full popcount.
void bitmap_add_array(Container& bm, const std::vector<uint16_t>& vals) {
  uint64_t* w = bm.bits.data();
  uint32_t card = bm.card;
  for (uint16_t v : vals) {
    const uint64_t m = 1ULL << (v & 63);
    card += (w[v >> 6] & m) == 0;
    w[v >> 6] |= m;
  }
  bm.card = card;
}
void bitmap_remove_array(Container& bm, const std::vector<uint16_t>& vals) {
  uint64_t* w = bm.bits.data();
  uint32_t card = bm.card;
  for (uint16_t v : vals) {
    const uint64_t m = 1ULL << (v & 63);
    card -= (w[v >> 6] & m) != 0;
    w[v >> 6] &= ~m;
  }
  bm.card = card;
}
void bitmap_flip_array(Container& bm, const std::vector<uint16_t>& vals) {
  uint64_t* w = bm.bits.data();
  uint32_t card = bm.card;
  for (uint16_t v : vals) {
    const uint64_t m = 1ULL << (v & 63);
    card = (w[v >> 6] & m) ? card - 1 : card + 1;
    w[v >> 6] ^= m;
  }
  bm.card = card;
}
void bitmap_add_runs(Container& bm, const std::vector<Run>& runs) {
  for (const Run& r : runs) {
    bm.card += uint32_t(r.len) + 1 - popcount_range(bm.bits.data(), r.start, r.end());
    set_range(bm.bits.data(), r.start, r.end());
  }
}
void bitmap_remove_runs(Container& bm, const std::vector<Run>& runs) {
  for (const Run& r : runs) {
    bm.card -= popcount_range(bm.bits.data(), r.start, r.end());
    clear_range(bm.bits.data(), r.start, r.end());
  }
}

// ---- AND ------------------------------------------------------------------

Container and_aa(const Container& a, const Container& b) {
  Container r;
  r.arr.resize(std::min(a.card, b.card));
  const size_t k = intersect_u16<Mode::kWrite>(a.arr.data(), a.arr.size(), b.arr.data(),
                                               b.arr.size(), r.arr.data());
  r.arr.resize(k);
  r.card = uint32_t(k);
  return r;
}

Container and_ab(const Container& arr, const Container& bm) {
  Container r = arr;
  filter_array(r, [w = bm.bits.data()](uint16_t v) { return test_bit(w, v); });
  return r;
}

Container and_ar(const Container& arr, const Container& run) {
  Container r = arr;
  RunCursor rc{run.runs.data(), run.runs.size()};
  filter_array(r, [&rc](uint16_t v) { return rc.contains(v); });
  return r;
}

Container and_br(const Container& bm, const Container& run) {
  if (is_full(run)) return bm;
  if (run.card <= kArrayMaxCard) {  // result fits an array: probe the run values
    Container r;
    r.arr.reserve(std::min(run.card, bm.card));
    for (const Run& x : run.runs)
      for (uint32_t v = x.start; v <= x.end(); ++v)
        if (test_bit(bm.bits.data(), v)) r.arr.push_back(uint16_t(v));
    r.card = uint32_t(r.arr.size());
    return r;
  }
  Container r = new_bitmap();
  for (const Run& x : run.runs) {
    const uint32_t a = x.start >> 6, b = x.end() >> 6;
    for (uint32_t i = a; i <= b; ++i) {
      uint64_t m = ~0ULL;
      if (i == a) m &= mask_from(x.start);
      if (i == b) m &= mask_upto(x.end());
      r.bits[i] |= bm.bits[i] & m;
    }
  }
  r.card = popcount_all(r.bits.data());
  fix_plain(r);
  return r;
}

Container and_rr(const Container& a, const Container& b) {
  Container r;
  r.kind = Kind::kRun;
  RunWriter w{r.runs};
  const auto &x = a.runs, &y = b.runs;
  size_t i = 0, j = 0;
  while (i < x.size() && j < y.size()) {
    const uint32_t s = std::max(x[i].start, y[j].start);
    const uint32_t e = std::min(x[i].end(), y[j].end());
    if (s <= e) w.add(s, e);
    if (x[i].end() < y[j].end()) {
      ++i;
    } else {
      ++j;
    }
  }
  r.card = w.card;
  fix_run(r);
  return r;
}

Container c_and(const Container& a, const Container& b) {
  switch (pc(a.kind, b.kind)) {
    case pc(kA, kA): return and_aa(a, b);
    case pc(kA, kB): return and_ab(a, b);
    case pc(kB, kA): return and_ab(b, a);
    case pc(kB, kB): return bitmap_combine(a.bits.data(), b.bits.data(), kAndOp);
    case pc(kA, kR): return and_ar(a, b);
    case pc(kR, kA): return and_ar(b, a);
    case pc(kB, kR): return and_br(a, b);
    case pc(kR, kB): return and_br(b, a);
    default: return and_rr(a, b);
  }
}

// |a & b| without materialising; in kAny mode returns 1 as soon as any common
// value is found (0 otherwise).
template <Mode M>
uint32_t count_ab(const Container& arr, const Container& bm) {
  uint32_t n = 0;
  const uint64_t* w = bm.bits.data();
  for (uint16_t v : arr.arr) {
    const bool hit = test_bit(w, v);
    if constexpr (M == Mode::kAny) {
      if (hit) return 1;
    }
    n += hit;
  }
  return n;
}

template <Mode M>
uint32_t count_ar(const Container& arr, const Container& run) {
  uint32_t n = 0;
  RunCursor rc{run.runs.data(), run.runs.size()};
  for (uint16_t v : arr.arr) {
    const bool hit = rc.contains(v);
    if constexpr (M == Mode::kAny) {
      if (hit) return 1;
    }
    n += hit;
    if (rc.j == rc.n) break;
  }
  return n;
}

template <Mode M>
uint32_t count_br(const Container& bm, const Container& run) {
  uint32_t n = 0;
  for (const Run& r : run.runs) {
    n += popcount_range(bm.bits.data(), r.start, r.end());
    if constexpr (M == Mode::kAny) {
      if (n) return 1;
    }
  }
  return n;
}

template <Mode M>
uint32_t count_rr(const Container& a, const Container& b) {
  uint32_t n = 0;
  const auto &x = a.runs, &y = b.runs;
  size_t i = 0, j = 0;
  while (i < x.size() && j < y.size()) {
    const uint32_t s = std::max(x[i].start, y[j].start);
    const uint32_t e = std::min(x[i].end(), y[j].end());
    if (s <= e) {
      if constexpr (M == Mode::kAny) return 1;
      n += e - s + 1;
    }
    if (x[i].end() < y[j].end()) {
      ++i;
    } else {
      ++j;
    }
  }
  return n;
}

template <Mode M>
uint32_t and_count(const Container& a, const Container& b) {
  static_assert(M != Mode::kWrite);
  switch (pc(a.kind, b.kind)) {
    case pc(kA, kA):
      return uint32_t(
          intersect_u16<M>(a.arr.data(), a.arr.size(), b.arr.data(), b.arr.size(), nullptr));
    case pc(kA, kB): return count_ab<M>(a, b);
    case pc(kB, kA): return count_ab<M>(b, a);
    case pc(kB, kB): {
      uint32_t n = 0;
      for (uint32_t i = 0; i < kBitmapWords; ++i) {
        const uint64_t w = a.bits[i] & b.bits[i];
        if constexpr (M == Mode::kAny) {
          if (w) return 1;
        }
        n += popcnt(w);
      }
      return n;
    }
    case pc(kA, kR): return count_ar<M>(a, b);
    case pc(kR, kA): return count_ar<M>(b, a);
    case pc(kB, kR): return count_br<M>(a, b);
    case pc(kR, kB): return count_br<M>(b, a);
    default: return count_rr<M>(a, b);
  }
}

// ---- OR -------------------------------------------------------------------

Container or_aa(const Container& a, const Container& b) {
  if (a.card + b.card <= kArrayMaxCard) {
    Container r;
    r.arr.resize(a.card + b.card);
    auto end = std::set_union(a.arr.begin(), a.arr.end(), b.arr.begin(), b.arr.end(), r.arr.begin());
    r.arr.resize(size_t(end - r.arr.begin()));
    r.card = uint32_t(r.arr.size());
    return r;
  }
  Container r = new_bitmap();
  for (uint16_t v : a.arr) set_bit(r.bits.data(), v);
  r.card = a.card;
  bitmap_add_array(r, b.arr);
  fix_plain(r);  // duplicates may leave it <= 4096
  return r;
}

Container or_ab(const Container& arr, const Container& bm) {
  Container r = bm;
  bitmap_add_array(r, arr.arr);
  return r;
}

Container or_br(const Container& bm, const Container& run) {
  if (is_full(run)) return run;
  Container r = bm;
  bitmap_add_runs(r, run.runs);
  return r;
}

Container or_ar(const Container& arr, const Container& run) {
  if (is_full(run)) return run;
  Container r;
  r.kind = Kind::kRun;
  RunWriter w{r.runs};
  const auto& v = arr.arr;
  const auto& x = run.runs;
  size_t i = 0, j = 0;
  while (i < x.size() || j < v.size()) {
    if (j == v.size() || (i < x.size() && x[i].start <= v[j])) {
      w.add(x[i].start, x[i].end());
      ++i;
    } else {
      w.add(v[j], v[j]);
      ++j;
    }
  }
  r.card = w.card;
  fix_run(r);
  return r;
}

Container or_rr(const Container& a, const Container& b) {
  if (is_full(a)) return a;
  if (is_full(b)) return b;
  Container r;
  r.kind = Kind::kRun;
  RunWriter w{r.runs};
  const auto &x = a.runs, &y = b.runs;
  size_t i = 0, j = 0;
  while (i < x.size() || j < y.size()) {
    if (j == y.size() || (i < x.size() && x[i].start <= y[j].start)) {
      w.add(x[i].start, x[i].end());
      ++i;
    } else {
      w.add(y[j].start, y[j].end());
      ++j;
    }
  }
  r.card = w.card;
  fix_run(r);
  return r;
}

Container c_or(const Container& a, const Container& b) {
  switch (pc(a.kind, b.kind)) {
    case pc(kA, kA): return or_aa(a, b);
    case pc(kA, kB): return or_ab(a, b);
    case pc(kB, kA): return or_ab(b, a);
    case pc(kB, kB): return bitmap_combine(a.bits.data(), b.bits.data(), kOrOp);
    case pc(kA, kR): return or_ar(a, b);
    case pc(kR, kA): return or_ar(b, a);
    case pc(kB, kR): return or_br(a, b);
    case pc(kR, kB): return or_br(b, a);
    default: return or_rr(a, b);
  }
}

// ---- ANDNOT ---------------------------------------------------------------

Container andnot_aa(const Container& a, const Container& b) {
  Container r = a;
  const size_t k = difference_u16(r.arr.data(), r.arr.size(), b.arr.data(), b.arr.size(), r.arr.data());
  r.arr.resize(k);
  r.card = uint32_t(k);
  return r;
}

Container andnot_ab(const Container& arr, const Container& bm) {
  Container r = arr;
  filter_array(r, [w = bm.bits.data()](uint16_t v) { return !test_bit(w, v); });
  return r;
}

Container andnot_ar(const Container& arr, const Container& run) {
  Container r = arr;
  RunCursor rc{run.runs.data(), run.runs.size()};
  filter_array(r, [&rc](uint16_t v) { return !rc.contains(v); });
  return r;
}

Container andnot_ba(const Container& bm, const Container& arr) {
  Container r = bm;
  bitmap_remove_array(r, arr.arr);
  fix_plain(r);
  return r;
}

Container andnot_br(const Container& bm, const Container& run) {
  Container r = bm;
  bitmap_remove_runs(r, run.runs);
  fix_plain(r);
  return r;
}

Container andnot_ra(const Container& run, const Container& arr) {
  Container r;
  r.kind = Kind::kRun;
  RunWriter w{r.runs};
  const auto& v = arr.arr;
  size_t j = 0;
  for (const Run& x : run.runs) {
    uint32_t cur = x.start;  // first value of x not yet emitted or removed
    const uint32_t e = x.end();
    j = gallop_u16(v.data(), j, v.size(), x.start);
    for (; j < v.size() && v[j] <= e; ++j) {
      if (v[j] > cur) w.add(cur, v[j] - 1u);
      cur = v[j] + 1u;
    }
    if (cur <= e) w.add(cur, e);
  }
  r.card = w.card;
  fix_run(r);
  return r;
}

Container andnot_rb(const Container& run, const Container& bm) {
  if (run.card <= kArrayMaxCard) {  // result fits an array: probe the run values
    Container r;
    r.arr.reserve(run.card);
    for (const Run& x : run.runs)
      for (uint32_t v = x.start; v <= x.end(); ++v)
        if (!test_bit(bm.bits.data(), v)) r.arr.push_back(uint16_t(v));
    r.card = uint32_t(r.arr.size());
    return r;
  }
  Container r = new_bitmap();
  for (const Run& x : run.runs) set_range(r.bits.data(), x.start, x.end());
  bitmap_combine_inplace(r, bm.bits.data(), kAndNotOp);
  return r;
}

Container andnot_rr(const Container& a, const Container& b) {
  Container r;
  r.kind = Kind::kRun;
  RunWriter w{r.runs};
  const auto& y = b.runs;
  size_t j = 0;
  for (const Run& x : a.runs) {
    uint32_t cur = x.start;
    const uint32_t e = x.end();
    j = gallop_run(y.data(), j, y.size(), cur);  // first b-run ending at/after cur
    while (j < y.size() && y[j].start <= e) {
      if (y[j].start > cur) w.add(cur, y[j].start - 1u);
      cur = y[j].end() + 1;
      if (y[j].end() >= e) break;  // y[j] may also cover the next a-run: keep j
      ++j;
    }
    if (cur <= e) w.add(cur, e);
  }
  r.card = w.card;
  fix_run(r);
  return r;
}

Container c_andnot(const Container& a, const Container& b) {
  switch (pc(a.kind, b.kind)) {
    case pc(kA, kA): return andnot_aa(a, b);
    case pc(kA, kB): return andnot_ab(a, b);
    case pc(kB, kA): return andnot_ba(a, b);
    case pc(kB, kB): return bitmap_combine(a.bits.data(), b.bits.data(), kAndNotOp);
    case pc(kA, kR): return andnot_ar(a, b);
    case pc(kR, kA): return andnot_ra(a, b);
    case pc(kB, kR): return andnot_br(a, b);
    case pc(kR, kB): return andnot_rb(a, b);
    default: return andnot_rr(a, b);
  }
}

// ---- XOR (runs are expanded first; xor is rare on the query path) ----------

Container c_xor(const Container& a, const Container& b) {
  if (a.kind == Kind::kRun) return c_xor(plain_copy(a), b);
  if (b.kind == Kind::kRun) return c_xor(a, plain_copy(b));
  switch (pc(a.kind, b.kind)) {
    case pc(kA, kA): {
      if (a.card + b.card <= kArrayMaxCard) {
        Container r;
        r.arr.resize(a.card + b.card);
        auto end = std::set_symmetric_difference(a.arr.begin(), a.arr.end(), b.arr.begin(),
                                                 b.arr.end(), r.arr.begin());
        r.arr.resize(size_t(end - r.arr.begin()));
        r.card = uint32_t(r.arr.size());
        return r;
      }
      Container r = new_bitmap();
      for (uint16_t v : a.arr) set_bit(r.bits.data(), v);
      r.card = a.card;
      bitmap_flip_array(r, b.arr);
      fix_plain(r);
      return r;
    }
    case pc(kA, kB):
    case pc(kB, kA): {
      const Container& arr = a.kind == Kind::kArray ? a : b;
      Container r = a.kind == Kind::kArray ? b : a;
      bitmap_flip_array(r, arr.arr);
      fix_plain(r);
      return r;
    }
    default:
      return bitmap_combine(a.bits.data(), b.bits.data(), kXorOp);
  }
}

// ---- in-place variants: avoid allocation where the left side can absorb
// the result; otherwise fall back to the out-of-place kernel. ---------------

void c_and_inplace(Container& a, const Container& b) {
  if (a.kind == Kind::kArray) {
    switch (b.kind) {
      case Kind::kArray: {
        const size_t k = intersect_u16<Mode::kWrite>(a.arr.data(), a.arr.size(), b.arr.data(),
                                                     b.arr.size(), a.arr.data());
        a.arr.resize(k);
        a.card = uint32_t(k);
        return;
      }
      case Kind::kBitmap:
        filter_array(a, [w = b.bits.data()](uint16_t v) { return test_bit(w, v); });
        return;
      case Kind::kRun: {
        RunCursor rc{b.runs.data(), b.runs.size()};
        filter_array(a, [&rc](uint16_t v) { return rc.contains(v); });
        return;
      }
    }
  } else if (a.kind == Kind::kBitmap) {
    if (b.kind == Kind::kBitmap) {
      bitmap_combine_inplace(a, b.bits.data(), kAndOp);
      return;
    }
    if (b.kind == Kind::kRun) {
      // Clear everything outside b's runs.
      uint32_t next = 0;  // first value not covered by an earlier run
      for (const Run& r : b.runs) {
        if (r.start > next) clear_range(a.bits.data(), next, r.start - 1u);
        next = r.end() + 1;
      }
      if (next <= 0xFFFF) clear_range(a.bits.data(), next, 0xFFFF);
      a.card = popcount_all(a.bits.data());
      fix_plain(a);
      return;
    }
  }
  a = c_and(a, b);
}

void c_or_inplace(Container& a, const Container& b) {
  if (is_full(a)) return;
  if (is_full(b)) {
    a = new_full();
    return;
  }
  if (a.kind == Kind::kBitmap) {
    switch (b.kind) {
      case Kind::kArray: bitmap_add_array(a, b.arr); return;
      case Kind::kBitmap: bitmap_combine_inplace(a, b.bits.data(), kOrOp); return;
      case Kind::kRun: bitmap_add_runs(a, b.runs); return;
    }
  }
  a = c_or(a, b);
}

void c_andnot_inplace(Container& a, const Container& b) {
  if (a.kind == Kind::kArray) {
    switch (b.kind) {
      case Kind::kArray: {
        const size_t k =
            difference_u16(a.arr.data(), a.arr.size(), b.arr.data(), b.arr.size(), a.arr.data());
        a.arr.resize(k);
        a.card = uint32_t(k);
        return;
      }
      case Kind::kBitmap:
        filter_array(a, [w = b.bits.data()](uint16_t v) { return !test_bit(w, v); });
        return;
      case Kind::kRun: {
        RunCursor rc{b.runs.data(), b.runs.size()};
        filter_array(a, [&rc](uint16_t v) { return !rc.contains(v); });
        return;
      }
    }
  } else if (a.kind == Kind::kBitmap) {
    switch (b.kind) {
      case Kind::kArray: bitmap_remove_array(a, b.arr); fix_plain(a); return;
      case Kind::kBitmap: bitmap_combine_inplace(a, b.bits.data(), kAndNotOp); return;
      case Kind::kRun: bitmap_remove_runs(a, b.runs); fix_plain(a); return;
    }
  }
  a = c_andnot(a, b);
}

bool c_equal(const Container& a, const Container& b) {
  if (a.card != b.card) return false;
  if (a.kind == b.kind) {
    switch (a.kind) {
      case Kind::kArray: return a.arr == b.arr;
      case Kind::kBitmap: return a.bits == b.bits;
      case Kind::kRun: return a.runs == b.runs;  // runs are canonical
    }
  }
  return and_count<Mode::kCount>(a, b) == a.card;
}

// One container of a k-way union.
struct KeyedContainer {
  uint16_t key;
  const Container* c;
};

Container union_of(const KeyedContainer* items, size_t n) {
  size_t array_total = 0;
  bool all_arrays = true;
  const Container* seed = nullptr;  // a bitmap we can copy instead of zeroing
  for (size_t i = 0; i < n; ++i) {
    const Container& c = *items[i].c;
    if (is_full(c)) return new_full();
    if (c.kind == Kind::kArray) {
      array_total += c.card;
    } else {
      all_arrays = false;
      if (c.kind == Kind::kBitmap && seed == nullptr) seed = &c;
    }
  }
  if (all_arrays && array_total <= 128) {  // tiny: concatenate and sort
    Container r;
    r.arr.reserve(array_total);
    for (size_t i = 0; i < n; ++i) r.arr.insert(r.arr.end(), items[i].c->arr.begin(), items[i].c->arr.end());
    std::sort(r.arr.begin(), r.arr.end());
    r.arr.erase(std::unique(r.arr.begin(), r.arr.end()), r.arr.end());
    r.card = uint32_t(r.arr.size());
    return r;
  }
  // Accumulate into one bitmap; cardinality is computed once at the end.
  Container r = seed ? *seed : new_bitmap();
  uint64_t* w = r.bits.data();
  for (size_t i = 0; i < n; ++i) {
    const Container& c = *items[i].c;
    if (&c == seed) continue;
    switch (c.kind) {
      case Kind::kArray:
        for (uint16_t v : c.arr) set_bit(w, v);
        break;
      case Kind::kBitmap:
        for (uint32_t k = 0; k < kBitmapWords; ++k) w[k] |= c.bits[k];
        break;
      case Kind::kRun:
        for (const Run& x : c.runs) set_range(w, x.start, x.end());
        break;
    }
  }
  r.card = popcount_all(w);
  fix_plain(r);
  return r;
}

}  // namespace

// ===========================================================================
// 5. RoaringBitmap

RoaringBitmap RoaringBitmap::from_sorted(const uint32_t* vals, size_t n) {
  RoaringBitmap r;
  if (n == 0) return r;
  for (size_t i = 1; i < n; ++i)
    DC_CHECK(vals[i] > vals[i - 1], "from_sorted: input must be strictly increasing");
  DC_CHECK(vals[n - 1] <= kMaxValue, kSentinelMsg);
  size_t i = 0;
  while (i < n) {
    const uint32_t key = vals[i] >> 16;
    // A container holds at most 65536 values, which bounds the search.
    const uint32_t* group_end =
        std::upper_bound(vals + i, vals + std::min(n, i + kFullCard), key << 16 | 0xFFFFu);
    const size_t cnt = size_t(group_end - (vals + i));
    Container c;
    c.card = uint32_t(cnt);
    if (cnt <= kArrayMaxCard) {
      c.arr.resize(cnt);
      for (size_t k = 0; k < cnt; ++k) c.arr[k] = uint16_t(vals[i + k]);
    } else {
      c.kind = Kind::kBitmap;
      c.bits.assign(kBitmapWords, 0);
      for (size_t k = 0; k < cnt; ++k) set_bit(c.bits.data(), uint16_t(vals[i + k]));
    }
    r.append(uint16_t(key), std::move(c));
    i += cnt;
  }
  return r;
}

RoaringBitmap RoaringBitmap::from_range(uint32_t lo, uint32_t hi) {
  RoaringBitmap r;
  r.add_range(lo, hi);
  return r;
}

void RoaringBitmap::clear() {
  keys_.clear();
  cs_.clear();
}

void RoaringBitmap::add(uint32_t x) {
  DC_CHECK(x <= kMaxValue, kSentinelMsg);
  const uint16_t key = uint16_t(x >> 16);
  const size_t n = keys_.size();
  size_t i;
  if (n > 0 && keys_[n - 1] <= key) {  // fast path: appending in key order
    i = keys_[n - 1] == key ? n - 1 : n;
  } else {
    i = lower_bound_u16(keys_.data(), n, key);
  }
  if (i == n || keys_[i] != key) {
    Container c;
    c.arr.push_back(uint16_t(x));
    c.card = 1;
    keys_.insert(keys_.begin() + ptrdiff_t(i), key);
    cs_.insert(cs_.begin() + ptrdiff_t(i), std::move(c));
    return;
  }
  c_add(cs_[i], uint16_t(x));
}

void RoaringBitmap::add_many(const uint32_t* vals, size_t n) {
  if (n == 0) return;
  // Validate everything first so a bad value leaves the bitmap untouched.
  for (size_t i = 0; i < n; ++i) DC_CHECK(vals[i] <= kMaxValue, "add_many: reserved sentinel value");
  bool sorted = true;
  for (size_t i = 1; i < n && sorted; ++i) sorted = vals[i] > vals[i - 1];
  if (!sorted && n < 64) {
    for (size_t i = 0; i < n; ++i) add(vals[i]);
    return;
  }
  RoaringBitmap batch;
  if (sorted) {
    batch = from_sorted(vals, n);
  } else {
    std::vector<uint32_t> tmp(vals, vals + n);
    std::sort(tmp.begin(), tmp.end());
    tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
    batch = from_sorted(tmp.data(), tmp.size());
  }
  if (empty()) {
    *this = std::move(batch);
  } else {
    or_inplace(batch);
  }
}

void RoaringBitmap::add_range(uint32_t lo, uint32_t hi) {
  DC_CHECK(lo <= hi, "add_range: lo must be <= hi");
  DC_CHECK(hi <= kMaxValue + 1, kSentinelMsg);
  if (lo == hi) return;
  const uint32_t last = hi - 1;
  const uint32_t k0 = lo >> 16, k1 = last >> 16;
  // Existing containers with keys in [k0, k1] are keys_[i0, i1).
  const size_t i0 = lower_bound_u16(keys_.data(), keys_.size(), uint16_t(k0));
  const size_t i1 = k1 == 0xFFFF ? keys_.size() : lower_bound_u16(keys_.data(), keys_.size(), uint16_t(k1 + 1));
  auto lo_of = [&](uint32_t k) { return k == k0 ? (lo & 0xFFFFu) : 0u; };
  auto hi_of = [&](uint32_t k) { return k == k1 ? (last & 0xFFFFu) : 0xFFFFu; };

  if (i1 - i0 == size_t(k1 - k0) + 1) {  // every container already exists
    for (size_t i = i0; i < i1; ++i) c_add_range(cs_[i], lo_of(keys_[i]), hi_of(keys_[i]));
    return;
  }
  // Build the [k0, k1] segment, then splice it in: O(#containers + range).
  std::vector<uint16_t> seg_keys;
  std::vector<Container> seg;
  seg_keys.reserve(k1 - k0 + 1);
  seg.reserve(k1 - k0 + 1);
  size_t i = i0;
  for (uint32_t k = k0; k <= k1; ++k) {
    seg_keys.push_back(uint16_t(k));
    if (i < i1 && keys_[i] == k) {
      seg.push_back(std::move(cs_[i++]));
      c_add_range(seg.back(), lo_of(k), hi_of(k));
    } else {
      seg.push_back(new_range(lo_of(k), hi_of(k)));
    }
  }
  keys_.erase(keys_.begin() + ptrdiff_t(i0), keys_.begin() + ptrdiff_t(i1));
  keys_.insert(keys_.begin() + ptrdiff_t(i0), seg_keys.begin(), seg_keys.end());
  cs_.erase(cs_.begin() + ptrdiff_t(i0), cs_.begin() + ptrdiff_t(i1));
  cs_.insert(cs_.begin() + ptrdiff_t(i0), std::make_move_iterator(seg.begin()),
             std::make_move_iterator(seg.end()));
}

bool RoaringBitmap::remove(uint32_t x) {
  const uint16_t key = uint16_t(x >> 16);
  const size_t i = lower_bound_u16(keys_.data(), keys_.size(), key);
  if (i == keys_.size() || keys_[i] != key) return false;
  if (!c_remove(cs_[i], uint16_t(x))) return false;
  if (cs_[i].card == 0) {
    keys_.erase(keys_.begin() + ptrdiff_t(i));
    cs_.erase(cs_.begin() + ptrdiff_t(i));
  }
  return true;
}

uint64_t RoaringBitmap::cardinality() const {
  uint64_t n = 0;
  for (const Container& c : cs_) n += c.card;
  return n;
}

uint32_t RoaringBitmap::minimum() const {
  if (empty()) return kNoMoreDocs;
  return uint32_t(keys_.front()) << 16 | c_min(cs_.front());
}

uint32_t RoaringBitmap::maximum() const {
  if (empty()) return kNoMoreDocs;
  return uint32_t(keys_.back()) << 16 | c_max(cs_.back());
}

uint64_t RoaringBitmap::rank(uint32_t x) const {
  const uint16_t key = uint16_t(x >> 16);
  uint64_t n = 0;
  for (size_t i = 0; i < keys_.size() && keys_[i] <= key; ++i)
    n += keys_[i] < key ? cs_[i].card : c_rank(cs_[i], x & 0xFFFFu);
  return n;
}

// ---- set algebra ------------------------------------------------------------
// Key walks gallop over the key arrays, so a small bitmap against a large one
// costs O(small * log(large)) key comparisons.

RoaringBitmap RoaringBitmap::and_(const RoaringBitmap& a, const RoaringBitmap& b) {
  RoaringBitmap r;
  const size_t na = a.keys_.size(), nb = b.keys_.size();
  size_t i = 0, j = 0;
  while (i < na && j < nb) {
    const uint16_t ka = a.keys_[i], kb = b.keys_[j];
    if (ka < kb) {
      i = gallop_u16(a.keys_.data(), i, na, kb);
    } else if (kb < ka) {
      j = gallop_u16(b.keys_.data(), j, nb, ka);
    } else {
      Container c = c_and(a.cs_[i], b.cs_[j]);
      if (c.card) r.append(ka, std::move(c));
      ++i;
      ++j;
    }
  }
  return r;
}

RoaringBitmap RoaringBitmap::or_(const RoaringBitmap& a, const RoaringBitmap& b) {
  RoaringBitmap r;
  const size_t na = a.keys_.size(), nb = b.keys_.size();
  r.keys_.reserve(na + nb);
  r.cs_.reserve(na + nb);
  size_t i = 0, j = 0;
  while (i < na && j < nb) {
    const uint16_t ka = a.keys_[i], kb = b.keys_[j];
    if (ka < kb) {
      r.append(ka, Container(a.cs_[i++]));
    } else if (kb < ka) {
      r.append(kb, Container(b.cs_[j++]));
    } else {
      r.append(ka, c_or(a.cs_[i++], b.cs_[j++]));
    }
  }
  for (; i < na; ++i) r.append(a.keys_[i], Container(a.cs_[i]));
  for (; j < nb; ++j) r.append(b.keys_[j], Container(b.cs_[j]));
  return r;
}

RoaringBitmap RoaringBitmap::andnot_(const RoaringBitmap& a, const RoaringBitmap& b) {
  RoaringBitmap r;
  const size_t na = a.keys_.size(), nb = b.keys_.size();
  r.keys_.reserve(na);
  r.cs_.reserve(na);
  size_t j = 0;
  for (size_t i = 0; i < na; ++i) {
    const uint16_t ka = a.keys_[i];
    j = gallop_u16(b.keys_.data(), j, nb, ka);
    if (j < nb && b.keys_[j] == ka) {
      Container c = c_andnot(a.cs_[i], b.cs_[j]);
      if (c.card) r.append(ka, std::move(c));
    } else {
      r.append(ka, Container(a.cs_[i]));
    }
  }
  return r;
}

RoaringBitmap RoaringBitmap::xor_(const RoaringBitmap& a, const RoaringBitmap& b) {
  RoaringBitmap r;
  const size_t na = a.keys_.size(), nb = b.keys_.size();
  size_t i = 0, j = 0;
  while (i < na && j < nb) {
    const uint16_t ka = a.keys_[i], kb = b.keys_[j];
    if (ka < kb) {
      r.append(ka, Container(a.cs_[i++]));
    } else if (kb < ka) {
      r.append(kb, Container(b.cs_[j++]));
    } else {
      Container c = c_xor(a.cs_[i++], b.cs_[j++]);
      if (c.card) r.append(ka, std::move(c));
    }
  }
  for (; i < na; ++i) r.append(a.keys_[i], Container(a.cs_[i]));
  for (; j < nb; ++j) r.append(b.keys_[j], Container(b.cs_[j]));
  return r;
}

RoaringBitmap RoaringBitmap::or_many(const std::vector<const RoaringBitmap*>& inputs) {
  size_t total = 0, live = 0;
  const RoaringBitmap* last = nullptr;
  for (const RoaringBitmap* bm : inputs) {
    if (bm == nullptr || bm->empty()) continue;
    total += bm->keys_.size();
    ++live;
    last = bm;
  }
  if (live == 0) return RoaringBitmap();
  if (live == 1) return *last;

  // Group all containers by key; each output container is then built once
  // (one bitmap accumulator per key) instead of k-1 pairwise unions.
  std::vector<KeyedContainer> items;
  items.reserve(total);
  for (const RoaringBitmap* bm : inputs) {
    if (bm == nullptr) continue;
    for (size_t i = 0; i < bm->keys_.size(); ++i) items.push_back({bm->keys_[i], &bm->cs_[i]});
  }
  std::sort(items.begin(), items.end(),
            [](const KeyedContainer& x, const KeyedContainer& y) { return x.key < y.key; });

  RoaringBitmap r;
  for (size_t g = 0; g < items.size();) {
    size_t h = g + 1;
    while (h < items.size() && items[h].key == items[g].key) ++h;
    if (h - g == 1) {
      r.append(items[g].key, Container(*items[g].c));
    } else {
      r.append(items[g].key, union_of(items.data() + g, h - g));
    }
    g = h;
  }
  return r;
}

void RoaringBitmap::and_inplace(const RoaringBitmap& o) {
  if (&o == this) return;
  const size_t na = keys_.size(), nb = o.keys_.size();
  size_t w = 0, j = 0;
  for (size_t i = 0; i < na && j < nb; ++i) {
    j = gallop_u16(o.keys_.data(), j, nb, keys_[i]);
    if (j == nb) break;
    if (o.keys_[j] != keys_[i]) continue;
    c_and_inplace(cs_[i], o.cs_[j++]);
    if (cs_[i].card == 0) continue;
    if (w != i) {
      keys_[w] = keys_[i];
      cs_[w] = std::move(cs_[i]);
    }
    ++w;
  }
  keys_.resize(w);
  cs_.resize(w);
}

void RoaringBitmap::or_inplace(const RoaringBitmap& o) {
  if (&o == this || o.empty()) return;
  if (empty()) {
    *this = o;
    return;
  }
  const size_t na = keys_.size(), nb = o.keys_.size();
  size_t missing = 0;
  for (size_t i = 0, j = 0; j < nb; ++j) {
    i = gallop_u16(keys_.data(), i, na, o.keys_[j]);
    missing += i == na || keys_[i] != o.keys_[j];
  }
  if (missing == 0) {  // every key of o exists here: merge containers in place
    for (size_t i = 0, j = 0; j < nb; ++j) {
      i = gallop_u16(keys_.data(), i, na, o.keys_[j]);
      c_or_inplace(cs_[i], o.cs_[j]);
    }
    return;
  }
  std::vector<uint16_t> keys;
  std::vector<Container> cs;
  keys.reserve(na + missing);
  cs.reserve(na + missing);
  size_t i = 0, j = 0;
  while (i < na || j < nb) {
    if (j == nb || (i < na && keys_[i] < o.keys_[j])) {
      keys.push_back(keys_[i]);
      cs.push_back(std::move(cs_[i++]));
    } else if (i == na || o.keys_[j] < keys_[i]) {
      keys.push_back(o.keys_[j]);
      cs.push_back(o.cs_[j++]);
    } else {
      c_or_inplace(cs_[i], o.cs_[j++]);
      keys.push_back(keys_[i]);
      cs.push_back(std::move(cs_[i++]));
    }
  }
  keys_.swap(keys);
  cs_.swap(cs);
}

void RoaringBitmap::andnot_inplace(const RoaringBitmap& o) {
  if (&o == this) {
    clear();
    return;
  }
  const size_t na = keys_.size(), nb = o.keys_.size();
  size_t w = 0, j = 0;
  for (size_t i = 0; i < na; ++i) {
    j = gallop_u16(o.keys_.data(), j, nb, keys_[i]);
    if (j < nb && o.keys_[j] == keys_[i]) {
      c_andnot_inplace(cs_[i], o.cs_[j++]);
      if (cs_[i].card == 0) continue;
    }
    if (w != i) {
      keys_[w] = keys_[i];
      cs_[w] = std::move(cs_[i]);
    }
    ++w;
  }
  keys_.resize(w);
  cs_.resize(w);
}

uint64_t RoaringBitmap::and_cardinality(const RoaringBitmap& o) const {
  uint64_t n = 0;
  const size_t na = keys_.size(), nb = o.keys_.size();
  size_t i = 0, j = 0;
  while (i < na && j < nb) {
    const uint16_t ka = keys_[i], kb = o.keys_[j];
    if (ka < kb) {
      i = gallop_u16(keys_.data(), i, na, kb);
    } else if (kb < ka) {
      j = gallop_u16(o.keys_.data(), j, nb, ka);
    } else {
      n += and_count<Mode::kCount>(cs_[i++], o.cs_[j++]);
    }
  }
  return n;
}

bool RoaringBitmap::intersects(const RoaringBitmap& o) const {
  const size_t na = keys_.size(), nb = o.keys_.size();
  size_t i = 0, j = 0;
  while (i < na && j < nb) {
    const uint16_t ka = keys_[i], kb = o.keys_[j];
    if (ka < kb) {
      i = gallop_u16(keys_.data(), i, na, kb);
    } else if (kb < ka) {
      j = gallop_u16(o.keys_.data(), j, nb, ka);
    } else if (and_count<Mode::kAny>(cs_[i++], o.cs_[j++])) {
      return true;
    }
  }
  return false;
}

// ---- maintenance / enumeration ----------------------------------------------

void RoaringBitmap::run_optimize() {
  for (Container& c : cs_) {
    const size_t nr = count_runs(c);
    const bool want_run = run_is_best(nr, c.card);
    if (want_run && c.kind != Kind::kRun) {
      to_run(c, nr);
    } else if (!want_run && c.kind == Kind::kRun) {
      run_to_plain(c);
    }
  }
}

void RoaringBitmap::shrink_to_fit() {
  keys_.shrink_to_fit();
  cs_.shrink_to_fit();
  for (Container& c : cs_) {
    c.arr.shrink_to_fit();
    c.bits.shrink_to_fit();
    c.runs.shrink_to_fit();
  }
}

std::vector<uint32_t> RoaringBitmap::to_vector() const {
  std::vector<uint32_t> out(cardinality());
  uint32_t* p = out.data();
  for_each([&p](uint32_t v) { *p++ = v; });
  return out;
}

size_t RoaringBitmap::memory_bytes() const {
  size_t n = sizeof(*this) + keys_.capacity() * sizeof(uint16_t) + cs_.capacity() * sizeof(Container);
  for (const Container& c : cs_)
    n += c.arr.capacity() * sizeof(uint16_t) + c.bits.capacity() * sizeof(uint64_t) +
         c.runs.capacity() * sizeof(Run);
  return n;
}

bool RoaringBitmap::operator==(const RoaringBitmap& o) const {
  if (keys_ != o.keys_) return false;
  for (size_t i = 0; i < cs_.size(); ++i)
    if (!c_equal(cs_[i], o.cs_[i])) return false;
  return true;
}

RoaringBitmap::ContainerStats RoaringBitmap::container_stats() const {
  ContainerStats s;
  for (const Container& c : cs_) {
    switch (c.kind) {
      case Kind::kArray: ++s.arrays; break;
      case Kind::kBitmap: ++s.bitmaps; break;
      case Kind::kRun: ++s.runs; break;
    }
  }
  return s;
}

// ===========================================================================
// 6. Iterator

RoaringBitmap::Iterator::Iterator(const RoaringBitmap& bm)
    : keys_(bm.keys_.data()), cs_(bm.cs_.data()), nc_(bm.keys_.size()), cost_(bm.cardinality()) {}

void RoaringBitmap::Iterator::load(size_t ci) {
  const Container& c = cs_[ci];
  ci_ = ci;
  hi_ = uint32_t(keys_[ci]) << 16;
  kind_ = c.kind;
  pos_ = 0;
  arr_ = c.arr.data();
  words_ = c.bits.data();
  runs_ = c.runs.data();
  n_ = uint32_t(c.kind == Kind::kRun ? c.runs.size() : c.arr.size());
}

uint32_t RoaringBitmap::Iterator::enter(size_t ci) {
  if (ci >= nc_) return cur_ = kNoMoreDocs;
  load(ci);
  switch (kind_) {
    case Kind::kArray:
      return cur_ = hi_ | arr_[0];
    case Kind::kRun:
      run_end_ = runs_[0].end();
      return cur_ = hi_ | runs_[0].start;
    case Kind::kBitmap:
      seek_low(0);  // containers are never empty, so this always succeeds
      return cur_;
  }
  return cur_;
}

bool RoaringBitmap::Iterator::seek_low(uint32_t low) {
  switch (kind_) {
    case Kind::kArray:
      pos_ = uint32_t(gallop_u16(arr_, pos_, n_, uint16_t(low)));
      if (pos_ >= n_) return false;
      cur_ = hi_ | arr_[pos_];
      return true;
    case Kind::kRun:
      pos_ = uint32_t(gallop_run(runs_, pos_, n_, low));
      if (pos_ >= n_) return false;
      run_end_ = runs_[pos_].end();
      cur_ = hi_ | std::max<uint32_t>(low, runs_[pos_].start);
      return true;
    case Kind::kBitmap: {
      uint32_t w = low >> 6;
      uint64_t word = words_[w] & mask_from(low);
      while (word == 0) {
        if (++w == kBitmapWords) return false;
        word = words_[w];
      }
      pos_ = w;
      cur_ = hi_ | (w << 6) | ctz64(word);
      word_ = word & (word - 1);
      return true;
    }
  }
  return false;
}

uint32_t RoaringBitmap::Iterator::advance(uint32_t target) {
  const bool positioned = cur_ != kUnpositioned;
  if (positioned && target <= cur_) return cur_;  // includes the exhausted state
  if (DC_UNLIKELY(target > kMaxValue)) return cur_ = kNoMoreDocs;
  const uint16_t tkey = uint16_t(target >> 16);
  if (!positioned || tkey != (hi_ >> 16)) {
    // Binary search the keys after the current container.
    const size_t from = positioned ? ci_ + 1 : 0;
    const size_t ci = from + lower_bound_u16(keys_ + from, nc_ - from, tkey);
    if (ci == nc_) return cur_ = kNoMoreDocs;
    if (keys_[ci] != tkey) return enter(ci);  // target falls in a gap between containers
    load(ci);
  }
  if (seek_low(target & 0xFFFFu)) return cur_;
  return enter(ci_ + 1);
}

// ===========================================================================
// 7. Serialization

size_t RoaringBitmap::serialized_size() const {
  size_t n = 8 + 5 * keys_.size();
  for (const Container& c : cs_) n += payload_bytes(c);
  return n;
}

void RoaringBitmap::serialize(ByteWriter& out) const {
  out.reserve(out.size() + serialized_size());
  out.put_u32(kMagic);
  out.put_u32(uint32_t(keys_.size()));
  for (size_t i = 0; i < keys_.size(); ++i) {
    out.put_u16(keys_[i]);
    out.put_u8(uint8_t(cs_[i].kind));
    out.put_u16(uint16_t(cs_[i].card - 1));
  }
  for (const Container& c : cs_) {
    switch (c.kind) {
      case Kind::kArray:
        out.put_bytes(c.arr.data(), c.arr.size() * sizeof(uint16_t));
        break;
      case Kind::kBitmap:
        out.put_bytes(c.bits.data(), kBitmapBytes);
        break;
      case Kind::kRun:
        out.put_u16(uint16_t(c.runs.size()));
        out.put_bytes(c.runs.data(), c.runs.size() * sizeof(Run));
        break;
    }
  }
}

RoaringBitmap RoaringBitmap::deserialize(ByteReader& in) {
  DC_CHECK(in.get_u32() == kMagic, "roaring: bad magic");
  const uint32_t n = in.get_u32();
  DC_CHECK(n <= 65536, "roaring: too many containers");
  DC_CHECK(in.remaining() >= size_t(n) * 5, "roaring: truncated container headers");

  RoaringBitmap r;
  r.keys_.resize(n);
  r.cs_.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    const uint16_t key = in.get_u16();
    const uint8_t kind = in.get_u8();
    const uint32_t card = uint32_t(in.get_u16()) + 1;
    DC_CHECK(i == 0 || key > r.keys_[i - 1], "roaring: container keys not strictly increasing");
    DC_CHECK(kind >= uint8_t(Kind::kArray) && kind <= uint8_t(Kind::kRun), "roaring: bad container type");
    if (kind == uint8_t(Kind::kArray)) DC_CHECK(card <= kArrayMaxCard, "roaring: oversized array container");
    if (kind == uint8_t(Kind::kBitmap)) DC_CHECK(card > kArrayMaxCard, "roaring: undersized bitmap container");
    r.keys_[i] = key;
    r.cs_[i].kind = Kind(kind);
    r.cs_[i].card = card;
  }

  for (uint32_t i = 0; i < n; ++i) {
    Container& c = r.cs_[i];
    switch (c.kind) {
      case Kind::kArray: {
        const uint8_t* p = in.get_bytes(size_t(c.card) * sizeof(uint16_t));
        c.arr.resize(c.card);
        std::memcpy(c.arr.data(), p, size_t(c.card) * sizeof(uint16_t));
        for (size_t k = 1; k < c.arr.size(); ++k)
          DC_CHECK(c.arr[k] > c.arr[k - 1], "roaring: array values not strictly increasing");
        break;
      }
      case Kind::kBitmap: {
        const uint8_t* p = in.get_bytes(kBitmapBytes);
        c.bits.resize(kBitmapWords);
        std::memcpy(c.bits.data(), p, kBitmapBytes);
        DC_CHECK(popcount_all(c.bits.data()) == c.card, "roaring: bitmap cardinality mismatch");
        break;
      }
      case Kind::kRun: {
        const uint32_t nr = in.get_u16();
        DC_CHECK(nr >= 1 && nr <= 32768, "roaring: bad run count");
        const uint8_t* p = in.get_bytes(size_t(nr) * sizeof(Run));
        c.runs.resize(nr);
        std::memcpy(c.runs.data(), p, size_t(nr) * sizeof(Run));
        uint64_t total = 0;
        for (size_t k = 0; k < nr; ++k) {
          const Run& x = c.runs[k];
          DC_CHECK(x.end() <= 0xFFFF, "roaring: run overflows its container");
          // Canonical runs are sorted with at least one gap value between them.
          DC_CHECK(k == 0 || x.start > c.runs[k - 1].end() + 1, "roaring: runs overlap or touch");
          total += uint64_t(x.len) + 1;
        }
        DC_CHECK(total == c.card, "roaring: run cardinality mismatch");
        break;
      }
    }
  }
  if (n > 0 && r.keys_.back() == 0xFFFF)
    DC_CHECK(c_max(r.cs_.back()) <= (kMaxValue & 0xFFFFu), "roaring: contains a reserved sentinel value");
  return r;
}

}  // namespace dc
