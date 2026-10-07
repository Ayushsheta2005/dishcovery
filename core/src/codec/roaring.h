// Roaring bitmap: a compressed set of uint32 document ids with fast set
// algebra and a skip-capable cursor.
//
// Used for filter fields (type / diet / price_tier / cuisine / chain), geo-cell
// -> deliverable-docs sets, tombstones, and materialised filter sets that drive
// filter-aware retrieval through Iterator::advance().
//
// Layout
// ------
// A value is split into its high 16 bits (the container key) and low 16 bits.
// Every distinct key owns one container holding the low halves in one of three
// encodings:
//   array  - sorted uint16 values; used while cardinality <= 4096 (<= 8 KB)
//   bitmap - 1024 x uint64 words (8 KB) with a cached cardinality; > 4096
//   run    - sorted, non-overlapping, non-adjacent (start, length-1) pairs;
//            produced by add_range() and run_optimize() when it is the
//            strictly smallest encoding
// Array and bitmap containers are always canonical by cardinality (array iff
// card <= 4096). Keys and containers live in two parallel vectors sorted by
// key, so the key array stays dense for binary search. Containers are never
// empty.
//
// Reserved values
// ---------------
// 0xFFFFFFFF (kNoMoreDocs) and 0xFFFFFFFE (kUnpositioned) are iterator
// sentinels and must never be inserted: add / add_many / add_range /
// from_sorted / from_range DC_CHECK this, and deserialize rejects them. The
// largest storable value is kMaxValue = 0xFFFFFFFD.
//
// Thread safety: const methods may run concurrently; mutation needs exclusive
// access. An Iterator borrows the bitmap, which must outlive it and must not be
// modified (or moved) while it is in use.
//
// Serialized format (little-endian, see serialize()):
//   u32 magic "RBM1" | u32 container count
//   per container: u16 key | u8 type (1 array, 2 bitmap, 3 run) | u16 card-1
//   payloads in container order:
//     array : card x u16 values
//     bitmap: 1024 x u64 words
//     run   : u16 run count, then count x (u16 start, u16 length-1)
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "util/bytes.h"
#include "util/common.h"

namespace dc {

namespace roaring_detail {

enum class Kind : uint8_t { kArray = 1, kBitmap = 2, kRun = 3 };

inline constexpr uint32_t kArrayMaxCard = 4096;  // array containers hold at most this many
inline constexpr uint32_t kBitmapWords = 1024;   // 65536 bits

// Consecutive values [start, start + len]; storing length-1 lets one run cover
// all 65536 values of a container.
struct Run {
  uint16_t start;
  uint16_t len;
  uint32_t end() const { return uint32_t(start) + len; }
  bool operator==(const Run&) const = default;
};

// Exactly one of arr / bits / runs is populated, as selected by `kind`.
struct Container {
  Kind kind = Kind::kArray;
  uint32_t card = 0;            // number of values, 1..65536
  std::vector<uint16_t> arr;    // kArray: sorted low halves
  std::vector<uint64_t> bits;   // kBitmap: kBitmapWords words
  std::vector<Run> runs;        // kRun: canonical runs
};

// First index i in [0, n) with a[i] >= x, or n. Branch-free binary search.
inline size_t lower_bound_u16(const uint16_t* a, size_t n, uint16_t x) {
  if (n == 0) return 0;
  const uint16_t* base = a;
  while (n > 1) {
    const size_t half = n >> 1;
    base = (base[half] < x) ? base + half : base;
    n -= half;
  }
  return size_t(base - a) + (*base < x);
}

// First run index i in [from, n) with r[i].end() >= x, or n.
inline size_t run_search(const Run* r, size_t from, size_t n, uint32_t x) {
  size_t lo = from, len = n - from;
  while (len > 0) {
    const size_t half = len >> 1;
    if (r[lo + half].end() < x) {
      lo += half + 1;
      len -= half + 1;
    } else {
      len = half;
    }
  }
  return lo;
}

inline bool container_contains(const Container& c, uint16_t x) {
  switch (c.kind) {
    case Kind::kArray: {
      const size_t i = lower_bound_u16(c.arr.data(), c.arr.size(), x);
      return i < c.arr.size() && c.arr[i] == x;
    }
    case Kind::kBitmap:
      return (c.bits[x >> 6] >> (x & 63)) & 1;
    case Kind::kRun: {
      const size_t i = run_search(c.runs.data(), 0, c.runs.size(), x);
      return i < c.runs.size() && c.runs[i].start <= x;
    }
  }
  return false;
}

}  // namespace roaring_detail

class RoaringBitmap {
 public:
  // Largest insertable value; the two values above it are iterator sentinels.
  static constexpr uint32_t kMaxValue = kUnpositioned - 1;  // 0xFFFFFFFD

  RoaringBitmap() = default;

  // Bulk build from strictly increasing values (DC_CHECKed).
  static RoaringBitmap from_sorted(const uint32_t* vals, size_t n);
  // All values in [lo, hi). Requires lo <= hi <= kMaxValue + 1.
  static RoaringBitmap from_range(uint32_t lo, uint32_t hi);

  // --- mutation -----------------------------------------------------------
  void add(uint32_t x);                          // any order; x <= kMaxValue
  void add_many(const uint32_t* vals, size_t n);  // any order, duplicates ok
  void add_range(uint32_t lo, uint32_t hi);       // [lo, hi); hi <= kMaxValue + 1
  bool remove(uint32_t x);                        // true if x was present
  void clear();

  // --- queries ------------------------------------------------------------
  bool contains(uint32_t x) const;
  uint64_t cardinality() const;
  bool empty() const { return keys_.empty(); }
  uint32_t minimum() const;           // kNoMoreDocs if empty
  uint32_t maximum() const;           // kNoMoreDocs if empty
  uint64_t rank(uint32_t x) const;    // number of values <= x

  // --- set algebra --------------------------------------------------------
  static RoaringBitmap and_(const RoaringBitmap& a, const RoaringBitmap& b);
  static RoaringBitmap or_(const RoaringBitmap& a, const RoaringBitmap& b);
  static RoaringBitmap andnot_(const RoaringBitmap& a, const RoaringBitmap& b);  // a \ b
  static RoaringBitmap xor_(const RoaringBitmap& a, const RoaringBitmap& b);
  // k-way union: each output container is built once (null inputs are skipped).
  static RoaringBitmap or_many(const std::vector<const RoaringBitmap*>& inputs);
  void and_inplace(const RoaringBitmap& o);
  void or_inplace(const RoaringBitmap& o);
  void andnot_inplace(const RoaringBitmap& o);
  uint64_t and_cardinality(const RoaringBitmap& o) const;  // |this & o|, no allocation
  bool intersects(const RoaringBitmap& o) const;

  // --- maintenance --------------------------------------------------------
  // Re-encode every container as the smallest of array / bitmap / run.
  void run_optimize();
  void shrink_to_fit();

  // --- enumeration --------------------------------------------------------
  std::vector<uint32_t> to_vector() const;
  template <class F>
  void for_each(F&& f) const;  // f(uint32_t) in increasing order

  // --- serialization ------------------------------------------------------
  size_t serialized_size() const;  // exact number of bytes serialize() appends
  void serialize(ByteWriter& out) const;
  // Validates structure and canonical form; throws DcError on corruption.
  static RoaringBitmap deserialize(ByteReader& in);

  size_t memory_bytes() const;  // approximate heap + object footprint
  // Set equality (independent of container encodings).
  bool operator==(const RoaringBitmap& o) const;

  // Introspection (tests, stats).
  struct ContainerStats {
    size_t arrays = 0, bitmaps = 0, runs = 0;
  };
  ContainerStats container_stats() const;
  size_t container_count() const { return keys_.size(); }

  // Forward-only cursor used as a document iterator in query execution.
  class Iterator {
   public:
    explicit Iterator(const RoaringBitmap& bm);

    // Current value: kUnpositioned before the first next()/advance(),
    // kNoMoreDocs once exhausted.
    uint32_t value() const { return cur_; }
    // Move to the next value and return it (kNoMoreDocs at the end).
    inline uint32_t next();
    // Move to the first value >= target and return it (kNoMoreDocs if none).
    // Intended for target > value(); if target <= value() the cursor does not
    // move and value() is returned. O(log #containers + log container size).
    uint32_t advance(uint32_t target);
    // Number of values (for iterator cost ordering).
    uint64_t cost() const { return cost_; }

   private:
    void load(size_t ci);         // point at container ci without positioning
    uint32_t enter(size_t ci);    // position at the first value of container ci
    bool seek_low(uint32_t low);  // first value >= low within the current container

    const uint16_t* keys_ = nullptr;
    const roaring_detail::Container* cs_ = nullptr;
    size_t nc_ = 0;  // number of containers
    size_t ci_ = 0;  // current container index
    uint32_t hi_ = 0;  // current key << 16
    roaring_detail::Kind kind_ = roaring_detail::Kind::kArray;
    const uint16_t* arr_ = nullptr;
    const uint64_t* words_ = nullptr;
    const roaring_detail::Run* runs_ = nullptr;
    uint32_t n_ = 0;        // array length / run count
    uint32_t pos_ = 0;      // array index / run index / bitmap word index
    uint32_t run_end_ = 0;  // run: inclusive end (low half) of runs_[pos_]
    uint64_t word_ = 0;     // bitmap: bits of words_[pos_] above the current value
    uint32_t cur_ = kUnpositioned;
    uint64_t cost_ = 0;
  };

 private:
  using Container = roaring_detail::Container;

  void append(uint16_t key, Container&& c) {
    keys_.push_back(key);
    cs_.push_back(std::move(c));
  }

  std::vector<uint16_t> keys_;  // sorted, unique
  std::vector<Container> cs_;   // cs_[i] holds the values with high half keys_[i]
};

// ---------------------------------------------------------------------------
// Inline hot paths.

inline bool RoaringBitmap::contains(uint32_t x) const {
  const uint16_t key = uint16_t(x >> 16);
  const size_t n = keys_.size();
  const size_t i = roaring_detail::lower_bound_u16(keys_.data(), n, key);
  return i < n && keys_[i] == key && roaring_detail::container_contains(cs_[i], uint16_t(x));
}

template <class F>
void RoaringBitmap::for_each(F&& f) const {
  using roaring_detail::Kind;
  for (size_t i = 0; i < keys_.size(); ++i) {
    const uint32_t hi = uint32_t(keys_[i]) << 16;
    const Container& c = cs_[i];
    switch (c.kind) {
      case Kind::kArray:
        for (uint16_t v : c.arr) f(hi | v);
        break;
      case Kind::kBitmap:
        for (uint32_t w = 0; w < roaring_detail::kBitmapWords; ++w) {
          for (uint64_t word = c.bits[w]; word; word &= word - 1)
            f(hi | (w << 6) | uint32_t(__builtin_ctzll(word)));
        }
        break;
      case Kind::kRun:
        for (const auto& r : c.runs) {
          const uint32_t end = hi | r.end();
          for (uint32_t v = hi | r.start; v <= end; ++v) f(v);  // end <= kMaxValue: no wrap
        }
        break;
    }
  }
}

inline uint32_t RoaringBitmap::Iterator::next() {
  using roaring_detail::Kind;
  if (DC_LIKELY(cur_ < kUnpositioned)) {
    switch (kind_) {
      case Kind::kArray:
        if (++pos_ < n_) return cur_ = hi_ | arr_[pos_];
        break;
      case Kind::kBitmap:
        if (word_ == 0) {
          while (++pos_ < roaring_detail::kBitmapWords && (word_ = words_[pos_]) == 0) {
          }
          if (pos_ >= roaring_detail::kBitmapWords) break;
        }
        cur_ = hi_ | (pos_ << 6) | uint32_t(__builtin_ctzll(word_));
        word_ &= word_ - 1;
        return cur_;
      case Kind::kRun:
        if ((cur_ & 0xFFFFu) < run_end_) return ++cur_;
        if (++pos_ < n_) {
          run_end_ = runs_[pos_].end();
          return cur_ = hi_ | runs_[pos_].start;
        }
        break;
    }
    return enter(ci_ + 1);
  }
  return cur_ == kUnpositioned ? enter(0) : kNoMoreDocs;
}

}  // namespace dc
