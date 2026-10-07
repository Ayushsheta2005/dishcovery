// Roaring bitmap tests: edge cases plus randomized differential tests against
// std::set / sorted std::vector references across sparse, dense, clustered
// and run-heavy distributions.

#include "codec/roaring.h"

#include <algorithm>
#include <iterator>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "testing.h"

using dc::ByteReader;
using dc::ByteWriter;
using dc::kNoMoreDocs;
using dc::kUnpositioned;
using dc::RoaringBitmap;

namespace {

using Vec = std::vector<uint32_t>;
using Rng = std::mt19937_64;
constexpr uint32_t kMax = RoaringBitmap::kMaxValue;  // 0xFFFFFFFD

uint32_t rnd(Rng& rng, uint32_t lo, uint32_t hi) {  // inclusive
  return std::uniform_int_distribution<uint32_t>(lo, hi)(rng);
}

Vec sorted_unique(Vec v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
}

std::vector<uint8_t> bytes_of(const RoaringBitmap& bm) {
  ByteWriter w;
  bm.serialize(w);
  return w.release();
}

RoaringBitmap roundtrip(const RoaringBitmap& bm) {
  ByteWriter w;
  bm.serialize(w);
  EXPECT_EQ(w.size(), bm.serialized_size());
  ByteReader r(w.buffer());
  RoaringBitmap back = RoaringBitmap::deserialize(r);  // validates canonical form
  EXPECT_TRUE(r.eof());
  return back;
}

// Full comparison against a sorted, unique reference.
void expect_matches(const RoaringBitmap& bm, const Vec& ref) {
  ASSERT_EQ(bm.cardinality(), uint64_t(ref.size()));
  ASSERT_TRUE(bm.to_vector() == ref);
  EXPECT_EQ(bm.empty(), ref.empty());
  EXPECT_EQ(bm.minimum(), ref.empty() ? kNoMoreDocs : ref.front());
  EXPECT_EQ(bm.maximum(), ref.empty() ? kNoMoreDocs : ref.back());
  const RoaringBitmap back = roundtrip(bm);
  EXPECT_TRUE(back == bm);
  ASSERT_TRUE(back.to_vector() == ref);
}

// Point queries (contains / rank) at values in and around the reference.
void expect_point_queries(Rng& rng, const RoaringBitmap& bm, const Vec& ref, int probes) {
  for (int p = 0; p < probes; ++p) {
    uint32_t x;
    switch (rnd(rng, 0, 3)) {
      case 0: x = rnd(rng, 0, 0xFFFFFFFFu); break;
      case 1:
        x = ref.empty() ? 0 : ref[rnd(rng, 0, uint32_t(ref.size() - 1))];
        break;
      case 2:
        x = ref.empty() ? 1 : ref[rnd(rng, 0, uint32_t(ref.size() - 1))] + 1;
        break;
      default:
        x = ref.empty() ? 0 : ref[rnd(rng, 0, uint32_t(ref.size() - 1))] - 1;
        break;
    }
    const bool in = std::binary_search(ref.begin(), ref.end(), x);
    ASSERT_EQ(bm.contains(x), in);
    const uint64_t rank = uint64_t(std::upper_bound(ref.begin(), ref.end(), x) - ref.begin());
    ASSERT_EQ(bm.rank(x), rank);
  }
}

// ---------------------------------------------------------------------------
// Random generators. A bitmap is generated container by container, each with
// its own style, so one bitmap mixes array / bitmap / run containers and two
// bitmaps over the same keys exercise every container-type pair.

enum Style { kSparse, kMediumArray, kNearThreshold, kDense, kRuns, kFull, kRunsWithNoise, kStyles };

struct Spec {
  Vec points;                                         // single values
  std::vector<std::pair<uint32_t, uint32_t>> ranges;  // [lo, hi)
};

void gen_container(Rng& rng, uint32_t key, int style, Spec& s) {
  const uint32_t base = key << 16;
  const uint32_t max_low = key == 0xFFFF ? (kMax & 0xFFFF) : 0xFFFF;
  auto pt = [&](uint32_t low) {
    if (low <= max_low) s.points.push_back(base | low);
  };
  auto rg = [&](uint32_t lo, uint32_t hi_incl) {
    hi_incl = std::min(hi_incl, max_low);
    if (lo <= hi_incl) s.ranges.push_back({base | lo, (base | hi_incl) + 1});
  };
  switch (style) {
    case kSparse:
      for (uint32_t n = rnd(rng, 1, 200); n--;) pt(rnd(rng, 0, 0xFFFF));
      break;
    case kMediumArray:
      for (uint32_t n = rnd(rng, 200, 4000); n--;) pt(rnd(rng, 0, 0xFFFF));
      break;
    case kNearThreshold: {  // exactly 4093..4099 distinct values, stride 15
      const uint32_t off = rnd(rng, 0, 3000), n = 4096 + rnd(rng, 0, 6) - 3;
      for (uint32_t i = 0; i < n; ++i) pt(off + 15 * i);
      break;
    }
    case kDense: {
      std::bernoulli_distribution keep(0.1 + 0.85 * std::uniform_real_distribution<double>(0, 1)(rng));
      for (uint32_t low = 0; low <= 0xFFFF; ++low)
        if (keep(rng)) pt(low);
      break;
    }
    case kRuns:
    case kRunsWithNoise: {
      for (uint32_t n = rnd(rng, 1, 60); n--;) {
        const uint32_t lo = rnd(rng, 0, 0xFFFF);
        rg(lo, std::min<uint32_t>(0xFFFF, lo + rnd(rng, 0, rnd(rng, 0, 1) ? 30 : 3000)));
      }
      if (style == kRunsWithNoise)
        for (uint32_t n = rnd(rng, 1, 300); n--;) pt(rnd(rng, 0, 0xFFFF));
      break;
    }
    case kFull:
      rg(0, 0xFFFF);
      break;
  }
}

Vec expand(const Spec& s) {
  Vec v = s.points;
  for (auto [lo, hi] : s.ranges)
    for (uint32_t x = lo; x < hi; ++x) v.push_back(x);
  return sorted_unique(std::move(v));
}

struct Sample {
  RoaringBitmap bm;
  Vec ref;
};

// Builds the bitmap through one of several construction paths.
Sample build(Rng& rng, const Spec& s) {
  Sample out;
  out.ref = expand(s);
  Vec pts = s.points;
  std::shuffle(pts.begin(), pts.end(), rng);
  switch (rnd(rng, 0, 3)) {
    case 0:
      out.bm = RoaringBitmap::from_sorted(out.ref.data(), out.ref.size());
      break;
    case 1:
      for (uint32_t v : pts) out.bm.add(v);
      for (auto [lo, hi] : s.ranges) out.bm.add_range(lo, hi);
      break;
    case 2:
      for (auto [lo, hi] : s.ranges) out.bm.add_range(lo, hi);
      out.bm.add_many(pts.data(), pts.size());
      break;
    default: {
      out.bm.add_many(pts.data(), pts.size());
      for (auto [lo, hi] : s.ranges) out.bm.add_range(lo, hi);
      break;
    }
  }
  if (rnd(rng, 0, 1)) out.bm.run_optimize();
  return out;
}

const std::vector<uint32_t> kKeyDomain = {0, 1, 2, 3, 4, 7, 0xFFFE, 0xFFFF};

Sample random_mixed(Rng& rng, const std::vector<uint32_t>& keys = kKeyDomain) {
  Spec s;
  for (uint32_t key : keys) {
    const int style = int(rnd(rng, 0, kStyles));  // == kStyles: leave key empty
    if (style < kStyles) gen_container(rng, key, style, s);
  }
  return build(rng, s);
}

Vec ref_and(const Vec& a, const Vec& b) {
  Vec r;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(r));
  return r;
}
Vec ref_or(const Vec& a, const Vec& b) {
  Vec r;
  std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(r));
  return r;
}
Vec ref_andnot(const Vec& a, const Vec& b) {
  Vec r;
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(r));
  return r;
}
Vec ref_xor(const Vec& a, const Vec& b) {
  Vec r;
  std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(r));
  return r;
}

void check_set_ops(const Sample& a, const Sample& b) {
  const Vec e_and = ref_and(a.ref, b.ref), e_or = ref_or(a.ref, b.ref);
  const Vec e_andnot = ref_andnot(a.ref, b.ref), e_xor = ref_xor(a.ref, b.ref);

  const RoaringBitmap r_and = RoaringBitmap::and_(a.bm, b.bm);
  expect_matches(r_and, e_and);
  expect_matches(RoaringBitmap::or_(a.bm, b.bm), e_or);
  const RoaringBitmap r_andnot = RoaringBitmap::andnot_(a.bm, b.bm);
  expect_matches(r_andnot, e_andnot);
  expect_matches(RoaringBitmap::xor_(a.bm, b.bm), e_xor);

  RoaringBitmap c = a.bm;
  c.and_inplace(b.bm);
  expect_matches(c, e_and);
  c = a.bm;
  c.or_inplace(b.bm);
  expect_matches(c, e_or);
  c = a.bm;
  c.andnot_inplace(b.bm);
  expect_matches(c, e_andnot);

  EXPECT_EQ(a.bm.and_cardinality(b.bm), uint64_t(e_and.size()));
  EXPECT_EQ(b.bm.and_cardinality(a.bm), uint64_t(e_and.size()));
  EXPECT_EQ(a.bm.intersects(b.bm), !e_and.empty());
  EXPECT_EQ(b.bm.intersects(a.bm), !e_and.empty());
  EXPECT_FALSE(r_andnot.intersects(b.bm));
  EXPECT_EQ(r_andnot.and_cardinality(b.bm), uint64_t(0));
  EXPECT_EQ(a.bm == b.bm, a.ref == b.ref);
}

// Random next()/advance() walk compared against lower_bound on the reference.
void check_iterator_walk(Rng& rng, const RoaringBitmap& bm, const Vec& ref) {
  RoaringBitmap::Iterator it(bm);
  ASSERT_EQ(it.value(), kUnpositioned);
  ASSERT_EQ(it.cost(), uint64_t(ref.size()));
  size_t idx = 0;  // index in ref of the expected value after the next step
  bool positioned = false;
  for (int step = 0; step < 400; ++step) {
    uint32_t got, want;
    if (rnd(rng, 0, 9) < 4) {
      got = it.next();
      if (positioned) ++idx;
      want = idx < ref.size() ? ref[idx] : kNoMoreDocs;
    } else {
      const uint64_t from = positioned ? uint64_t(it.value()) + 1 : 0;
      uint64_t target;
      switch (rnd(rng, 0, 4)) {
        case 0: target = from + rnd(rng, 0, 63); break;             // within a word
        case 1: target = from + rnd(rng, 0, 70000); break;          // nearby container
        case 2: target = from + rnd(rng, 0, 1u << 24); break;       // far
        case 3: {                                                   // an exact member
          auto lb = std::lower_bound(ref.begin(), ref.end(), from);
          const size_t left = size_t(ref.end() - lb);
          target = left ? *(lb + rnd(rng, 0, uint32_t(std::min<size_t>(left - 1, 5000)))) : from;
          break;
        }
        default: target = from; break;
      }
      target = std::min<uint64_t>(target, 0xFFFFFFFFu);
      got = it.advance(uint32_t(target));
      idx = size_t(std::lower_bound(ref.begin(), ref.end(), target) - ref.begin());
      want = idx < ref.size() ? ref[idx] : kNoMoreDocs;
    }
    positioned = true;
    ASSERT_EQ(got, want);
    ASSERT_EQ(it.value(), want);
    if (got == kNoMoreDocs) break;
  }
}

}  // namespace

// ===========================================================================
// Edge cases

TEST(Roaring, EmptyBitmap) {
  RoaringBitmap bm;
  EXPECT_TRUE(bm.empty());
  EXPECT_EQ(bm.cardinality(), uint64_t(0));
  EXPECT_EQ(bm.minimum(), kNoMoreDocs);
  EXPECT_EQ(bm.maximum(), kNoMoreDocs);
  EXPECT_EQ(bm.rank(0), uint64_t(0));
  EXPECT_EQ(bm.rank(0xFFFFFFFFu), uint64_t(0));
  EXPECT_FALSE(bm.contains(0));
  EXPECT_FALSE(bm.remove(0));
  EXPECT_TRUE(bm.to_vector().empty());
  EXPECT_EQ(bm.serialized_size(), size_t(8));
  expect_matches(bm, {});

  RoaringBitmap::Iterator it(bm);
  EXPECT_EQ(it.value(), kUnpositioned);
  EXPECT_EQ(it.cost(), uint64_t(0));
  EXPECT_EQ(it.next(), kNoMoreDocs);
  EXPECT_EQ(it.next(), kNoMoreDocs);
  RoaringBitmap::Iterator it2(bm);
  EXPECT_EQ(it2.advance(0), kNoMoreDocs);

  RoaringBitmap other = RoaringBitmap::from_range(10, 20);
  expect_matches(RoaringBitmap::and_(bm, other), {});
  expect_matches(RoaringBitmap::andnot_(bm, other), {});
  expect_matches(RoaringBitmap::or_(bm, other), other.to_vector());
  expect_matches(RoaringBitmap::xor_(other, bm), other.to_vector());
  expect_matches(RoaringBitmap::andnot_(other, bm), other.to_vector());
  EXPECT_FALSE(bm.intersects(other));
  EXPECT_EQ(other.and_cardinality(bm), uint64_t(0));
  expect_matches(RoaringBitmap::or_many({}), {});
  expect_matches(RoaringBitmap::or_many({&bm, &bm}), {});
  expect_matches(RoaringBitmap::from_sorted(nullptr, 0), {});
  expect_matches(RoaringBitmap::from_range(5, 5), {});
  bm.add_many(nullptr, 0);
  bm.run_optimize();
  bm.shrink_to_fit();
  EXPECT_TRUE(bm == RoaringBitmap());
}

TEST(Roaring, SingleElements) {
  for (uint32_t x : {0u, 1u, 65535u, 65536u, 65537u, 0x12345678u, kMax - 1, kMax}) {
    RoaringBitmap bm;
    bm.add(x);
    bm.add(x);  // duplicate is a no-op
    expect_matches(bm, {x});
    EXPECT_TRUE(bm.contains(x));
    if (x > 0) EXPECT_FALSE(bm.contains(x - 1));
    EXPECT_FALSE(bm.contains(x + 1));
    EXPECT_EQ(bm.rank(x), uint64_t(1));
    if (x > 0) EXPECT_EQ(bm.rank(x - 1), uint64_t(0));
    RoaringBitmap::Iterator it(bm);
    EXPECT_EQ(it.next(), x);
    EXPECT_EQ(it.next(), kNoMoreDocs);
    RoaringBitmap::Iterator it2(bm);
    EXPECT_EQ(it2.advance(x), x);
    EXPECT_EQ(it2.advance(x), x);  // target <= current: no movement
    EXPECT_EQ(it2.advance(x + 1), kNoMoreDocs);
    bm.run_optimize();
    expect_matches(bm, {x});
    EXPECT_TRUE(bm.remove(x));
    EXPECT_FALSE(bm.remove(x));
    expect_matches(bm, {});
  }
}

TEST(Roaring, ReservedSentinelsRejected) {
  RoaringBitmap bm;
  EXPECT_THROW(bm.add(kNoMoreDocs));
  EXPECT_THROW(bm.add(kUnpositioned));
  EXPECT_THROW(bm.add_range(0, 0xFFFFFFFFu));
  EXPECT_THROW(RoaringBitmap::from_range(kMax, kNoMoreDocs));
  EXPECT_THROW(RoaringBitmap::from_range(10, 5));
  const uint32_t bad[] = {1, kUnpositioned};
  EXPECT_THROW(RoaringBitmap::from_sorted(bad, 2));
  const uint32_t unsorted[] = {5, 3};
  EXPECT_THROW(RoaringBitmap::from_sorted(unsorted, 2));
  const uint32_t dup[] = {3, 3};
  EXPECT_THROW(RoaringBitmap::from_sorted(dup, 2));
  const uint32_t many_bad[] = {9, 8, 7, kNoMoreDocs};
  EXPECT_THROW(bm.add_many(many_bad, 4));
  EXPECT_TRUE(bm.empty());
  EXPECT_FALSE(bm.contains(kNoMoreDocs));
  EXPECT_FALSE(bm.contains(kUnpositioned));
  EXPECT_FALSE(bm.remove(kNoMoreDocs));
  bm.add(kMax);  // the largest legal value
  EXPECT_TRUE(bm.contains(kMax));
  EXPECT_FALSE(bm.contains(kUnpositioned));
  bm.add_range(kMax - 10, kMax + 1);  // hi = kMaxValue + 1 is allowed
  EXPECT_EQ(bm.cardinality(), uint64_t(11));
  EXPECT_EQ(bm.maximum(), kMax);
}

TEST(Roaring, FullContainer) {
  // Via add_range (run container).
  RoaringBitmap r = RoaringBitmap::from_range(65536, 131072);
  EXPECT_EQ(r.cardinality(), uint64_t(65536));
  EXPECT_FALSE(r.contains(65535));
  EXPECT_TRUE(r.contains(65536));
  EXPECT_TRUE(r.contains(131071));
  EXPECT_FALSE(r.contains(131072));
  EXPECT_EQ(r.rank(131071), uint64_t(65536));
  EXPECT_EQ(r.rank(100000), uint64_t(100000 - 65536 + 1));
  EXPECT_EQ(r.serialized_size(), size_t(8 + 5 + 6));
  EXPECT_EQ(r.container_stats().runs, size_t(1));

  // Via individual adds in random order (array -> bitmap transition).
  Rng rng(1);
  Vec vals(65536);
  for (uint32_t i = 0; i < 65536; ++i) vals[i] = 65536 + i;
  Vec shuffled = vals;
  std::shuffle(shuffled.begin(), shuffled.end(), rng);
  RoaringBitmap b;
  for (uint32_t v : shuffled) b.add(v);
  EXPECT_EQ(b.container_stats().bitmaps, size_t(1));
  expect_matches(b, vals);
  EXPECT_TRUE(b == r);  // equality across encodings
  EXPECT_TRUE(r == b);
  b.run_optimize();
  EXPECT_EQ(b.container_stats().runs, size_t(1));
  EXPECT_EQ(b.serialized_size(), r.serialized_size());

  // Punch a hole into the run container and fill it back.
  EXPECT_TRUE(r.remove(100000));
  EXPECT_FALSE(r.contains(100000));
  EXPECT_EQ(r.cardinality(), uint64_t(65535));
  EXPECT_EQ(r.container_stats().runs, size_t(1));  // two runs, still the smallest
  RoaringBitmap::Iterator it(r);
  EXPECT_EQ(it.advance(99999), 99999u);
  EXPECT_EQ(it.next(), 100001u);
  r.add(100000);
  EXPECT_TRUE(r == b);
  expect_matches(r, vals);

  // Remove every value: run -> bitmap -> array -> empty.
  std::shuffle(shuffled.begin(), shuffled.end(), rng);
  for (size_t i = 0; i < shuffled.size(); ++i) {
    ASSERT_TRUE(r.remove(shuffled[i]));
    if (i % 8192 == 0 || i + 4100 == shuffled.size()) {
      Vec rest(shuffled.begin() + ptrdiff_t(i) + 1, shuffled.end());
      expect_matches(r, sorted_unique(rest));
    }
  }
  EXPECT_TRUE(r.empty());
}

TEST(Roaring, AddRangeSpanningContainers) {
  Rng rng(2);
  for (int iter = 0; iter < 60; ++iter) {
    Sample s = random_mixed(rng, {0, 1, 2, 3, 4, 5});
    std::set<uint32_t> ref(s.ref.begin(), s.ref.end());
    for (int k = 0; k < 3; ++k) {
      uint32_t lo = rnd(rng, 0, 6 * 65536), hi = lo + rnd(rng, 0, rnd(rng, 0, 1) ? 100 : 200000);
      if (rnd(rng, 0, 3) == 0) lo &= ~0xFFFFu;   // container-aligned start
      if (rnd(rng, 0, 3) == 0) hi |= 0xFFFFu;    // container-aligned end
      s.bm.add_range(lo, hi);
      for (uint32_t x = lo; x < hi; ++x) ref.insert(x);
    }
    expect_matches(s.bm, Vec(ref.begin(), ref.end()));
  }
  // The whole universe: 65536 full run containers.
  RoaringBitmap all = RoaringBitmap::from_range(0, kMax + 1);
  EXPECT_EQ(all.cardinality(), uint64_t(kMax) + 1);
  EXPECT_EQ(all.container_count(), size_t(65536));
  EXPECT_EQ(all.minimum(), 0u);
  EXPECT_EQ(all.maximum(), kMax);
  EXPECT_TRUE(all.contains(kMax));
  EXPECT_FALSE(all.contains(kUnpositioned));
  EXPECT_EQ(all.rank(kMax), uint64_t(kMax) + 1);
  EXPECT_EQ(all.rank(0xFFFFFFFFu), uint64_t(kMax) + 1);
  RoaringBitmap::Iterator it(all);
  EXPECT_EQ(it.advance(kMax - 1), kMax - 1);
  EXPECT_EQ(it.next(), kMax);
  EXPECT_EQ(it.next(), kNoMoreDocs);
  EXPECT_TRUE(roundtrip(all) == all);
  EXPECT_EQ(all.and_cardinality(RoaringBitmap::from_range(5, 70000)), uint64_t(70000 - 5));
}

// ===========================================================================
// Randomized differential tests

TEST(Roaring, RandomMutationsVsStdSet) {
  Rng rng(3);
  struct Dist {
    const char* name;
    uint32_t lo, hi;      // value domain
    int range_pct;        // share of add_range operations
    uint32_t max_range;   // max add_range length
  };
  const Dist dists[] = {
      {"sparse", 0, kMax, 2, 50},                       // spans many containers
      {"dense", 0, 3 * 65536 - 1, 2, 300},              // crosses the 4096 threshold
      {"clustered", 5 * 65536, 9 * 65536, 10, 2000},    // runs + noise
      {"run-heavy", 0, 4 * 65536, 30, 5000},
      {"near-top", kMax - 200000, kMax, 5, 1000},       // up to 0xFFFFFFFD
  };
  for (const Dist& d : dists) {
    std::set<uint32_t> ref;
    RoaringBitmap bm;
    const int steps = 24000;
    for (int step = 1; step <= steps; ++step) {
      const uint32_t op = rnd(rng, 0, 99);
      const uint32_t v = rnd(rng, d.lo, d.hi);
      if (op < uint32_t(d.range_pct)) {
        const uint32_t hi = uint32_t(std::min<uint64_t>(uint64_t(v) + rnd(rng, 0, d.max_range), uint64_t(d.hi) + 1));
        bm.add_range(v, hi);
        for (uint32_t x = v; x < hi; ++x) ref.insert(x);
      } else if (op < 60) {
        bm.add(v);
        ref.insert(v);
      } else if (op < 80) {  // remove an existing value
        auto it = ref.lower_bound(v);
        if (it == ref.end()) continue;
        const uint32_t x = *it;
        ASSERT_TRUE(bm.remove(x));
        ref.erase(it);
      } else if (op < 95) {  // remove a random (often absent) value
        ASSERT_EQ(bm.remove(v), ref.erase(v) == 1);
      } else if (op < 97) {
        ASSERT_EQ(bm.contains(v), ref.count(v) == 1);
      } else {
        bm.run_optimize();  // interleave encodings with mutation
      }
      if (step % 6000 == 0) {
        const Vec snap(ref.begin(), ref.end());
        expect_point_queries(rng, bm, snap, 300);
        expect_matches(bm, snap);
      }
    }
    // Drain everything (exercises bitmap -> array and run -> plain on removal).
    Vec rest(ref.begin(), ref.end());
    std::shuffle(rest.begin(), rest.end(), rng);
    if (rest.size() > 30000) rest.resize(30000);
    for (uint32_t x : rest) {
      ASSERT_TRUE(bm.remove(x));
      ref.erase(x);
    }
    const Vec snap(ref.begin(), ref.end());
    expect_point_queries(rng, bm, snap, 300);
    expect_matches(bm, snap);
  }
}

TEST(Roaring, MixedContainersPointQueries) {
  Rng rng(4);
  for (int iter = 0; iter < 30; ++iter) {
    Sample s = random_mixed(rng);
    expect_matches(s.bm, s.ref);
    expect_point_queries(rng, s.bm, s.ref, 500);
  }
}

TEST(Roaring, SetOpsAllContainerPairs) {
  Rng rng(5);
  RoaringBitmap::ContainerStats seen;
  for (int iter = 0; iter < 50; ++iter) {
    Sample a = random_mixed(rng), b = random_mixed(rng);
    for (const Sample* s : {&a, &b}) {
      const auto st = s->bm.container_stats();
      seen.arrays += st.arrays;
      seen.bitmaps += st.bitmaps;
      seen.runs += st.runs;
    }
    check_set_ops(a, b);
    if (iter % 10 == 0) {
      check_set_ops(a, a);  // identical inputs
      RoaringBitmap self = a.bm;
      self.and_inplace(self);
      expect_matches(self, a.ref);
      self.or_inplace(self);
      expect_matches(self, a.ref);
      self.andnot_inplace(self);
      expect_matches(self, {});
    }
  }
  EXPECT_GT(seen.arrays, size_t(0));
  EXPECT_GT(seen.bitmaps, size_t(0));
  EXPECT_GT(seen.runs, size_t(0));

  // Skewed array sizes (galloping path) and disjoint key sets.
  for (int iter = 0; iter < 20; ++iter) {
    Spec small, big;
    for (uint32_t n = rnd(rng, 1, 40); n--;) small.points.push_back(rnd(rng, 0, 65535));
    for (uint32_t n = 4000; n--;) big.points.push_back(rnd(rng, 0, 65535));
    for (uint32_t v : Vec(small.points)) if (rnd(rng, 0, 1)) big.points.push_back(v);
    check_set_ops(build(rng, small), build(rng, big));
    check_set_ops(build(rng, big), build(rng, small));
  }
  check_set_ops(random_mixed(rng, {0, 2, 4}), random_mixed(rng, {1, 3, 5}));
}

TEST(Roaring, OrMany) {
  Rng rng(6);
  for (int iter = 0; iter < 12; ++iter) {
    const size_t k = rnd(rng, 0, 9);
    std::vector<Sample> samples;
    for (size_t i = 0; i < k; ++i) {
      if (rnd(rng, 0, 2) == 0) {  // small sparse inputs: tiny all-array groups
        Spec s;
        for (uint32_t n = rnd(rng, 0, 20); n--;) s.points.push_back(rnd(rng, 0, 4 * 65536));
        samples.push_back(build(rng, s));
      } else {
        samples.push_back(random_mixed(rng));
      }
    }
    std::vector<const RoaringBitmap*> ptrs;
    Vec expected;
    RoaringBitmap pairwise;
    for (const Sample& s : samples) {
      ptrs.push_back(&s.bm);
      expected = ref_or(expected, s.ref);
      pairwise.or_inplace(s.bm);
    }
    ptrs.push_back(nullptr);  // null inputs are skipped
    const RoaringBitmap u = RoaringBitmap::or_many(ptrs);
    expect_matches(u, expected);
    EXPECT_TRUE(u == pairwise);
  }
}

TEST(Roaring, IteratorNextAndAdvance) {
  Rng rng(7);
  for (int iter = 0; iter < 30; ++iter) {
    Sample s = random_mixed(rng);
    // Plain scan.
    RoaringBitmap::Iterator it(s.bm);
    for (uint32_t v : s.ref) ASSERT_EQ(it.next(), v);
    ASSERT_EQ(it.next(), kNoMoreDocs);
    ASSERT_EQ(it.next(), kNoMoreDocs);
    ASSERT_EQ(it.advance(123), kNoMoreDocs);
    ASSERT_EQ(it.value(), kNoMoreDocs);
    // Random walks.
    for (int w = 0; w < 25; ++w) check_iterator_walk(rng, s.bm, s.ref);
  }
  // Sparse wide-key bitmaps: advance across many containers.
  for (int iter = 0; iter < 10; ++iter) {
    Vec v;
    for (int n = 0; n < 5000; ++n) v.push_back(rnd(rng, 0, kMax));
    v = sorted_unique(v);
    RoaringBitmap bm = RoaringBitmap::from_sorted(v.data(), v.size());
    for (int w = 0; w < 20; ++w) check_iterator_walk(rng, bm, v);
  }
  // Targets beyond the largest legal value.
  RoaringBitmap bm = RoaringBitmap::from_range(kMax - 3, kMax + 1);
  RoaringBitmap::Iterator a(bm);
  EXPECT_EQ(a.advance(kUnpositioned), kNoMoreDocs);
  RoaringBitmap::Iterator b(bm);
  EXPECT_EQ(b.advance(0), kMax - 3);
  EXPECT_EQ(b.advance(kMax), kMax);
  EXPECT_EQ(b.advance(kNoMoreDocs), kNoMoreDocs);
}

TEST(Roaring, RunOptimize) {
  Rng rng(8);
  // Run-heavy content shrinks and keeps its contents.
  for (int iter = 0; iter < 20; ++iter) {
    Spec s;
    for (uint32_t key = 0; key < 6; ++key) gen_container(rng, key, kRuns, s);
    Sample smp = build(rng, s);
    const size_t before = smp.bm.serialized_size();
    smp.bm.run_optimize();
    EXPECT_LE(smp.bm.serialized_size(), before);
    EXPECT_GT(smp.bm.container_stats().runs, size_t(0));
    expect_matches(smp.bm, smp.ref);
    smp.bm.run_optimize();  // idempotent
    expect_matches(smp.bm, smp.ref);
  }
  // Random sparse values: runs never pay off.
  Vec v;
  for (int n = 0; n < 3000; ++n) v.push_back(rnd(rng, 0, 4 * 65536));
  v = sorted_unique(v);
  RoaringBitmap sparse = RoaringBitmap::from_sorted(v.data(), v.size());
  const size_t sz = sparse.serialized_size();
  sparse.run_optimize();
  EXPECT_EQ(sparse.container_stats().runs, size_t(0));
  EXPECT_EQ(sparse.serialized_size(), sz);
  // Alternating bits (32768 single-value runs) stay a bitmap.
  RoaringBitmap alt;
  for (uint32_t x = 0; x < 65536; x += 2) alt.add(x);
  alt.run_optimize();
  EXPECT_EQ(alt.container_stats().bitmaps, size_t(1));
  // Runs + scattered noise: mutate optimised containers against a reference.
  for (int iter = 0; iter < 10; ++iter) {
    Spec s;
    for (uint32_t key = 0; key < 3; ++key) gen_container(rng, key, kRunsWithNoise, s);
    Sample smp = build(rng, s);
    smp.bm.run_optimize();
    std::set<uint32_t> ref(smp.ref.begin(), smp.ref.end());
    for (int k = 0; k < 4000; ++k) {
      const uint32_t x = rnd(rng, 0, 3 * 65536);
      if (rnd(rng, 0, 1)) {
        smp.bm.add(x);
        ref.insert(x);
      } else {
        ASSERT_EQ(smp.bm.remove(x), ref.erase(x) == 1);
      }
    }
    expect_matches(smp.bm, Vec(ref.begin(), ref.end()));
  }
}

TEST(Roaring, SerializationRoundtripAndCorruption) {
  Rng rng(9);
  for (int iter = 0; iter < 20; ++iter) {
    Sample s = random_mixed(rng);
    for (int pass = 0; pass < 2; ++pass) {
      // Embedded at a non-zero offset, followed by a second bitmap.
      ByteWriter w;
      w.put_u8(0xAB);
      s.bm.serialize(w);
      RoaringBitmap other = RoaringBitmap::from_range(iter * 1000u, iter * 1000u + 77);
      other.serialize(w);
      ByteReader r(w.buffer());
      EXPECT_EQ(r.get_u8(), 0xAB);
      RoaringBitmap back = RoaringBitmap::deserialize(r);
      RoaringBitmap back2 = RoaringBitmap::deserialize(r);
      EXPECT_TRUE(r.eof());
      expect_matches(back, s.ref);
      EXPECT_TRUE(back2 == other);
      s.bm.run_optimize();  // second pass: optimised encodings
    }
  }

  auto expect_corrupt = [](const std::vector<uint8_t>& buf) {
    ByteReader r(buf);
    EXPECT_THROW(RoaringBitmap::deserialize(r));
  };
  // Bad magic, truncation at every prefix length of a small bitmap.
  RoaringBitmap bm;
  for (uint32_t x : {3u, 70000u, 70001u, 70002u, 70003u, 70004u, 200000u}) bm.add(x);
  bm.run_optimize();
  const std::vector<uint8_t> good = bytes_of(bm);
  {
    auto b = good;
    b[0] ^= 1;
    expect_corrupt(b);
  }
  for (size_t len = 0; len < good.size(); ++len)
    expect_corrupt(std::vector<uint8_t>(good.begin(), good.begin() + ptrdiff_t(len)));

  // Hand-built headers.
  auto make = [](uint32_t count, std::vector<std::tuple<uint16_t, uint8_t, uint16_t>> hdrs,
                 std::vector<uint16_t> payload) {
    ByteWriter w;
    w.put_u32(0x314D4252u);
    w.put_u32(count);
    for (auto [k, t, c] : hdrs) {
      w.put_u16(k);
      w.put_u8(t);
      w.put_u16(c);
    }
    for (uint16_t p : payload) w.put_u16(p);
    return w.release();
  };
  {  // valid baseline: arrays {5,9} at key 1 and {7} at key 2
    auto ok = make(2, {{1, 1, 1}, {2, 1, 0}}, {5, 9, 7});
    ByteReader r(ok);
    const RoaringBitmap d = RoaringBitmap::deserialize(r);
    EXPECT_TRUE(d.to_vector() == (Vec{65536 + 5, 65536 + 9, 131072 + 7}));
  }
  expect_corrupt(make(2, {{2, 1, 0}, {1, 1, 0}}, {5, 7}));        // keys out of order
  expect_corrupt(make(2, {{1, 1, 0}, {1, 1, 0}}, {5, 7}));        // duplicate key
  expect_corrupt(make(1, {{1, 4, 0}}, {5}));                      // bad type
  expect_corrupt(make(1, {{1, 1, 1}}, {9, 5}));                   // unsorted array
  expect_corrupt(make(1, {{1, 1, 1}}, {5, 5}));                   // duplicate value
  expect_corrupt(make(1, {{1, 1, 5000}}, std::vector<uint16_t>(5001, 0)));  // oversized array
  expect_corrupt(make(1, {{1, 2, 10}}, std::vector<uint16_t>(4096, 0)));    // undersized bitmap
  expect_corrupt(make(1, {{1, 3, 9}}, {2, 0, 4, 2, 7}));          // runs overlap
  expect_corrupt(make(1, {{1, 3, 5}}, {2, 0, 2, 3, 3}));          // runs touch
  expect_corrupt(make(1, {{1, 3, 3}}, {1, 0, 4}));                // run card mismatch (header 4, runs 5)
  expect_corrupt(make(1, {{1, 3, 1}}, {1, 0xFFFF, 1}));           // run overflows container
  expect_corrupt(make(1, {{1, 3, 0}}, {0}));                      // zero runs
  expect_corrupt(make(1, {{0xFFFF, 1, 0}}, {0xFFFE}));            // sentinel value
  expect_corrupt(make(1, {{0xFFFF, 3, 2}}, {1, 0xFFFC, 2}));      // run reaching sentinels
  expect_corrupt(make(70000, {}, {}));                            // too many containers
  {  // bitmap payload whose popcount disagrees with the header
    RoaringBitmap dense = RoaringBitmap::from_range(0, 5000);
    dense.add(6000);
    dense.run_optimize();
    RoaringBitmap plain;
    for (uint32_t x = 0; x < 9000; x += 2) plain.add(x);
    ASSERT_EQ(plain.container_stats().bitmaps, size_t(1));
    auto b = bytes_of(plain);
    b[8 + 5 + 100] ^= 0x10;  // flip one payload bit
    expect_corrupt(b);
  }
}

TEST(Roaring, BulkBuildersAgree) {
  Rng rng(10);
  for (int iter = 0; iter < 15; ++iter) {
    Sample s = random_mixed(rng);
    const RoaringBitmap fs = RoaringBitmap::from_sorted(s.ref.data(), s.ref.size());
    expect_matches(fs, s.ref);
    // add_many with shuffled input and duplicates, onto a non-empty bitmap.
    Vec shuffled = s.ref;
    shuffled.insert(shuffled.end(), s.ref.begin(), s.ref.begin() + ptrdiff_t(s.ref.size() / 3));
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    RoaringBitmap am;
    const size_t half = shuffled.size() / 2;
    am.add_many(shuffled.data(), half);
    am.add_many(shuffled.data() + half, shuffled.size() - half);
    expect_matches(am, s.ref);
    EXPECT_TRUE(am == fs);
    EXPECT_TRUE(am == s.bm);
    // Small unsorted batches go through add().
    RoaringBitmap small;
    for (size_t i = 0; i < shuffled.size() && i < 2000; i += 10)
      small.add_many(shuffled.data() + i, std::min<size_t>(10, shuffled.size() - i));
    Vec head(shuffled.begin(), shuffled.begin() + ptrdiff_t(std::min<size_t>(2000, shuffled.size())));
    expect_matches(small, sorted_unique(head));
  }
}

TEST(Roaring, EqualityAcrossEncodings) {
  // Same set as array, bitmap and run containers.
  RoaringBitmap runs = RoaringBitmap::from_range(1000, 9000);
  RoaringBitmap bits;
  for (uint32_t x = 1000; x < 9000; ++x) bits.add(x);
  EXPECT_EQ(runs.container_stats().runs, size_t(1));
  EXPECT_EQ(bits.container_stats().bitmaps, size_t(1));
  EXPECT_TRUE(runs == bits);
  RoaringBitmap small_runs = RoaringBitmap::from_range(100, 200);
  RoaringBitmap arr;
  for (uint32_t x = 100; x < 200; ++x) arr.add(x);
  EXPECT_EQ(arr.container_stats().arrays, size_t(1));
  EXPECT_TRUE(small_runs == arr);
  arr.remove(150);
  EXPECT_FALSE(small_runs == arr);
  EXPECT_TRUE(small_runs != arr);
  EXPECT_FALSE(runs == small_runs);
}

TEST(Roaring, MemoryAndShrink) {
  RoaringBitmap bm;
  for (uint32_t x = 0; x < 1000000; x += 3) bm.add(x);
  const size_t before = bm.memory_bytes();
  bm.shrink_to_fit();
  EXPECT_LE(bm.memory_bytes(), before);
  EXPECT_GT(bm.memory_bytes(), size_t(1000000 / 3 / 8));  // at least one bit per value
  RoaringBitmap range = RoaringBitmap::from_range(0, 1000000);
  EXPECT_LT(range.memory_bytes(), size_t(4096));  // 16 run containers
  EXPECT_EQ(range.cardinality(), uint64_t(1000000));
}
