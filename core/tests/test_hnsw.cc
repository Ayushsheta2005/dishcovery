// Tests for the HNSW index (core/src/vector/hnsw.h) and the distance kernels.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <thread>
#include <vector>

#include "testing.h"
#include "util/bytes.h"
#include "vector/distance.h"
#include "vector/hnsw.h"

using dc::ByteWriter;
using namespace dc::vec;

namespace {

// Small, platform-independent RNG (std::normal_distribution differs between
// standard libraries, which would make test data platform-dependent).
struct Rng {
  uint64_t state;
  uint64_t next() {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  double uniform() { return (static_cast<double>(next() >> 11) + 0.5) * 0x1.0p-53; }  // (0,1)
  float normal() {  // Box-Muller
    const double u1 = uniform(), u2 = uniform();
    return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
  }
};

std::vector<float> random_unit_vectors(uint32_t n, uint32_t dim, uint64_t seed) {
  Rng rng{seed};
  std::vector<float> v(static_cast<size_t>(n) * dim);
  for (float& x : v) x = rng.normal();
  for (uint32_t i = 0; i < n; ++i) normalize(&v[static_cast<size_t>(i) * dim], dim);
  return v;
}

// Gaussian mixture on the unit sphere: pick a random centre, add isotropic
// noise of total norm ~`spread`, renormalise. Closer to real embedding
// distributions (topical clusters) than uniform random vectors.
std::vector<float> sample_mixture(const std::vector<float>& centers, uint32_t n, uint32_t dim,
                                  float spread, uint64_t seed) {
  Rng rng{seed};
  const uint32_t nc = static_cast<uint32_t>(centers.size() / dim);
  const float sigma = spread / std::sqrt(static_cast<float>(dim));
  std::vector<float> v(static_cast<size_t>(n) * dim);
  for (uint32_t i = 0; i < n; ++i) {
    const float* c = &centers[(rng.next() % nc) * dim];
    float* x = &v[static_cast<size_t>(i) * dim];
    for (uint32_t d = 0; d < dim; ++d) x[d] = c[d] + sigma * rng.normal();
    normalize(x, dim);
  }
  return v;
}

double recall(const HnswIndex& idx, const float* base, const std::vector<float>& queries,
              uint32_t k, uint32_t ef, const IdFilter* filter = nullptr) {
  const uint32_t n = idx.size(), dim = idx.dim();
  const size_t nq = queries.size() / dim;
  double hit = 0, total = 0;
  for (size_t q = 0; q < nq; ++q) {
    const float* qv = &queries[q * dim];
    const auto approx = idx.search(qv, k, ef, filter);
    const auto exact = brute_force_topk(base, n, dim, qv, k, filter);
    std::set<uint32_t> truth;
    for (const auto& e : exact) truth.insert(e.id);
    for (const auto& a : approx) hit += truth.count(a.id);
    total += static_cast<double>(exact.size());
  }
  return total == 0 ? 1.0 : hit / total;
}

bool same_results(const std::vector<ScoredId>& a, const std::vector<ScoredId>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].id != b[i].id || a[i].score != b[i].score) return false;
  return true;
}

std::vector<uint8_t> serialized(const HnswIndex& idx) {
  ByteWriter w;
  idx.serialize(w);
  return w.release();
}

// Header: magic(4) M(4) efc(4) seed(8) n(4) dim(4) max_level(4) entry(4) blocks(8).
constexpr size_t kHeaderBytes = 44;

// Shared clustered data set + serially built index (built once, reused).
struct Fixture {
  static constexpr uint32_t kN = 5000, kDim = 32, kQueries = 200, kClusters = 50;
  std::vector<float> centers = random_unit_vectors(kClusters, kDim, 1);
  std::vector<float> base = sample_mixture(centers, kN, kDim, 1.2f, 2);
  std::vector<float> queries = sample_mixture(centers, kQueries, kDim, 1.2f, 3);
  HnswIndex index{base.data(), kN, kDim, HnswParams{}};
  Fixture() { index.build(1); }
};

const Fixture& fixture() {
  static Fixture f;
  return f;
}

uint64_t mix(uint64_t x) { return Rng{x}.next(); }

}  // namespace

// ---------------------------------------------------------------------------

TEST(Distance, DotMatchesReference) {
  Rng rng{7};
  for (size_t dim : {0, 1, 3, 15, 16, 17, 31, 32, 100, 384}) {
    std::vector<float> a(dim), b(dim);
    for (size_t i = 0; i < dim; ++i) a[i] = rng.normal(), b[i] = rng.normal();
    double ref = 0;
    for (size_t i = 0; i < dim; ++i) ref += static_cast<double>(a[i]) * b[i];
    EXPECT_NEAR(dot(a.data(), b.data(), dim), ref, 1e-4 * (1.0 + std::sqrt(double(dim))));
  }
}

TEST(Distance, Normalize) {
  std::vector<float> v = {3.0f, 4.0f, 0.0f};
  normalize(v.data(), v.size());
  EXPECT_NEAR(v[0], 0.6, 1e-6);
  EXPECT_NEAR(v[1], 0.8, 1e-6);
  EXPECT_NEAR(dot(v.data(), v.data(), 3), 1.0, 1e-6);
  std::vector<float> z(5, 0.0f);
  normalize(z.data(), z.size());  // zero vector: no-op, no NaNs
  for (float x : z) EXPECT_EQ(x, 0.0f);
}

TEST(BruteForce, MatchesFullSort) {
  const uint32_t n = 300, dim = 8;
  const auto base = random_unit_vectors(n, dim, 11);
  const auto q = random_unit_vectors(1, dim, 12);
  std::vector<ScoredId> all;
  for (uint32_t i = 0; i < n; ++i) all.push_back({i, dot(q.data(), &base[i * dim], dim)});
  std::sort(all.begin(), all.end(), [](auto& a, auto& b) { return a.score > b.score; });
  const auto top = brute_force_topk(base.data(), n, dim, q.data(), 20);
  ASSERT_EQ(top.size(), 20u);
  for (size_t i = 0; i < top.size(); ++i) EXPECT_EQ(top[i].id, all[i].id);
  EXPECT_EQ(brute_force_topk(base.data(), n, dim, q.data(), 1000).size(), size_t{n});
  EXPECT_TRUE(brute_force_topk(base.data(), n, dim, q.data(), 0).empty());
}

TEST(Hnsw, EmptyIndex) {
  HnswIndex def;
  float q[4] = {1, 0, 0, 0};
  EXPECT_TRUE(def.search(q, 10, 10).empty());
  EXPECT_EQ(def.size(), 0u);

  HnswIndex idx(nullptr, 0, 4, HnswParams{});
  idx.build(4);
  EXPECT_EQ(idx.size(), 0u);
  EXPECT_EQ(idx.memory_bytes(), 0u);
  EXPECT_TRUE(idx.search(q, 10, 10).empty());
  const auto bytes = serialized(idx);
  HnswIndex loaded = HnswIndex::load(bytes.data(), bytes.size(), nullptr, 0, 4);
  EXPECT_EQ(loaded.size(), 0u);
  EXPECT_TRUE(loaded.search(q, 10, 10).empty());
}

TEST(Hnsw, SingleNode) {
  const auto v = random_unit_vectors(1, 16, 5);
  HnswIndex idx(v.data(), 1, 16, HnswParams{});
  idx.build(1);
  EXPECT_EQ(idx.entry_point(), 0u);
  const auto r = idx.search(v.data(), 5, 10);
  ASSERT_EQ(r.size(), 1u);
  EXPECT_EQ(r[0].id, 0u);
  EXPECT_NEAR(r[0].score, 1.0, 1e-5);
  const IdFilter none = [](uint32_t) { return false; };
  EXPECT_TRUE(idx.search(v.data(), 5, 10, &none).empty());
  EXPECT_TRUE(idx.search(v.data(), 0, 10).empty());

  const auto bytes = serialized(idx);
  HnswIndex loaded = HnswIndex::load(bytes.data(), bytes.size(), v.data(), 1, 16);
  EXPECT_TRUE(same_results(loaded.search(v.data(), 5, 10), r));
}

TEST(Hnsw, KLargerThanN) {
  const uint32_t n = 50, dim = 16;
  const auto base = random_unit_vectors(n, dim, 21);
  const auto q = random_unit_vectors(1, dim, 22);
  for (unsigned threads : {1u, 4u}) {
    HnswIndex idx(base.data(), n, dim, HnswParams{});
    idx.build(threads);
    const auto r = idx.search(q.data(), 100, 10);
    ASSERT_EQ(r.size(), size_t{n});
    EXPECT_TRUE(same_results(r, brute_force_topk(base.data(), n, dim, q.data(), 100)));
  }
}

TEST(Hnsw, EfSmallerThanKIsClamped) {
  const auto& f = fixture();
  const auto r = f.index.search(&f.queries[0], 10, 1);
  EXPECT_EQ(r.size(), 10u);
}

TEST(Hnsw, RecallClustered) {
  const auto& f = fixture();
  const double r100 = recall(f.index, f.base.data(), f.queries, 10, 100);
  const double r20 = recall(f.index, f.base.data(), f.queries, 10, 20);
  std::cout << "    recall@10: ef=20 " << r20 << ", ef=100 " << r100 << "\n";
  EXPECT_GE(r100, 0.95);
  EXPECT_GE(r100, r20);
}

TEST(Hnsw, RecallUniformRandom) {
  const uint32_t n = 3000, dim = 24;
  const auto base = random_unit_vectors(n, dim, 31);
  const auto queries = random_unit_vectors(100, dim, 32);
  HnswIndex idx(base.data(), n, dim, HnswParams{});
  idx.build(4);
  const double r = recall(idx, base.data(), queries, 10, 100);
  std::cout << "    recall@10 (uniform, ef=100): " << r << "\n";
  EXPECT_GE(r, 0.95);
}

TEST(Hnsw, ResultsSortedAndScoresExact) {
  const auto& f = fixture();
  for (uint32_t q = 0; q < 20; ++q) {
    const float* qv = &f.queries[q * Fixture::kDim];
    const auto r = f.index.search(qv, 25, 50);
    ASSERT_EQ(r.size(), 25u);
    std::set<uint32_t> ids;
    for (size_t i = 0; i < r.size(); ++i) {
      ids.insert(r[i].id);
      EXPECT_EQ(r[i].score, dot(qv, &f.base[size_t{r[i].id} * Fixture::kDim], Fixture::kDim));
      if (i > 0) EXPECT_GE(r[i - 1].score, r[i].score);
    }
    EXPECT_EQ(ids.size(), r.size());
  }
}

TEST(Hnsw, SearchStatsAccumulate) {
  const auto& f = fixture();
  SearchStats st;
  f.index.search(&f.queries[0], 10, 32, nullptr, &st);
  const SearchStats one = st;
  EXPECT_GT(one.distance_computations, 0u);
  EXPECT_GT(one.nodes_visited, 0u);
  EXPECT_LT(one.distance_computations, uint64_t{Fixture::kN} / 2);  // sub-linear
  f.index.search(&f.queries[0], 10, 32, nullptr, &st);
  EXPECT_EQ(st.distance_computations, 2 * one.distance_computations);
  EXPECT_EQ(st.nodes_visited, 2 * one.nodes_visited);
}

TEST(Hnsw, ParallelBuildRecall) {
  const auto& f = fixture();
  HnswIndex par(f.base.data(), Fixture::kN, Fixture::kDim, HnswParams{});
  par.build(4);
  const double rs = recall(f.index, f.base.data(), f.queries, 10, 100);
  const double rp = recall(par, f.base.data(), f.queries, 10, 100);
  std::cout << "    recall@10 ef=100: serial " << rs << ", 4 threads " << rp << "\n";
  EXPECT_GE(rp, rs - 0.02);
  // Level assignment depends only on (seed, id): identical to the serial build.
  const auto bs = serialized(f.index), bp = serialized(par);
  EXPECT_EQ(par.max_level(), f.index.max_level());
  EXPECT_TRUE(std::equal(bs.begin() + kHeaderBytes, bs.begin() + kHeaderBytes + Fixture::kN,
                         bp.begin() + kHeaderBytes));
  // The parallel graph passes load()'s structural validation.
  HnswIndex loaded =
      HnswIndex::load(bp.data(), bp.size(), f.base.data(), Fixture::kN, Fixture::kDim);
  EXPECT_EQ(loaded.size(), Fixture::kN);
}

TEST(Hnsw, ConcurrentSearch) {
  const auto& f = fixture();
  std::vector<std::vector<ScoredId>> expected(Fixture::kQueries);
  for (uint32_t q = 0; q < Fixture::kQueries; ++q)
    expected[q] = f.index.search(&f.queries[q * Fixture::kDim], 10, 64);
  std::vector<int> ok(4, 1);
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&, t] {
      for (int rep = 0; rep < 3; ++rep)
        for (uint32_t q = 0; q < Fixture::kQueries; ++q)
          if (!same_results(f.index.search(&f.queries[q * Fixture::kDim], 10, 64), expected[q]))
            ok[t] = 0;
    });
  }
  for (auto& th : threads) th.join();
  for (int t = 0; t < 4; ++t) EXPECT_EQ(ok[t], 1);
}

TEST(Hnsw, FilteredSearchHalfSelectivity) {
  const auto& f = fixture();
  const IdFilter half = [](uint32_t id) { return mix(id) % 100 < 50; };
  for (uint32_t q = 0; q < Fixture::kQueries; ++q) {
    const auto r = f.index.search(&f.queries[q * Fixture::kDim], 10, 100, &half);
    ASSERT_EQ(r.size(), 10u);
    for (const auto& e : r) ASSERT_TRUE(half(e.id));
  }
  const double r = recall(f.index, f.base.data(), f.queries, 10, 100, &half);
  std::cout << "    filtered recall@10 (50%): " << r << "\n";
  EXPECT_GE(r, 0.9);
}

TEST(Hnsw, FilteredSearchLowSelectivity) {
  const auto& f = fixture();
  const IdFilter rare = [](uint32_t id) { return mix(id) % 100 < 5; };
  SearchStats st;
  for (uint32_t q = 0; q < Fixture::kQueries; ++q) {
    const auto r = f.index.search(&f.queries[q * Fixture::kDim], 10, 100, &rare, &st);
    ASSERT_EQ(r.size(), 10u);  // ~250 allowed ids, all reachable
    for (const auto& e : r) ASSERT_TRUE(rare(e.id));
  }
  const double r = recall(f.index, f.base.data(), f.queries, 10, 100, &rare);
  std::cout << "    filtered recall@10 (5%): " << r << ", distance comps/query "
            << st.distance_computations / Fixture::kQueries << "\n";
  EXPECT_GE(r, 0.8);
}

TEST(Hnsw, SerializeLoadRoundTrip) {
  const auto& f = fixture();
  const std::vector<uint8_t> original = serialized(f.index);
  std::vector<uint8_t> copy(original);  // load from an independent buffer
  HnswIndex loaded =
      HnswIndex::load(copy.data(), copy.size(), f.base.data(), Fixture::kN, Fixture::kDim);
  EXPECT_EQ(loaded.size(), f.index.size());
  EXPECT_EQ(loaded.dim(), f.index.dim());
  EXPECT_EQ(loaded.max_level(), f.index.max_level());
  EXPECT_EQ(loaded.entry_point(), f.index.entry_point());
  EXPECT_EQ(loaded.params().M, f.index.params().M);
  EXPECT_EQ(loaded.params().ef_construction, f.index.params().ef_construction);
  EXPECT_EQ(loaded.params().seed, f.index.params().seed);
  EXPECT_EQ(loaded.memory_bytes(), f.index.memory_bytes());
  const IdFilter half = [](uint32_t id) { return id % 2 == 0; };
  for (uint32_t q = 0; q < Fixture::kQueries; ++q) {
    const float* qv = &f.queries[q * Fixture::kDim];
    EXPECT_TRUE(same_results(loaded.search(qv, 10, 80), f.index.search(qv, 10, 80)));
    EXPECT_TRUE(
        same_results(loaded.search(qv, 10, 80, &half), f.index.search(qv, 10, 80, &half)));
  }
  EXPECT_TRUE(serialized(loaded) == original);  // re-serialisation is byte-identical

  // Moving the loaded index keeps it usable and leaves the source empty.
  HnswIndex moved = std::move(loaded);
  EXPECT_EQ(moved.size(), Fixture::kN);
  EXPECT_EQ(moved.search(&f.queries[0], 10, 80).size(), 10u);
  EXPECT_EQ(loaded.size(), 0u);  // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(loaded.search(&f.queries[0], 10, 80).empty());
}

TEST(Hnsw, SerializeEmbeddedInLargerBlob) {
  const uint32_t n = 300, dim = 8;
  const auto base = random_unit_vectors(n, dim, 41);
  HnswIndex idx(base.data(), n, dim, HnswParams{8, 64, 7});
  idx.build(1);
  const auto q = random_unit_vectors(1, dim, 42);
  const auto expected = idx.search(q.data(), 5, 20);

  // Blob at a 4-aligned offset inside a larger writer: loads in place.
  ByteWriter w;
  w.put_u32(0xABCD);
  const size_t start = w.size();
  idx.serialize(w);
  const size_t len = w.size() - start;
  w.put_u32(0xEF01);  // trailing data belonging to someone else
  HnswIndex a = HnswIndex::load(w.data() + start, len, base.data(), n, dim);
  EXPECT_TRUE(same_results(a.search(q.data(), 5, 20), expected));

  // Blob at a misaligned address: rejected rather than read misaligned.
  ByteWriter m;
  m.put_u8(1);
  idx.serialize(m);
  EXPECT_THROW(HnswIndex::load(m.data() + 1, m.size() - 1, base.data(), n, dim));
}

TEST(Hnsw, LoadRejectsCorruption) {
  const uint32_t n = 400, dim = 8;
  const auto base = random_unit_vectors(n, dim, 51);
  HnswIndex idx(base.data(), n, dim, HnswParams{});
  idx.build(1);
  const std::vector<uint8_t> good = serialized(idx);
  EXPECT_EQ(HnswIndex::load(good.data(), good.size(), base.data(), n, dim).size(), n);

  auto bad = good;
  bad[0] = 'X';
  EXPECT_THROW(HnswIndex::load(bad.data(), bad.size(), base.data(), n, dim));
  EXPECT_THROW(HnswIndex::load(good.data(), good.size() - 4, base.data(), n, dim));
  EXPECT_THROW(HnswIndex::load(good.data(), 10, base.data(), n, dim));
  EXPECT_THROW(HnswIndex::load(good.data(), good.size(), base.data(), n - 1, dim));
  EXPECT_THROW(HnswIndex::load(good.data(), good.size(), base.data(), n, dim + 1));
  bad = good;
  bad.resize(bad.size() + 4, 0);  // trailing bytes
  EXPECT_THROW(HnswIndex::load(bad.data(), bad.size(), base.data(), n, dim));

  // Level-0 section starts after the header + n level bytes, padded to 4.
  const size_t l0 = (kHeaderBytes + n + 3) / 4 * 4;
  const uint32_t stride0 = 1 + 2 * idx.params().M;
  bad = good;
  const uint32_t huge_degree = 1000;
  std::memcpy(&bad[l0 + 4 * stride0 * 7], &huge_degree, 4);  // node 7's count
  EXPECT_THROW(HnswIndex::load(bad.data(), bad.size(), base.data(), n, dim));
  bad = good;
  const uint32_t bad_id = n + 5;
  std::memcpy(&bad[l0 + 4 * (stride0 * 9 + 1)], &bad_id, 4);  // node 9's first neighbour
  EXPECT_THROW(HnswIndex::load(bad.data(), bad.size(), base.data(), n, dim));
  bad = good;
  bad[kHeaderBytes + 3] = 200;  // node 3's level
  EXPECT_THROW(HnswIndex::load(bad.data(), bad.size(), base.data(), n, dim));
}

TEST(Hnsw, DeterministicSerialBuild) {
  const uint32_t n = 2000, dim = 16;
  const auto centers = random_unit_vectors(20, dim, 61);
  const auto base = sample_mixture(centers, n, dim, 0.8f, 62);
  HnswParams p;
  p.ef_construction = 100;
  HnswIndex a(base.data(), n, dim, p), b(base.data(), n, dim, p);
  a.build(1);
  b.build(1);
  EXPECT_TRUE(serialized(a) == serialized(b));

  p.seed = 4242;  // a different seed changes the level draw and hence the graph
  HnswIndex c(base.data(), n, dim, p);
  c.build(1);
  EXPECT_FALSE(serialized(a) == serialized(c));
}

TEST(Hnsw, LevelDistribution) {
  // P(level >= 1) = 1/M; P(level >= 2) = 1/M^2.
  const auto& f = fixture();
  const auto bytes = serialized(f.index);
  uint32_t l1 = 0, l2 = 0;
  for (uint32_t i = 0; i < Fixture::kN; ++i) {
    l1 += bytes[kHeaderBytes + i] >= 1;
    l2 += bytes[kHeaderBytes + i] >= 2;
  }
  const double expect1 = Fixture::kN / 16.0, expect2 = Fixture::kN / 256.0;
  EXPECT_NEAR(l1, expect1, 4 * std::sqrt(expect1));
  EXPECT_NEAR(l2, expect2, 4 * std::sqrt(expect2) + 1);
  EXPECT_GE(f.index.max_level(), 2u);
}
