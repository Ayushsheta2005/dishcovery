// HNSW benchmark: build time, memory, and the recall/latency trade-off per ef.
//
//   bench_hnsw --n 100000 --dim 384 --M 16 --efc 200 --threads 8 --queries 1000 --k 100
//              --ef 50,100,200,400 [--vectors file.f32] [--queries-file q.f32]
//              [--clusters 64] [--spread 1.0] [--seed 42] [--selectivity 0.5,0.1,0.02]
//
// Without --vectors, base and query vectors are drawn from a Gaussian mixture on
// the unit sphere (`clusters` random centres, isotropic noise of norm ~`spread`).
// .f32 files are raw little-endian float32, row-major, `dim` floats per row;
// rows are L2-normalised after loading. With --vectors but no --queries-file,
// the last --queries rows of the vectors file are held out as queries. Queries run single-threaded; ground
// truth is exact brute force (computed with --threads threads).
//
// Output: one JSON object per line on stdout.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "util/bytes.h"
#include "vector/distance.h"
#include "vector/hnsw.h"

using namespace dc::vec;
using Clock = std::chrono::steady_clock;

namespace {

struct Args {
  uint32_t n = 100000;
  bool n_given = false;
  uint32_t dim = 384;
  uint32_t M = 16;
  uint32_t efc = 200;
  unsigned threads = 8;
  uint32_t queries = 1000;
  uint32_t k = 100;
  std::vector<uint32_t> efs = {50, 100, 200, 400};
  std::string vectors_file;
  std::string queries_file;
  uint32_t clusters = 64;
  float spread = 1.0f;
  uint64_t seed = 42;
  std::vector<double> selectivities;
};

[[noreturn]] void usage(const char* msg) {
  std::fprintf(stderr,
               "error: %s\nusage: bench_hnsw [--n N] [--dim D] [--M M] [--efc EF] [--threads T]\n"
               "  [--queries Q] [--k K] [--ef a,b,c] [--vectors f.f32] [--queries-file q.f32]\n"
               "  [--clusters C] [--spread S] [--seed S] [--selectivity a,b,c]\n",
               msg);
  std::exit(2);
}

template <typename T>
std::vector<T> parse_list(const std::string& s) {
  std::vector<T> out;
  size_t pos = 0;
  while (pos <= s.size()) {
    const size_t comma = std::min(s.find(',', pos), s.size());
    const std::string item = s.substr(pos, comma - pos);
    if (!item.empty()) out.push_back(static_cast<T>(std::stod(item)));
    pos = comma + 1;
  }
  return out;
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (i + 1 >= argc) usage(("missing value for " + flag).c_str());
    const std::string v = argv[++i];
    try {
      if (flag == "--n") a.n = static_cast<uint32_t>(std::stoul(v)), a.n_given = true;
      else if (flag == "--dim") a.dim = static_cast<uint32_t>(std::stoul(v));
      else if (flag == "--M") a.M = static_cast<uint32_t>(std::stoul(v));
      else if (flag == "--efc") a.efc = static_cast<uint32_t>(std::stoul(v));
      else if (flag == "--threads") a.threads = static_cast<unsigned>(std::stoul(v));
      else if (flag == "--queries") a.queries = static_cast<uint32_t>(std::stoul(v));
      else if (flag == "--k") a.k = static_cast<uint32_t>(std::stoul(v));
      else if (flag == "--ef") a.efs = parse_list<uint32_t>(v);
      else if (flag == "--vectors") a.vectors_file = v;
      else if (flag == "--queries-file") a.queries_file = v;
      else if (flag == "--clusters") a.clusters = static_cast<uint32_t>(std::stoul(v));
      else if (flag == "--spread") a.spread = std::stof(v);
      else if (flag == "--seed") a.seed = std::stoull(v);
      else if (flag == "--selectivity") a.selectivities = parse_list<double>(v);
      else usage(("unknown flag " + flag).c_str());
    } catch (const std::exception&) {
      usage(("bad value for " + flag).c_str());
    }
  }
  if (a.dim == 0 || a.k == 0 || a.clusters == 0 || a.efs.empty()) usage("invalid arguments");
  return a;
}

struct Rng {
  uint64_t state;
  uint64_t next() {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  double uniform() { return (static_cast<double>(next() >> 11) + 0.5) * 0x1.0p-53; }
  float normal() {
    const double u1 = uniform(), u2 = uniform();
    return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
  }
};

std::vector<float> mixture(const std::vector<float>& centers, uint32_t n, uint32_t dim,
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

std::vector<float> read_f32(const std::string& path, uint32_t dim, uint32_t max_rows,
                            uint32_t* rows) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) usage(("cannot open " + path).c_str());
  const size_t bytes = static_cast<size_t>(f.tellg());
  const size_t row_bytes = static_cast<size_t>(dim) * sizeof(float);
  if (bytes % row_bytes != 0) usage((path + ": size is not a multiple of dim*4").c_str());
  *rows = static_cast<uint32_t>(std::min<size_t>(bytes / row_bytes, max_rows));
  std::vector<float> v(static_cast<size_t>(*rows) * dim);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
  for (uint32_t i = 0; i < *rows; ++i) normalize(&v[static_cast<size_t>(i) * dim], dim);
  return v;
}

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Runs fn(i) for i in [0, n) on `threads` threads.
template <typename Fn>
void parallel_for(uint32_t n, unsigned threads, Fn fn) {
  std::vector<std::thread> pool;
  threads = std::max(1u, std::min<unsigned>(threads, std::max(1u, n)));
  for (unsigned t = 0; t < threads; ++t)
    pool.emplace_back([&, t] {
      for (uint32_t i = t; i < n; i += threads) fn(i);
    });
  for (auto& th : pool) th.join();
}

uint64_t mix(uint64_t x) { return Rng{x}.next(); }

struct SweepResult {
  double recall, mean_us, p50_us, p99_us, qps, dist, hops;
};

SweepResult run_queries(const HnswIndex& idx, const std::vector<float>& queries, uint32_t nq,
                        uint32_t dim, uint32_t k, uint32_t ef, const IdFilter* filter,
                        const std::vector<std::vector<ScoredId>>& truth) {
  std::vector<double> lat(nq);
  SearchStats st;
  double hits = 0, total = 0;
  for (uint32_t q = 0; q < nq; ++q) {
    const auto t0 = Clock::now();
    const auto r = idx.search(&queries[static_cast<size_t>(q) * dim], k, ef, filter, &st);
    lat[q] = seconds_since(t0) * 1e6;
    std::unordered_set<uint32_t> exact;
    for (const auto& e : truth[q]) exact.insert(e.id);
    for (const auto& e : r) hits += exact.count(e.id);
    total += static_cast<double>(truth[q].size());
  }
  double sum = 0;
  for (double l : lat) sum += l;
  std::sort(lat.begin(), lat.end());
  auto pct = [&](double p) { return lat[std::min<size_t>(lat.size() - 1, size_t(p * lat.size()))]; };
  return {total > 0 ? hits / total : 1.0, sum / nq, pct(0.50), pct(0.99), nq / (sum * 1e-6),
          double(st.distance_computations) / nq, double(st.nodes_visited) / nq};
}

std::vector<std::vector<ScoredId>> ground_truth(const std::vector<float>& base, uint32_t n,
                                                const std::vector<float>& queries, uint32_t nq,
                                                uint32_t dim, uint32_t k, unsigned threads,
                                                const IdFilter* filter) {
  std::vector<std::vector<ScoredId>> truth(nq);
  parallel_for(nq, threads, [&](uint32_t q) {
    truth[q] = brute_force_topk(base.data(), n, dim, &queries[static_cast<size_t>(q) * dim], k,
                                filter);
  });
  return truth;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse_args(argc, argv);

  // ---- data
  auto t0 = Clock::now();
  uint32_t n = a.n, nq = a.queries;
  std::vector<float> base, queries;
  if (a.vectors_file.empty()) {
    Rng rng{a.seed};
    std::vector<float> centers(static_cast<size_t>(a.clusters) * a.dim);
    for (float& x : centers) x = rng.normal();
    for (uint32_t c = 0; c < a.clusters; ++c) normalize(&centers[size_t{c} * a.dim], a.dim);
    base = mixture(centers, n, a.dim, a.spread, a.seed + 1);
    if (a.queries_file.empty()) queries = mixture(centers, nq, a.dim, a.spread, a.seed + 2);
  } else {
    // Without a queries file, the last `queries` rows are held out as queries.
    const bool holdout = a.queries_file.empty();
    const uint64_t want = a.n_given ? uint64_t{a.n} + (holdout ? nq : 0) : UINT32_MAX;
    uint32_t rows = 0;
    base = read_f32(a.vectors_file, a.dim, static_cast<uint32_t>(std::min<uint64_t>(want, UINT32_MAX)),
                    &rows);
    n = rows;
    if (holdout) {
      if (rows <= nq) usage("vectors file too small to hold out the queries");
      n = rows - nq;
      queries.assign(base.begin() + static_cast<ptrdiff_t>(size_t{n} * a.dim), base.end());
      base.resize(size_t{n} * a.dim);
    }
  }
  if (!a.queries_file.empty()) queries = read_f32(a.queries_file, a.dim, a.queries, &nq);
  if (n == 0 || nq == 0) usage("need at least one vector and one query");
  std::printf(
      "{\"event\":\"data\",\"n\":%u,\"dim\":%u,\"queries\":%u,\"source\":\"%s\",\"clusters\":%u,"
      "\"spread\":%.3f,\"gen_s\":%.3f}\n",
      n, a.dim, nq, a.vectors_file.empty() ? "gaussian_mixture" : "file",
      a.vectors_file.empty() ? a.clusters : 0, a.vectors_file.empty() ? a.spread : 0.0,
      seconds_since(t0));
  std::fflush(stdout);

  // ---- build
  HnswParams params;
  params.M = a.M;
  params.ef_construction = a.efc;
  params.seed = a.seed;
  HnswIndex idx(base.data(), n, a.dim, params);
  t0 = Clock::now();
  idx.build(a.threads);
  const double build_s = seconds_since(t0);
  dc::ByteWriter w;
  idx.serialize(w);
  t0 = Clock::now();
  HnswIndex loaded = HnswIndex::load(w.data(), w.size(), base.data(), n, a.dim);
  const double load_ms = seconds_since(t0) * 1e3;
  std::printf(
      "{\"event\":\"build\",\"n\":%u,\"dim\":%u,\"M\":%u,\"ef_construction\":%u,\"threads\":%u,"
      "\"build_s\":%.3f,\"inserts_per_s\":%.0f,\"memory_bytes\":%zu,\"bytes_per_node\":%.1f,"
      "\"vector_bytes\":%zu,\"max_level\":%u,\"serialized_bytes\":%zu,\"load_validate_ms\":%.2f}\n",
      n, a.dim, a.M, a.efc, a.threads, build_s, n / build_s, idx.memory_bytes(),
      double(idx.memory_bytes()) / n, base.size() * sizeof(float), idx.max_level(), w.size(),
      load_ms);
  std::fflush(stdout);

  // ---- exact ground truth (+ single-thread brute-force latency as a baseline)
  t0 = Clock::now();
  const auto truth = ground_truth(base, n, queries, nq, a.dim, a.k, a.threads, nullptr);
  const double gt_s = seconds_since(t0);
  const uint32_t bf_probe = std::min<uint32_t>(nq, 50);
  t0 = Clock::now();
  for (uint32_t q = 0; q < bf_probe; ++q)
    brute_force_topk(base.data(), n, a.dim, &queries[size_t{q} * a.dim], a.k);
  std::printf(
      "{\"event\":\"brute_force\",\"k\":%u,\"ground_truth_s\":%.3f,\"single_thread_mean_us\":%.1f}"
      "\n",
      a.k, gt_s, seconds_since(t0) * 1e6 / bf_probe);
  std::fflush(stdout);

  // ---- ef sweep (unfiltered, single thread, on the zero-copy loaded index)
  for (uint32_t ef : a.efs) {
    const SweepResult r = run_queries(loaded, queries, nq, a.dim, a.k, ef, nullptr, truth);
    std::printf(
        "{\"event\":\"search\",\"ef\":%u,\"effective_ef\":%u,\"k\":%u,\"queries\":%u,"
        "\"recall\":%.4f,\"mean_us\":%.1f,\"p50_us\":%.1f,\"p99_us\":%.1f,\"qps\":%.0f,"
        "\"dist_per_query\":%.1f,\"hops_per_query\":%.1f}\n",
        ef, std::max(ef, a.k), a.k, nq, r.recall, r.mean_us, r.p50_us, r.p99_us, r.qps, r.dist,
        r.hops);
    std::fflush(stdout);
  }

  // ---- optional filtered sweep: pseudo-random id filter at each selectivity
  for (double sel : a.selectivities) {
    const uint64_t cutoff = static_cast<uint64_t>(sel * 1e6);
    const IdFilter filter = [cutoff](uint32_t id) { return mix(id) % 1000000 < cutoff; };
    const auto ftruth = ground_truth(base, n, queries, nq, a.dim, a.k, a.threads, &filter);
    for (uint32_t ef : a.efs) {
      const SweepResult r = run_queries(loaded, queries, nq, a.dim, a.k, ef, &filter, ftruth);
      std::printf(
          "{\"event\":\"filtered_search\",\"selectivity\":%.4f,\"ef\":%u,\"k\":%u,"
          "\"recall\":%.4f,\"mean_us\":%.1f,\"p50_us\":%.1f,\"p99_us\":%.1f,\"qps\":%.0f,"
          "\"dist_per_query\":%.1f}\n",
          sel, ef, a.k, r.recall, r.mean_us, r.p50_us, r.p99_us, r.qps, r.dist);
      std::fflush(stdout);
    }
  }
  return 0;
}
