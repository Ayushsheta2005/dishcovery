// Micro-benchmark for the posting-block integer codecs (raw, vbyte, pfor).
//
// For each synthetic distribution it splits N integers into blocks of
// kBlockSize (the last block may be partial, as in a real posting list),
// encodes them with every codec and times encode and decode separately.
//
// Usage: bench_codec_micro [num_ints=1000000] [min_seconds_per_measurement=0.2]
//
// Output: one JSON object per line, e.g.
//   {"bench":"codec_micro","dist":"dense_gaps","codec":"pfor","ints":1000000,
//    "bytes":...,"bits_per_int":...,"encode_mb_s":...,"decode_mb_s":...,
//    "encode_mints_s":...,"decode_mints_s":...,"decode_prefix_mints_s":...}
// MB/s are measured on the *uncompressed* data (4 bytes per int, 1 MB = 1e6 B).
// decode_prefix_mints_s additionally runs prefix_sum on every block, i.e. the
// cost of turning docID gaps back into docIDs. Each figure is the best of
// repeated passes over the whole data set (which fits in L2 at the default N).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "codec/block_codec.h"
#include "util/bytes.h"

using namespace dc;
using namespace dc::codec;

namespace {

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }  // [0, 1)
  // Geometric number of failures before the first success, success prob p.
  uint32_t geometric(double p) {
    const double u = 1.0 - uniform();  // (0, 1]
    return static_cast<uint32_t>(std::floor(std::log(u) / std::log1p(-p)));
  }
};

struct Distribution {
  const char* name;
  std::function<uint32_t(Rng&)> sample;
};

const std::vector<Distribution>& distributions() {
  static const std::vector<Distribution> d = {
      // docID gaps of a frequent term: mean gap 4 (gap >= 1).
      {"dense_gaps", [](Rng& r) { return 1 + r.geometric(1.0 / 4); }},
      // docID gaps of a rare term: mean gap ~1000.
      {"sparse_gaps", [](Rng& r) { return 1 + r.geometric(1.0 / 1000); }},
      // Term frequencies: discrete power law, P(tf >= k) ~ k^-1.5, ~65% ones.
      {"tf",
       [](Rng& r) {
         const double u = 1.0 - r.uniform();
         return static_cast<uint32_t>(std::min(65535.0, std::floor(std::pow(u, -1.0 / 1.5))));
       }},
      // Small gaps with 1% huge outliers (exercise PFor exceptions).
      {"outlier_gaps",
       [](Rng& r) {
         if (r.next() % 100 == 0) return static_cast<uint32_t>(r.next() % (1u << 24));
         return 1 + r.geometric(1.0 / 3);
       }},
      // Incompressible worst case.
      {"uniform32", [](Rng& r) { return static_cast<uint32_t>(r.next()); }},
  };
  return d;
}

using Clock = std::chrono::steady_clock;

// Runs fn repeatedly for at least min_seconds (and >= 3 passes); returns the
// fastest pass in seconds.
template <typename Fn>
double best_time(double min_seconds, Fn&& fn) {
  double best = 1e30, total = 0;
  for (int pass = 0; pass < 3 || total < min_seconds; ++pass) {
    const auto t0 = Clock::now();
    fn();
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    best = std::min(best, s);
    total += s;
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  const size_t n_ints = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000000;
  const double min_seconds = argc > 2 ? std::atof(argv[2]) : 0.2;
  if (n_ints == 0) {
    std::fprintf(stderr, "num_ints must be > 0\n");
    return 2;
  }

  const CodecId codecs[] = {CodecId::kRaw, CodecId::kVByte, CodecId::kPFor};
  volatile uint32_t sink = 0;  // volatile stores keep the decode loops alive

  for (const Distribution& dist : distributions()) {
    Rng rng(0xD15C0 + std::hash<std::string>{}(dist.name) % 1000);
    std::vector<uint32_t> data(n_ints);
    for (auto& x : data) x = dist.sample(rng);

    // Block boundaries.
    std::vector<size_t> block_n;
    for (size_t off = 0; off < n_ints; off += kBlockSize)
      block_n.push_back(std::min(kBlockSize, n_ints - off));

    for (CodecId id : codecs) {
      ByteWriter w(n_ints * 5);
      auto encode_all = [&] {
        w.clear();
        const uint32_t* in = data.data();
        for (size_t n : block_n) {
          encode_block(id, in, n, w);
          in += n;
        }
      };
      const double enc_s = best_time(min_seconds, encode_all);

      // Decode from an exact-size copy; verify once.
      std::vector<uint8_t> enc(w.buffer());
      std::vector<uint32_t> check(n_ints);
      {
        const uint8_t* p = enc.data();
        uint32_t* out = check.data();
        for (size_t n : block_n) {
          p = decode_block(id, p, n, out);
          out += n;
        }
        if (p != enc.data() + enc.size() || check != data) {
          std::fprintf(stderr, "roundtrip mismatch: %s/%s\n", dist.name, codec_name(id));
          return 1;
        }
      }

      alignas(64) uint32_t out[kBlockSize];
      auto decode_all = [&] {
        const uint8_t* p = enc.data();
        uint32_t acc = 0;
        for (size_t n : block_n) {
          p = decode_block(id, p, n, out);
          acc += out[n - 1];
        }
        sink = sink + acc;
      };
      const double dec_s = best_time(min_seconds, decode_all);

      auto decode_prefix_all = [&] {
        const uint8_t* p = enc.data();
        uint32_t base = 0;
        for (size_t n : block_n) {
          p = decode_block(id, p, n, out);
          prefix_sum(out, n, base);
          base = out[n - 1];
        }
        sink = sink + base;
      };
      const double dec_prefix_s = best_time(min_seconds, decode_prefix_all);

      const double ints = static_cast<double>(n_ints);
      std::printf(
          "{\"bench\":\"codec_micro\",\"dist\":\"%s\",\"codec\":\"%s\",\"ints\":%zu,"
          "\"bytes\":%zu,\"bits_per_int\":%.3f,\"encode_mb_s\":%.1f,\"decode_mb_s\":%.1f,"
          "\"encode_mints_s\":%.1f,\"decode_mints_s\":%.1f,\"decode_prefix_mints_s\":%.1f}\n",
          dist.name, codec_name(id), n_ints, enc.size(), 8.0 * enc.size() / ints,
          4 * ints / enc_s / 1e6, 4 * ints / dec_s / 1e6, ints / enc_s / 1e6, ints / dec_s / 1e6,
          ints / dec_prefix_s / 1e6);
      std::fflush(stdout);
    }
  }
  return 0;
}
