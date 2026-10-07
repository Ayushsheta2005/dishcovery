// Lock-free log-linear latency histogram (HDR-style, ~3% relative error).
//
// Values are recorded in microseconds. Buckets 0..63 are exact; above that each
// power of two is split into 32 linear sub-buckets.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>

namespace dc {

class Histogram {
 public:
  static constexpr int kSubBits = 5;                 // 32 sub-buckets per octave
  static constexpr int kSub = 1 << kSubBits;
  static constexpr int kLinear = 2 * kSub;           // values < 64 are exact
  static constexpr int kOctaves = 40;
  static constexpr int kBuckets = kLinear + kOctaves * kSub;

  Histogram() { reset(); }
  Histogram(const Histogram& o) { copy_from(o); }
  Histogram& operator=(const Histogram& o) {
    copy_from(o);
    return *this;
  }

  void record(uint64_t value_us) {
    counts_[bucket_of(value_us)].fetch_add(1, std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_relaxed);
    sum_.fetch_add(value_us, std::memory_order_relaxed);
    uint64_t prev = max_.load(std::memory_order_relaxed);
    while (value_us > prev && !max_.compare_exchange_weak(prev, value_us, std::memory_order_relaxed)) {
    }
  }

  void merge(const Histogram& o) {
    for (int i = 0; i < kBuckets; ++i)
      counts_[i].fetch_add(o.counts_[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
    count_.fetch_add(o.count(), std::memory_order_relaxed);
    sum_.fetch_add(o.sum_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    uint64_t om = o.max();
    uint64_t prev = max_.load(std::memory_order_relaxed);
    while (om > prev && !max_.compare_exchange_weak(prev, om, std::memory_order_relaxed)) {
    }
  }

  void reset() {
    for (auto& c : counts_) c.store(0, std::memory_order_relaxed);
    count_.store(0);
    sum_.store(0);
    max_.store(0);
  }

  uint64_t count() const { return count_.load(std::memory_order_relaxed); }
  uint64_t max() const { return max_.load(std::memory_order_relaxed); }
  double mean() const {
    uint64_t c = count();
    return c ? static_cast<double>(sum_.load(std::memory_order_relaxed)) / c : 0.0;
  }

  // q in [0, 1]. Returns the representative (midpoint) value of the bucket.
  uint64_t percentile(double q) const {
    uint64_t total = count();
    if (!total) return 0;
    uint64_t rank = static_cast<uint64_t>(q * static_cast<double>(total - 1)) + 1;
    uint64_t seen = 0;
    for (int i = 0; i < kBuckets; ++i) {
      seen += counts_[i].load(std::memory_order_relaxed);
      if (seen >= rank) {
        uint64_t v = bucket_mid(i);
        return v > max() ? max() : v;
      }
    }
    return max();
  }

  // {"count":..,"mean":..,"p50":..,"p90":..,"p95":..,"p99":..,"p999":..,"max":..}
  std::string to_json() const;

  static int bucket_of(uint64_t v) {
    if (v < static_cast<uint64_t>(kLinear)) return static_cast<int>(v);
    int msb = 63 - __builtin_clzll(v);          // >= 6
    int shift = msb - kSubBits;                  // >= 1
    int sub = static_cast<int>((v >> shift) & (kSub - 1));
    int octave = msb - (kSubBits + 1);           // 0 for v in [64,128)
    int b = kLinear + octave * kSub + sub;
    return b < kBuckets ? b : kBuckets - 1;
  }
  static uint64_t bucket_low(int b) {
    if (b < kLinear) return static_cast<uint64_t>(b);
    int octave = (b - kLinear) / kSub;
    int sub = (b - kLinear) % kSub;
    int msb = octave + kSubBits + 1;
    int shift = msb - kSubBits;
    return (static_cast<uint64_t>(kSub + sub)) << shift;
  }
  static uint64_t bucket_mid(int b) {
    if (b < kLinear) return static_cast<uint64_t>(b);
    uint64_t lo = bucket_low(b);
    uint64_t hi = b + 1 < kBuckets ? bucket_low(b + 1) : lo * 2;
    return lo + (hi - lo) / 2;
  }

 private:
  void copy_from(const Histogram& o) {
    for (int i = 0; i < kBuckets; ++i) counts_[i].store(o.counts_[i].load(std::memory_order_relaxed));
    count_.store(o.count());
    sum_.store(o.sum_.load());
    max_.store(o.max());
  }

  std::array<std::atomic<uint64_t>, kBuckets> counts_;
  std::atomic<uint64_t> count_{0};
  std::atomic<uint64_t> sum_{0};
  std::atomic<uint64_t> max_{0};
};

}  // namespace dc
