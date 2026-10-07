#pragma once

#include <chrono>
#include <cstdint>

namespace dc {

inline uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

inline uint64_t wall_ms() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count());
}

class Stopwatch {
 public:
  Stopwatch() : start_(now_ns()) {}
  void reset() { start_ = now_ns(); }
  uint64_t elapsed_ns() const { return now_ns() - start_; }
  uint64_t elapsed_us() const { return elapsed_ns() / 1000; }
  double elapsed_ms() const { return static_cast<double>(elapsed_ns()) / 1e6; }
  double elapsed_s() const { return static_cast<double>(elapsed_ns()) / 1e9; }
  // Returns elapsed µs and restarts (for stage-by-stage timing).
  uint64_t lap_us() {
    uint64_t t = now_ns();
    uint64_t d = (t - start_) / 1000;
    start_ = t;
    return d;
  }

 private:
  uint64_t start_;
};

}  // namespace dc
