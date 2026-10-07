// Similarity kernels for dense float32 embeddings.
//
// Every embedding stored in a segment is unit-normalised, so the inner product
// equals cosine similarity and "higher = closer". These kernels are written as
// plain loops shaped for the auto-vectoriser (no intrinsics), so the same
// source compiles to NEON on arm64 and SSE/AVX on x86-64.
#pragma once

#include <cmath>
#include <cstddef>

namespace dc::vec {

// Inner product of two `dim`-dimensional vectors.
//
// IEEE float addition is not associative, so without -ffast-math a compiler
// will not vectorise a single-accumulator reduction. Instead we keep kLanes
// independent partial sums: the inner fixed-width loop maps onto 4 NEON
// (or 2 AVX / 4 SSE) registers, and the independent accumulator chains hide
// FMA latency. The result is deterministic for a given dim and build.
inline float dot(const float* __restrict a, const float* __restrict b, size_t dim) {
  constexpr size_t kLanes = 16;
  float acc[kLanes] = {};
  size_t i = 0;
  for (; i + kLanes <= dim; i += kLanes) {
    for (size_t j = 0; j < kLanes; ++j) acc[j] += a[i + j] * b[i + j];
  }
  float tail = 0.0f;
  for (; i < dim; ++i) tail += a[i] * b[i];
  // Pairwise (tree) reduction of the lanes.
  for (size_t w = kLanes / 2; w > 0; w /= 2) {
    for (size_t j = 0; j < w; ++j) acc[j] += acc[j + w];
  }
  return acc[0] + tail;
}

// Scale `v` to unit L2 norm in place. Zero (or denormal-norm) vectors are left
// untouched rather than turned into NaNs.
inline void normalize(float* v, size_t dim) {
  const float sq = dot(v, v, dim);
  if (!(sq > 0.0f) || !std::isfinite(sq)) return;
  const float inv = 1.0f / std::sqrt(sq);
  if (!std::isfinite(inv)) return;
  for (size_t i = 0; i < dim; ++i) v[i] *= inv;
}

}  // namespace dc::vec
