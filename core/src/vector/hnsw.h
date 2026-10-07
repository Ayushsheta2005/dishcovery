// HNSW (Hierarchical Navigable Small World) approximate nearest-neighbour index.
//
// Reference: Y. A. Malkov, D. A. Yashunin, "Efficient and robust approximate
// nearest neighbor search using Hierarchical Navigable Small World graphs",
// IEEE TPAMI 2018. Algorithm numbers in comments (Alg. 1-5) refer to that paper.
//
// One HnswIndex covers one segment: n unit-normalised float32 embeddings stored
// row-major in a (typically memory-mapped) buffer that the index borrows.
// Similarity is the inner product (== cosine for unit vectors); higher is closer.
//
// Graph layout
//   * Layer 0 holds every node with max degree M0 = 2*M, stored as one flat
//     fixed-stride uint32 array: node i occupies [count, nbr_0 .. nbr_{M0-1}]
//     at offset i * (1 + M0). This array dominates the graph's size and is
//     referenced in place (zero-copy) when the index is loaded from a segment.
//   * Layers >= 1 hold only nodes whose random level is >= 1 (~1/M of them),
//     each with max degree M, in a small owned array of fixed-size blocks
//     [count, nbr_0 .. nbr_{M-1}], one block per (node, layer >= 1).
//
// Filtered search
//   search() accepts a predicate over node ids. Traversal walks through
//   filtered-out nodes (they still act as stepping stones), but only passing
//   ids enter the result set, and exploration only stops once the result set
//   holds ef passing ids and the next candidate is worse than its furthest
//   element. At low selectivity this means exploring a large part of the
//   graph, so latency degrades towards (and past) a brute-force scan; callers
//   should fall back to brute_force_topk() over the allowed ids when the
//   estimated selectivity is below a threshold (a few percent).
//
// Thread safety
//   build() is the only mutating operation. After it returns (or after load()),
//   concurrent const calls (search() etc.) are safe.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "util/bytes.h"
#include "util/common.h"
#include "vector/distance.h"

namespace dc::vec {

struct HnswParams {
  uint32_t M = 16;                 // max degree on layers >= 1; layer 0 uses 2*M
  uint32_t ef_construction = 200;  // candidate list size during insertion
  uint64_t seed = 42;              // drives the (per-node deterministic) level draw
};

struct ScoredId {
  uint32_t id;
  float score;  // inner product with the query; higher = closer
};

// Work counters. search() *adds* to these, so one struct can aggregate a batch.
struct SearchStats {
  uint64_t distance_computations = 0;  // inner products evaluated
  uint64_t nodes_visited = 0;          // nodes whose adjacency list was expanded (graph hops)
};

using IdFilter = std::function<bool(uint32_t)>;

class HnswIndex {
 public:
  HnswIndex();  // empty index (size() == 0); search() returns nothing
  // `vectors` is borrowed (n x dim row-major, unit-normalised) and must outlive the index.
  HnswIndex(const float* vectors, uint32_t n, uint32_t dim, HnswParams params);
  ~HnswIndex();

  HnswIndex(HnswIndex&& other) noexcept;
  HnswIndex& operator=(HnswIndex&& other) noexcept;
  HnswIndex(const HnswIndex&) = delete;
  HnswIndex& operator=(const HnswIndex&) = delete;

  // Inserts ids 0..n-1 (Alg. 1). Node 0 seeds the graph; the remaining nodes are
  // inserted in parallel when num_threads > 1. Level assignment depends only on
  // (seed, id), and a serial build (num_threads <= 1) is fully deterministic.
  // May be called once.
  void build(unsigned num_threads);

  // Top-k by inner product, sorted by descending score (ties by ascending id).
  // ef is clamped to >= k. If filter != nullptr, only ids with (*filter)(id) == true
  // are returned (see "Filtered search" above). Fewer than k results are returned
  // when fewer than k ids are reachable/allowed. If stats != nullptr, work
  // counters are added to it.
  std::vector<ScoredId> search(const float* query, uint32_t k, uint32_t ef,
                               const IdFilter* filter = nullptr,
                               SearchStats* stats = nullptr) const;

  // Serialised layout (little-endian; offsets relative to the start of the blob):
  //   "HNS1" | u32 M | u32 ef_construction | u64 seed | u32 n | u32 dim
  //   | u32 max_level | u32 entry_point | u64 upper_blocks
  //   | u8 levels[n] | zero pad to a multiple of 4
  //   | u32 level0[n * (1 + 2M)] | u32 upper[upper_blocks * (1 + M)]
  // For the level-0 section to be 4-byte aligned in memory, the blob itself must
  // start at a 4-byte aligned address; callers embedding it in a larger file
  // should call out.align_to(4) (or coarser) before serialize().
  void serialize(ByteWriter& out) const;

  // Zero-copy load of a serialize()d blob of exactly `size` bytes. The level-0
  // adjacency is referenced in place (so `data` must outlive the index and the
  // level-0 section must be 4-byte aligned); levels and upper layers are copied.
  // The whole blob is validated (header, sizes, every neighbour id) so a corrupt
  // file fails with DcError instead of causing out-of-bounds reads; this is one
  // sequential pass over the adjacency.
  static HnswIndex load(const uint8_t* data, size_t size, const float* vectors, uint32_t n,
                        uint32_t dim);

  uint32_t size() const { return n_; }
  uint32_t dim() const { return dim_; }
  uint32_t max_level() const { return max_level_; }
  uint32_t entry_point() const { return entry_point_; }
  // Bytes of graph structure (level-0 array, levels, upper layers and their
  // offsets), whether owned or mapped. Excludes the borrowed vectors and
  // per-thread search scratch.
  size_t memory_bytes() const;
  const HnswParams& params() const { return params_; }

 private:
  struct Scratch;
  class ScratchPool;
  struct BuildState;
  struct Counters;

  const float* vec(uint32_t id) const { return vectors_ + static_cast<size_t>(id) * dim_; }
  uint32_t max_degree(uint32_t layer) const { return layer == 0 ? m0_ : params_.M; }
  // Adjacency block [count, nbrs...] of `id` at `layer` (requires level(id) >= layer).
  const uint32_t* list(uint32_t id, uint32_t layer) const;
  uint32_t* mutable_list(uint32_t id, uint32_t layer);

  void init_layout();  // derives strides from params_
  void assign_levels_and_allocate();
  uint64_t compute_upper_offsets();  // fills upper_offset_ from levels_, returns #blocks
  void validate_graph() const;       // bounds/consistency checks after load()

  template <bool kConcurrent>
  const uint32_t* neighbors(uint32_t id, uint32_t layer, Scratch& s, BuildState* bs) const;
  template <bool kConcurrent>
  void greedy_descend(const float* q, ScoredId& cur, uint32_t layer, Scratch& s, BuildState* bs,
                      Counters& c) const;
  template <bool kConcurrent>
  void search_layer(const float* q, uint32_t layer, uint32_t ef, const IdFilter* filter,
                    Scratch& s, BuildState* bs, Counters& c) const;
  void select_neighbors_heuristic(const std::vector<ScoredId>& sorted_cands, uint32_t m,
                                  std::vector<ScoredId>& out) const;
  void insert(uint32_t q, BuildState& bs, Scratch& s);
  void add_link_locked(uint32_t from, uint32_t to, float score, uint32_t layer, Scratch& s);
  void swap(HnswIndex& other) noexcept;

  const float* vectors_ = nullptr;
  uint32_t n_ = 0;
  uint32_t dim_ = 0;
  HnswParams params_;
  uint32_t m0_ = 0;       // layer-0 max degree (2*M)
  uint32_t stride0_ = 0;  // uint32 words per level-0 block (1 + m0_)
  uint32_t strideu_ = 0;  // uint32 words per upper-layer block (1 + M)
  uint32_t max_level_ = 0;
  uint32_t entry_point_ = 0;
  bool built_ = false;

  std::vector<uint32_t> level0_owned_;  // empty when level0_ points into a loaded blob
  const uint32_t* level0_ = nullptr;
  std::vector<uint8_t> levels_;         // random level per node
  std::vector<uint32_t> upper_offset_;  // per node: first block index in upper_ (if level >= 1)
  std::vector<uint32_t> upper_;         // blocks for (node, layer >= 1), node-major
  mutable std::unique_ptr<ScratchPool> pool_;  // reusable visited arrays + heaps
};

// Exact top-k by inner product (sorted descending, ties by ascending id),
// optionally restricted to ids passing `filter`.
std::vector<ScoredId> brute_force_topk(const float* vectors, uint32_t n, uint32_t dim,
                                       const float* query, uint32_t k,
                                       const IdFilter* filter = nullptr);

}  // namespace dc::vec
