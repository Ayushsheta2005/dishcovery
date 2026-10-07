// HNSW index implementation. See hnsw.h for the layout and the paper reference.
#include "vector/hnsw.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>

namespace dc::vec {

// The serialised graph is mapped and used in place, so the host must match the
// on-disk byte order.
static_assert(std::endian::native == std::endian::little, "HNSW format is little-endian");

namespace {

constexpr char kMagic[4] = {'H', 'N', 'S', '1'};
constexpr uint32_t kMaxM = 1024;
// Hard cap on a node's level (stored as u8). P(level >= 31) = M^-31: never hit
// in practice, it only bounds the damage of an extreme random draw.
constexpr uint32_t kMaxLevel = 31;
constexpr uint32_t kNoOffset = std::numeric_limits<uint32_t>::max();

uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Alg. 1 line 4: l = floor(-ln(U) * mL) with U ~ Uniform(0, 1] and mL = 1/ln(M).
// U is a pure function of (seed, id), so the level of every node is fixed before
// insertion starts and is identical for serial and parallel builds.
uint32_t random_level(uint64_t seed, uint32_t id, double ml) {
  const uint64_t h = splitmix64(seed ^ splitmix64(id));
  const double u = static_cast<double>((h >> 11) + 1) * 0x1.0p-53;  // (0, 1]
  const double level = std::floor(-std::log(u) * ml);
  return level >= kMaxLevel ? kMaxLevel : static_cast<uint32_t>(level);
}

inline void prefetch(const void* p) {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(p);
#else
  (void)p;
#endif
}

// Total order "a is closer than b" (higher score; ties broken by lower id so
// results are deterministic).
inline bool closer(const ScoredId& a, const ScoredId& b) {
  return a.score > b.score || (a.score == b.score && a.id < b.id);
}
// Heap comparators: std heaps keep the comparator-maximum at front().
struct NearestOnTop {  // candidate queue C: front() = closest
  bool operator()(const ScoredId& a, const ScoredId& b) const { return closer(b, a); }
};
struct FurthestOnTop {  // result set W: front() = furthest
  bool operator()(const ScoredId& a, const ScoredId& b) const { return closer(a, b); }
};

// Byte-sized test-and-test-and-set lock guarding one node's adjacency during a
// parallel build. Critical sections are short (copying <= 2M+1 words, or one
// heuristic prune), so spinning beats a futex and costs 1 byte per node.
class SpinLock {
 public:
  void lock() noexcept {
    for (;;) {
      if (!flag_.exchange(true, std::memory_order_acquire)) return;
      for (unsigned spins = 0; flag_.load(std::memory_order_relaxed);) {
        if (++spins >= 64) {
          std::this_thread::yield();
          spins = 0;
        }
      }
    }
  }
  void unlock() noexcept { flag_.store(false, std::memory_order_release); }

 private:
  std::atomic<bool> flag_{false};
};

// Locks `l` if non-null (the serial build passes null and skips locking).
class MaybeLock {
 public:
  explicit MaybeLock(SpinLock* l) : l_(l) {
    if (l_) l_->lock();
  }
  ~MaybeLock() {
    if (l_) l_->unlock();
  }
  MaybeLock(const MaybeLock&) = delete;
  MaybeLock& operator=(const MaybeLock&) = delete;

 private:
  SpinLock* l_;
};

}  // namespace

// Per-search (or per-build-thread) working memory, recycled through a pool so a
// query never allocates in steady state and never clears an O(n) array.
struct HnswIndex::Scratch {
  explicit Scratch(uint32_t n, uint32_t stride0) : visited(n, 0) {
    nbr_copy.reserve(stride0);
    fresh.reserve(stride0);
  }

  // Generation-stamped visited set: a node is visited iff visited[id] == epoch.
  // Bumping the epoch "clears" the set in O(1); the array is only zeroed when
  // the 16-bit epoch wraps (once every 65535 searches).
  void new_epoch() {
    if (++epoch == 0) {
      std::fill(visited.begin(), visited.end(), uint16_t{0});
      epoch = 1;
    }
  }
  // Marks `id` visited; returns true if it already was.
  bool test_and_mark(uint32_t id) {
    if (visited[id] == epoch) return true;
    visited[id] = epoch;
    return false;
  }

  std::vector<uint16_t> visited;
  uint16_t epoch = 0;
  std::vector<ScoredId> cand;        // C: candidate heap (NearestOnTop)
  std::vector<ScoredId> top;         // W: result heap (FurthestOnTop)
  std::vector<ScoredId> entry;       // entry points for the next layer during insertion
  std::vector<ScoredId> sorted;      // W sorted closest-first
  std::vector<ScoredId> selected;    // heuristic output for the inserted node
  std::vector<ScoredId> prune_in;    // overflowing neighbour list being shrunk
  std::vector<ScoredId> prune_out;   // its heuristic output
  std::vector<uint32_t> nbr_copy;    // adjacency snapshot taken under a node lock
  std::vector<uint32_t> fresh;       // unvisited neighbours of the expanded node
};

class HnswIndex::ScratchPool {
 public:
  ScratchPool(uint32_t n, uint32_t stride0) : n_(n), stride0_(stride0) {}

  class Lease {
   public:
    Lease(ScratchPool& pool, std::unique_ptr<Scratch> s) : pool_(pool), s_(std::move(s)) {}
    ~Lease() { pool_.release(std::move(s_)); }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Scratch& operator*() const { return *s_; }

   private:
    ScratchPool& pool_;
    std::unique_ptr<Scratch> s_;
  };

  Lease acquire() {
    {
      std::lock_guard<std::mutex> g(mu_);
      if (!free_.empty()) {
        std::unique_ptr<Scratch> s = std::move(free_.back());
        free_.pop_back();
        return Lease(*this, std::move(s));
      }
    }
    return Lease(*this, std::make_unique<Scratch>(n_, stride0_));
  }

 private:
  void release(std::unique_ptr<Scratch> s) {
    std::lock_guard<std::mutex> g(mu_);
    free_.push_back(std::move(s));
  }

  const uint32_t n_;
  const uint32_t stride0_;
  std::mutex mu_;
  std::vector<std::unique_ptr<Scratch>> free_;
};

// Synchronisation used only while build() runs.
struct HnswIndex::BuildState {
  BuildState(uint32_t n, bool concurrent)
      : locks(concurrent ? std::make_unique<SpinLock[]>(n) : nullptr) {}
  // Per-node adjacency lock (all layers of a node share one), or null when serial.
  SpinLock* lock_for(uint32_t id) const { return locks ? &locks[id] : nullptr; }
  bool concurrent() const { return locks != nullptr; }

  std::unique_ptr<SpinLock[]> locks;
  std::mutex global;  // guards entry_point_ / max_level_
};

struct HnswIndex::Counters {
  uint64_t distances = 0;
  uint64_t hops = 0;
};

// ---------------------------------------------------------------------------
// Construction / lifetime

HnswIndex::HnswIndex() { init_layout(); }

HnswIndex::HnswIndex(const float* vectors, uint32_t n, uint32_t dim, HnswParams params)
    : vectors_(vectors), n_(n), dim_(dim), params_(params) {
  DC_CHECK(params_.M >= 2 && params_.M <= kMaxM, "HNSW M must be in [2, 1024]");
  DC_CHECK(n == 0 || vectors != nullptr, "HNSW vectors must be non-null");
  DC_CHECK(n == 0 || dim > 0, "HNSW dim must be positive");
  init_layout();
  pool_ = std::make_unique<ScratchPool>(n_, stride0_);
}

HnswIndex::~HnswIndex() = default;

HnswIndex::HnswIndex(HnswIndex&& other) noexcept : HnswIndex() { swap(other); }

HnswIndex& HnswIndex::operator=(HnswIndex&& other) noexcept {
  HnswIndex tmp(std::move(other));
  swap(tmp);
  return *this;
}

void HnswIndex::swap(HnswIndex& o) noexcept {
  using std::swap;
  swap(vectors_, o.vectors_);
  swap(n_, o.n_);
  swap(dim_, o.dim_);
  swap(params_, o.params_);
  swap(m0_, o.m0_);
  swap(stride0_, o.stride0_);
  swap(strideu_, o.strideu_);
  swap(max_level_, o.max_level_);
  swap(entry_point_, o.entry_point_);
  swap(built_, o.built_);
  swap(level0_owned_, o.level0_owned_);  // buffers keep their address, so level0_ stays valid
  swap(level0_, o.level0_);
  swap(levels_, o.levels_);
  swap(upper_offset_, o.upper_offset_);
  swap(upper_, o.upper_);
  swap(pool_, o.pool_);
}

void HnswIndex::init_layout() {
  m0_ = 2 * params_.M;
  stride0_ = 1 + m0_;
  strideu_ = 1 + params_.M;
}

uint64_t HnswIndex::compute_upper_offsets() {
  upper_offset_.assign(n_, kNoOffset);
  uint64_t blocks = 0;
  for (uint32_t i = 0; i < n_; ++i) {
    if (levels_[i] == 0) continue;
    upper_offset_[i] = static_cast<uint32_t>(blocks);
    blocks += levels_[i];
    DC_CHECK(blocks < kNoOffset, "HNSW upper layers too large");
  }
  return blocks;
}

void HnswIndex::assign_levels_and_allocate() {
  const double ml = 1.0 / std::log(static_cast<double>(params_.M));
  levels_.resize(n_);
  for (uint32_t i = 0; i < n_; ++i)
    levels_[i] = static_cast<uint8_t>(random_level(params_.seed, i, ml));
  level0_owned_.assign(static_cast<size_t>(n_) * stride0_, 0u);
  level0_ = level0_owned_.data();
  const uint64_t blocks = compute_upper_offsets();
  upper_.assign(static_cast<size_t>(blocks) * strideu_, 0u);
}

const uint32_t* HnswIndex::list(uint32_t id, uint32_t layer) const {
  if (layer == 0) return level0_ + static_cast<size_t>(id) * stride0_;
  return upper_.data() + (static_cast<size_t>(upper_offset_[id]) + layer - 1) * strideu_;
}

uint32_t* HnswIndex::mutable_list(uint32_t id, uint32_t layer) {
  // Only called during build(), when level0_ points into level0_owned_.
  if (layer == 0) return level0_owned_.data() + static_cast<size_t>(id) * stride0_;
  return upper_.data() + (static_cast<size_t>(upper_offset_[id]) + layer - 1) * strideu_;
}

size_t HnswIndex::memory_bytes() const {
  return static_cast<size_t>(n_) * stride0_ * sizeof(uint32_t) + levels_.size() +
         upper_offset_.size() * sizeof(uint32_t) + upper_.size() * sizeof(uint32_t);
}

// ---------------------------------------------------------------------------
// Graph traversal

// Returns the adjacency block [count, ids...] of `id` at `layer`. During a
// parallel build the block may be rewritten by other threads, so it is copied
// into scratch under the node's lock; afterwards the graph is immutable and the
// block is read in place.
template <bool kConcurrent>
const uint32_t* HnswIndex::neighbors(uint32_t id, uint32_t layer, Scratch& s,
                                     BuildState* bs) const {
  const uint32_t* blk = list(id, layer);
  if constexpr (!kConcurrent) {
    (void)s;
    (void)bs;
    return blk;
  } else {
    MaybeLock g(bs->lock_for(id));
    s.nbr_copy.assign(blk, blk + 1 + blk[0]);
    return s.nbr_copy.data();
  }
}

// SEARCH-LAYER with ef = 1 (used by Alg. 1 lines 5-7 and Alg. 5 lines 4-6):
// hill-climb from `cur` to a local maximum of similarity on `layer`.
template <bool kConcurrent>
void HnswIndex::greedy_descend(const float* q, ScoredId& cur, uint32_t layer, Scratch& s,
                               BuildState* bs, Counters& c) const {
  for (bool improved = true; improved;) {
    improved = false;
    ++c.hops;
    const uint32_t* blk = neighbors<kConcurrent>(cur.id, layer, s, bs);
    const uint32_t cnt = blk[0];
    c.distances += cnt;
    for (uint32_t j = 1; j <= cnt; ++j) {
      if (j < cnt) prefetch(vec(blk[j + 1]));
      const float d = dot(q, vec(blk[j]), dim_);
      if (d > cur.score) {
        cur = {blk[j], d};
        improved = true;
      }
    }
  }
}

// Alg. 2 SEARCH-LAYER. On entry s.cand (C) and s.top (W) hold the entry points
// as heaps and those points are marked visited in the current epoch. On return
// s.top holds up to ef nearest elements (passing `filter`, if given) as a heap.
//
// With a filter, only passing ids enter W, but every improving node enters C,
// so filtered-out nodes still serve as stepping stones. Termination (line 7) is
// unchanged: stop once W holds ef elements and the closest unexpanded candidate
// is further than W's furthest element.
template <bool kConcurrent>
void HnswIndex::search_layer(const float* q, uint32_t layer, uint32_t ef, const IdFilter* filter,
                             Scratch& s, BuildState* bs, Counters& c) const {
  std::vector<ScoredId>& cand = s.cand;
  std::vector<ScoredId>& top = s.top;
  while (!cand.empty()) {
    const ScoredId cur = cand.front();
    if (top.size() >= ef && cur.score < top.front().score) break;  // lines 6-8
    std::pop_heap(cand.begin(), cand.end(), NearestOnTop{});
    cand.pop_back();
    ++c.hops;

    // Collect unvisited neighbours first and prefetch their vectors, so the
    // memory loads overlap instead of serialising behind each dot product.
    const uint32_t* blk = neighbors<kConcurrent>(cur.id, layer, s, bs);
    const uint32_t cnt = blk[0];
    std::vector<uint32_t>& fresh = s.fresh;
    fresh.clear();
    for (uint32_t j = 1; j <= cnt; ++j) {
      const uint32_t e = blk[j];
      if (s.test_and_mark(e)) continue;
      prefetch(vec(e));
      fresh.push_back(e);
    }
    c.distances += fresh.size();

    for (const uint32_t e : fresh) {  // lines 9-17
      const float d = dot(q, vec(e), dim_);
      if (top.size() < ef || d > top.front().score) {
        cand.push_back({e, d});
        std::push_heap(cand.begin(), cand.end(), NearestOnTop{});
        if (filter == nullptr || (*filter)(e)) {
          top.push_back({e, d});
          std::push_heap(top.begin(), top.end(), FurthestOnTop{});
          if (top.size() > ef) {
            std::pop_heap(top.begin(), top.end(), FurthestOnTop{});
            top.pop_back();
          }
        }
      }
    }
  }
}

// Alg. 4 SELECT-NEIGHBORS-HEURISTIC with extendCandidates = false and
// keepPrunedConnections = false. `cands` holds candidates with their similarity
// to the base element, sorted closest-first. A candidate is kept only if it is
// closer to the base than to every neighbour already kept; otherwise the base
// can reach it through that neighbour and the edge would be redundant. This
// spreads edges across directions and keeps clusters connected to each other.
void HnswIndex::select_neighbors_heuristic(const std::vector<ScoredId>& cands, uint32_t m,
                                           std::vector<ScoredId>& out) const {
  out.clear();
  for (const ScoredId& c : cands) {
    if (out.size() >= m) break;
    const float* vc = vec(c.id);
    bool keep = true;
    for (const ScoredId& r : out) {
      if (dot(vc, vec(r.id), dim_) > c.score) {
        keep = false;
        break;
      }
    }
    if (keep) out.push_back(c);
  }
}

// ---------------------------------------------------------------------------
// Construction (Alg. 1)

void HnswIndex::build(unsigned num_threads) {
  DC_CHECK(!built_, "HnswIndex::build called twice");
  assign_levels_and_allocate();
  if (n_ > 0) {
    // Node 0 seeds the graph: it is the entry point at its own level, with no links.
    entry_point_ = 0;
    max_level_ = levels_[0];
    const unsigned threads = std::clamp(num_threads, 1u, std::max(1u, n_ - 1));
    BuildState bs(n_, threads > 1);
    if (threads == 1) {
      ScratchPool::Lease lease = pool_->acquire();
      for (uint32_t i = 1; i < n_; ++i) insert(i, bs, *lease);
    } else {
      std::atomic<uint64_t> next{1};
      std::mutex err_mu;
      std::exception_ptr err;
      auto worker = [&] {
        try {
          ScratchPool::Lease lease = pool_->acquire();
          for (uint64_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < n_;)
            insert(static_cast<uint32_t>(i), bs, *lease);
        } catch (...) {
          std::lock_guard<std::mutex> g(err_mu);
          if (!err) err = std::current_exception();
          next.store(n_, std::memory_order_relaxed);
        }
      };
      std::vector<std::thread> pool;
      pool.reserve(threads);
      for (unsigned t = 0; t < threads; ++t) pool.emplace_back(worker);
      for (std::thread& t : pool) t.join();
      if (err) std::rethrow_exception(err);
    }
  }
  built_ = true;
}

// Alg. 1 INSERT for node q.
//
// Locking protocol (parallel build): a thread holds at most one node lock at a
// time — adjacency blocks are copied out under the lock before use — so node
// locks can never deadlock. The global lock is only ever taken before any node
// lock. A node whose level exceeds the current max level keeps the global lock
// for its whole insertion (as hnswlib does), so two new top-level nodes cannot
// race to become the entry point and leave each other's upper layers disconnected.
// Other threads can see q while it is half-inserted (e.g. carried down as an
// entry point from a layer where q is already linked); they then just find a
// short or empty list on the lower layer, and any edges they add to q are
// merged with q's own selection rather than overwritten.
void HnswIndex::insert(uint32_t q, BuildState& bs, Scratch& s) {
  const float* v = vec(q);
  const uint32_t lq = levels_[q];
  const bool conc = bs.concurrent();
  const uint32_t efc = std::max(params_.ef_construction, params_.M);
  Counters unused;

  std::unique_lock<std::mutex> top_lock(bs.global);
  const uint32_t L = max_level_;
  ScoredId cur{entry_point_, 0.0f};
  if (lq <= L) top_lock.unlock();
  cur.score = dot(v, vec(cur.id), dim_);

  // Lines 5-7: greedy descent through the layers above q's level.
  for (uint32_t l = L; l > lq; --l) {
    if (conc)
      greedy_descend<true>(v, cur, l, s, &bs, unused);
    else
      greedy_descend<false>(v, cur, l, s, &bs, unused);
  }

  // Lines 8-17: on every layer q lives on, find ef_construction candidates,
  // link q to M of them chosen by the heuristic, and add the reverse edges.
  s.entry.assign(1, cur);
  for (uint32_t l = std::min(lq, L) + 1; l-- > 0;) {
    s.new_epoch();
    s.test_and_mark(q);  // q may already be reachable here (parallel build); never self-link
    s.cand.assign(s.entry.begin(), s.entry.end());
    s.top.assign(s.entry.begin(), s.entry.end());
    for (const ScoredId& e : s.entry) s.test_and_mark(e.id);
    std::make_heap(s.cand.begin(), s.cand.end(), NearestOnTop{});
    std::make_heap(s.top.begin(), s.top.end(), FurthestOnTop{});
    if (conc)
      search_layer<true>(v, l, efc, nullptr, s, &bs, unused);
    else
      search_layer<false>(v, l, efc, nullptr, s, &bs, unused);

    s.sorted.assign(s.top.begin(), s.top.end());
    std::sort(s.sorted.begin(), s.sorted.end(), closer);
    select_neighbors_heuristic(s.sorted, params_.M, s.selected);  // line 10

    {  // line 11: q's own edges
      // Usually q's list on this layer is still empty, but in a parallel build
      // another thread may already have linked to q here (it can reach q as an
      // entry point carried down from the layer above), so merge, never overwrite.
      MaybeLock g(bs.lock_for(q));
      for (const ScoredId& e : s.selected) add_link_locked(q, e.id, e.score, l, s);
    }
    for (const ScoredId& e : s.selected) {  // lines 12-16: reverse edges
      MaybeLock g(bs.lock_for(e.id));
      add_link_locked(e.id, q, e.score, l, s);
    }
    s.entry.swap(s.sorted);  // line 17: ep <- W
  }

  if (lq > L) {  // lines 18-19 (still holding the global lock)
    entry_point_ = q;
    max_level_ = lq;
  }
}

// Alg. 1 lines 12-16 for one directed edge: add `to` (with similarity `score`)
// to the adjacency of `from` on `layer`; if `from` then exceeds its max degree,
// shrink its neighbourhood with the heuristic. Caller holds `from`'s lock.
void HnswIndex::add_link_locked(uint32_t from, uint32_t to, float score, uint32_t layer,
                                Scratch& s) {
  const uint32_t mmax = max_degree(layer);
  uint32_t* blk = mutable_list(from, layer);
  const uint32_t cnt = blk[0];
  for (uint32_t j = 1; j <= cnt; ++j)
    if (blk[j] == to) return;
  if (cnt < mmax) {
    blk[1 + cnt] = to;
    blk[0] = cnt + 1;
    return;
  }
  const float* vf = vec(from);
  s.prune_in.clear();
  s.prune_in.push_back({to, score});
  for (uint32_t j = 1; j <= cnt; ++j) s.prune_in.push_back({blk[j], dot(vf, vec(blk[j]), dim_)});
  std::sort(s.prune_in.begin(), s.prune_in.end(), closer);
  select_neighbors_heuristic(s.prune_in, mmax, s.prune_out);
  blk[0] = static_cast<uint32_t>(s.prune_out.size());
  for (size_t i = 0; i < s.prune_out.size(); ++i) blk[1 + i] = s.prune_out[i].id;
}

// ---------------------------------------------------------------------------
// Query (Alg. 5 K-NN-SEARCH)

std::vector<ScoredId> HnswIndex::search(const float* query, uint32_t k, uint32_t ef,
                                        const IdFilter* filter, SearchStats* stats) const {
  std::vector<ScoredId> out;
  if (n_ == 0 || k == 0) return out;
  DC_CHECK(built_, "HnswIndex::search before build()/load()");
  ef = std::max(ef, k);

  ScratchPool::Lease lease = pool_->acquire();
  Scratch& s = *lease;
  Counters c;

  // Lines 4-6: greedy descent from the entry point down to layer 1.
  ScoredId cur{entry_point_, dot(query, vec(entry_point_), dim_)};
  c.distances = 1;
  for (uint32_t l = max_level_; l > 0; --l) greedy_descend<false>(query, cur, l, s, nullptr, c);

  // Line 7: best-first search on layer 0 with ef.
  s.new_epoch();
  s.test_and_mark(cur.id);
  s.cand.assign(1, cur);
  s.top.clear();
  if (filter == nullptr || (*filter)(cur.id)) s.top.push_back(cur);
  search_layer<false>(query, 0, ef, filter, s, nullptr, c);

  // Line 8: the k nearest of W.
  std::sort(s.top.begin(), s.top.end(), closer);
  out.assign(s.top.begin(), s.top.begin() + std::min<size_t>(k, s.top.size()));
  if (stats != nullptr) {
    stats->distance_computations += c.distances;
    stats->nodes_visited += c.hops;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Serialisation

void HnswIndex::serialize(ByteWriter& out) const {
  DC_CHECK(built_ || n_ == 0, "HnswIndex::serialize before build()");
  const size_t start = out.size();
  const uint64_t blocks = upper_.size() / strideu_;
  out.put_bytes(kMagic, sizeof kMagic);
  out.put_u32(params_.M);
  out.put_u32(params_.ef_construction);
  out.put_u64(params_.seed);
  out.put_u32(n_);
  out.put_u32(dim_);
  out.put_u32(max_level_);
  out.put_u32(entry_point_);
  out.put_u64(blocks);
  if (n_ > 0) out.put_bytes(levels_.data(), n_);
  // Pad relative to the blob start so the level-0 array is 4-byte aligned
  // whenever the blob is.
  while ((out.size() - start) % sizeof(uint32_t) != 0) out.put_u8(0);
  if (n_ > 0) out.put_bytes(level0_, static_cast<size_t>(n_) * stride0_ * sizeof(uint32_t));
  if (!upper_.empty()) out.put_bytes(upper_.data(), upper_.size() * sizeof(uint32_t));
}

HnswIndex HnswIndex::load(const uint8_t* data, size_t size, const float* vectors, uint32_t n,
                          uint32_t dim) {
  DC_CHECK(data != nullptr, "HNSW blob is null");
  DC_CHECK(size >= sizeof kMagic && std::memcmp(data, kMagic, sizeof kMagic) == 0,
           "bad HNSW magic");
  ByteReader r(data, size);
  r.skip(sizeof kMagic);
  HnswParams p;
  p.M = r.get_u32();
  p.ef_construction = r.get_u32();
  p.seed = r.get_u64();
  const uint32_t hn = r.get_u32();
  const uint32_t hdim = r.get_u32();
  const uint32_t max_level = r.get_u32();
  const uint32_t ep = r.get_u32();
  const uint64_t blocks = r.get_u64();
  DC_CHECK(p.M >= 2 && p.M <= kMaxM, "HNSW M out of range");
  DC_CHECK(hn == n, "HNSW node count does not match the vectors");
  DC_CHECK(hdim == dim, "HNSW dim does not match the vectors");

  HnswIndex idx(vectors, n, dim, p);
  const uint8_t* lv = r.get_bytes(n);
  idx.levels_.assign(lv, lv + n);
  uint32_t top_level = 0;
  for (uint8_t l : idx.levels_) {
    DC_CHECK(l <= kMaxLevel, "HNSW node level out of range");
    top_level = std::max<uint32_t>(top_level, l);
  }
  DC_CHECK(max_level == top_level, "HNSW max_level inconsistent with levels");
  if (n > 0) {
    DC_CHECK(ep < n && idx.levels_[ep] == max_level, "HNSW entry point invalid");
  } else {
    DC_CHECK(ep == 0, "HNSW entry point invalid");
  }
  DC_CHECK(blocks == idx.compute_upper_offsets(), "HNSW upper-layer size mismatch");

  r.align_to(sizeof(uint32_t));
  const uint8_t* l0 = r.get_bytes(static_cast<size_t>(n) * idx.stride0_ * sizeof(uint32_t));
  DC_CHECK(reinterpret_cast<uintptr_t>(l0) % alignof(uint32_t) == 0,
           "HNSW level-0 section is not 4-byte aligned (align the blob start)");
  idx.level0_ = reinterpret_cast<const uint32_t*>(l0);

  const size_t upper_words = static_cast<size_t>(blocks) * idx.strideu_;
  const uint8_t* up = r.get_bytes(upper_words * sizeof(uint32_t));
  idx.upper_.resize(upper_words);
  if (upper_words > 0) std::memcpy(idx.upper_.data(), up, upper_words * sizeof(uint32_t));
  DC_CHECK(r.remaining() == 0, "trailing bytes after HNSW blob");

  idx.max_level_ = max_level;
  idx.entry_point_ = ep;
  idx.validate_graph();
  idx.built_ = true;
  return idx;
}

// Every neighbour id must be a valid node other than the owner, every degree
// within bounds, and an upper-layer neighbour must itself live on that layer
// (otherwise the traversal would index past its upper-layer blocks).
void HnswIndex::validate_graph() const {
  for (uint32_t i = 0; i < n_; ++i) {
    const uint32_t* blk = list(i, 0);
    DC_CHECK(blk[0] <= m0_, "HNSW level-0 degree out of range");
    for (uint32_t j = 1; j <= blk[0]; ++j)
      DC_CHECK(blk[j] < n_ && blk[j] != i, "HNSW level-0 neighbour id invalid");
    for (uint32_t l = 1; l <= levels_[i]; ++l) {
      const uint32_t* ub = list(i, l);
      DC_CHECK(ub[0] <= params_.M, "HNSW upper-layer degree out of range");
      for (uint32_t j = 1; j <= ub[0]; ++j)
        DC_CHECK(ub[j] < n_ && ub[j] != i && levels_[ub[j]] >= l,
                 "HNSW upper-layer neighbour invalid");
    }
  }
}

// ---------------------------------------------------------------------------

std::vector<ScoredId> brute_force_topk(const float* vectors, uint32_t n, uint32_t dim,
                                       const float* query, uint32_t k, const IdFilter* filter) {
  std::vector<ScoredId> heap;  // FurthestOnTop: front() is the current k-th best
  if (k == 0) return heap;
  heap.reserve(std::min<size_t>(k, n) + 1);
  for (uint32_t i = 0; i < n; ++i) {
    if (filter != nullptr && !(*filter)(i)) continue;
    const ScoredId c{i, dot(query, vectors + static_cast<size_t>(i) * dim, dim)};
    if (heap.size() < k) {
      heap.push_back(c);
      std::push_heap(heap.begin(), heap.end(), FurthestOnTop{});
    } else if (closer(c, heap.front())) {
      std::pop_heap(heap.begin(), heap.end(), FurthestOnTop{});
      heap.back() = c;
      std::push_heap(heap.begin(), heap.end(), FurthestOnTop{});
    }
  }
  std::sort(heap.begin(), heap.end(), closer);
  return heap;
}

}  // namespace dc::vec
