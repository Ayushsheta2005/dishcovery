#include "fst/fst.h"

#include <algorithm>
#include <bit>
#include <cstring>

namespace dc::fst {

namespace {

constexpr uint32_t kMagic = 0x54534644;  // "DFST" little-endian
constexpr uint32_t kVersion = 1;

constexpr uint32_t kEmptySlot = 0xFFFFFFFFu;
constexpr uint32_t kPending = 0xFFFFFFFFu;
constexpr size_t kInitialRegister = 1024;  // power of two

// States with at most this many transitions are searched linearly.
constexpr uint32_t kLinearScanMax = 8;

// RMQ block size: the sparse table indexes blocks of 2^kRmqShift weights.
constexpr uint32_t kRmqShift = 5;
constexpr uint32_t kRmqBlock = 1u << kRmqShift;

uint64_t mix64(uint64_t x) {  // splitmix64 finaliser
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

uint64_t hash_transitions(bool is_final, const uint8_t* labels, const uint32_t* targets,
                          size_t n) {
  uint64_t h = is_final ? 0x9E3779B97F4A7C15ULL : 0x632BE59BD9B4E019ULL;
  for (size_t i = 0; i < n; ++i) h = mix64(h ^ (uint64_t{labels[i]} << 32 | targets[i]));
  return h;
}

}  // namespace

// ---------------------------------------------------------------------------
// FstBuilder — Daciuk et al. (2000), algorithm for sorted data.
// ---------------------------------------------------------------------------

FstBuilder::FstBuilder() { reset(); }

void FstBuilder::reset() {
  fst_ = Fst();
  path_.assign(1, PendingNode{});
  path_len_ = 1;
  prev_.clear();
  register_.assign(kInitialRegister, kEmptySlot);
  registered_ = 0;
}

void FstBuilder::add(std::string_view key, uint32_t weight) {
  DC_CHECK(!key.empty(), "FstBuilder: empty key");
  // std::string_view comparison is byte-wise unsigned (char_traits<char>).
  DC_CHECK(fst_.weights_.empty() || std::string_view(prev_) < key,
           "FstBuilder: keys must be added in strictly increasing byte order");
  DC_CHECK(fst_.weights_.size() < 0xFFFFFFFFu, "FstBuilder: too many keys");

  // Longest common prefix with the previous key. Since key > prev_, key cannot
  // be a prefix of prev_, so p < key.size().
  size_t p = 0;
  const size_t lim = std::min(prev_.size(), key.size());
  while (p < lim && prev_[p] == key[p]) ++p;

  // Everything below depth p on the previous path is final: freeze it.
  freeze_tail(p);

  // Append the new suffix as a chain of fresh pending nodes. key[p] is larger
  // than any label already leaving path_[p], so labels stay sorted.
  if (path_.size() < key.size() + 1) path_.resize(key.size() + 1);
  for (size_t i = p; i < key.size(); ++i) {
    path_[i].labels.push_back(static_cast<uint8_t>(key[i]));
    path_[i].targets.push_back(kPending);
  }
  path_len_ = key.size() + 1;
  path_[key.size()].is_final = true;

  prev_.assign(key);
  fst_.weights_.push_back(weight);
}

void FstBuilder::freeze_tail(size_t depth) {
  // Bottom-up: a node can only be compared against the register once all its
  // children are canonical ("replace_or_register" in Daciuk et al.).
  for (size_t d = path_len_ - 1; d > depth; --d) {
    uint32_t id = freeze(path_[d]);
    path_[d - 1].targets.back() = id;
    path_[d].clear();
  }
  path_len_ = depth + 1;
}

uint32_t FstBuilder::freeze(const PendingNode& node) {
  const size_t n = node.labels.size();
  const uint64_t h = hash_transitions(node.is_final, node.labels.data(), node.targets.data(), n);
  const size_t mask = register_.size() - 1;
  for (size_t i = h & mask;; i = (i + 1) & mask) {
    uint32_t id = register_[i];
    if (id == kEmptySlot) {
      id = append_state(node);
      register_[i] = id;
      if (++registered_ * 2 > register_.size()) grow_register();
      return id;
    }
    const Fst::State& s = fst_.states_[id];
    if (s.is_final == node.is_final && s.num_transitions == n &&
        std::equal(node.labels.begin(), node.labels.end(),
                   fst_.labels_.begin() + s.first_transition) &&
        std::equal(node.targets.begin(), node.targets.end(),
                   fst_.targets_.begin() + s.first_transition)) {
      return id;  // equivalent state already exists: share it
    }
  }
}

uint32_t FstBuilder::append_state(const PendingNode& node) {
  DC_CHECK(fst_.states_.size() < 0xFFFFFFFFu, "FstBuilder: too many states");
  DC_CHECK(fst_.labels_.size() + node.labels.size() <= 0xFFFFFFFFu,
           "FstBuilder: too many transitions");
  Fst::State s;
  s.first_transition = static_cast<uint32_t>(fst_.labels_.size());
  s.num_transitions = static_cast<uint16_t>(node.labels.size());
  s.is_final = node.is_final ? 1 : 0;
  uint64_t count = s.is_final;
  for (size_t i = 0; i < node.labels.size(); ++i) {
    fst_.labels_.push_back(node.labels[i]);
    fst_.targets_.push_back(node.targets[i]);
    count += fst_.states_[node.targets[i]].count;
  }
  s.count = static_cast<uint32_t>(count);  // <= number of keys added
  fst_.states_.push_back(s);
  return static_cast<uint32_t>(fst_.states_.size() - 1);
}

uint64_t FstBuilder::state_hash(uint32_t id) const {
  const Fst::State& s = fst_.states_[id];
  return hash_transitions(s.is_final, fst_.labels_.data() + s.first_transition,
                          fst_.targets_.data() + s.first_transition, s.num_transitions);
}

void FstBuilder::grow_register() {
  std::vector<uint32_t> old = std::move(register_);
  register_.assign(old.size() * 2, kEmptySlot);
  const size_t mask = register_.size() - 1;
  for (uint32_t id : old) {
    if (id == kEmptySlot) continue;
    size_t i = state_hash(id) & mask;
    while (register_[i] != kEmptySlot) i = (i + 1) & mask;
    register_[i] = id;
  }
}

Fst FstBuilder::finish() {
  freeze_tail(0);
  fst_.root_ = freeze(path_[0]);
  fst_.num_keys_ = static_cast<uint32_t>(fst_.weights_.size());
  fst_.states_.shrink_to_fit();
  fst_.labels_.shrink_to_fit();
  fst_.targets_.shrink_to_fit();
  fst_.weights_.shrink_to_fit();
  fst_.build_derived();
  Fst result = std::move(fst_);
  reset();
  return result;
}

// ---------------------------------------------------------------------------
// Derived data: per-transition ordinal offsets and the RMQ table.
// ---------------------------------------------------------------------------

void Fst::build_derived() {
  // out_[t] = (state is final) + sum of counts of the earlier siblings of t,
  // i.e. the number of keys accepted from the state that sort before every
  // key routed through t. Strictly increasing within a state since every
  // state accepts at least one key.
  out_.assign(labels_.size(), 0);
  for (const State& s : states_) {
    uint64_t acc = s.is_final;
    for (uint32_t t = s.first_transition; t < s.first_transition + s.num_transitions; ++t) {
      out_[t] = static_cast<uint32_t>(acc);
      acc += states_[targets_[t]].count;
    }
    DC_CHECK(acc == s.count, "Fst: inconsistent state count");
  }
  build_rmq();
}

void Fst::build_rmq() {
  rmq_.clear();
  rmq_level_.clear();
  if (num_keys_ == 0) return;
  const uint32_t nb = (num_keys_ + kRmqBlock - 1) >> kRmqShift;
  const int levels = std::bit_width(nb);  // level l covers 2^l blocks
  size_t total = 0;
  for (int l = 0; l < levels; ++l) total += nb - (1u << l) + 1;
  rmq_.resize(total);
  rmq_level_.resize(levels);

  // Level 0: argmax inside each block.
  for (uint32_t b = 0; b < nb; ++b) {
    uint32_t lo = b << kRmqShift;
    uint32_t hi = std::min(num_keys_, lo + kRmqBlock);
    rmq_[b] = scan_argmax(lo, hi);
  }
  size_t off = 0;
  rmq_level_[0] = 0;
  for (int l = 1; l < levels; ++l) {
    const size_t prev_off = off;
    const uint32_t prev_len = nb - (1u << (l - 1)) + 1;
    off += prev_len;
    rmq_level_[l] = static_cast<uint32_t>(off);
    const uint32_t len = nb - (1u << l) + 1;
    const uint32_t half = 1u << (l - 1);
    for (uint32_t b = 0; b < len; ++b) {
      uint32_t a = rmq_[prev_off + b], c = rmq_[prev_off + b + half];
      rmq_[off + b] = better(a, c) ? a : c;
    }
  }
}

uint32_t Fst::scan_argmax(uint32_t lo, uint32_t hi) const {
  uint32_t best = lo;
  for (uint32_t i = lo + 1; i < hi; ++i)
    if (weights_[i] > weights_[best]) best = i;  // strict: keeps the smallest ord on ties
  return best;
}

uint32_t Fst::range_argmax(uint32_t lo, uint32_t hi) const {
  const uint32_t bl = lo >> kRmqShift, bh = (hi - 1) >> kRmqShift;
  if (bl == bh) return scan_argmax(lo, hi);
  uint32_t best = scan_argmax(lo, (bl + 1) << kRmqShift);
  if (bl + 1 < bh) {  // whole blocks [bl + 1, bh): two overlapping sparse-table lookups
    const uint32_t nblocks = bh - bl - 1;
    const int l = std::bit_width(nblocks) - 1;
    const uint32_t a = rmq_[rmq_level_[l] + bl + 1];
    const uint32_t b = rmq_[rmq_level_[l] + bh - (1u << l)];
    const uint32_t mid = better(a, b) ? a : b;
    if (better(mid, best)) best = mid;
  }
  const uint32_t right = scan_argmax(bh << kRmqShift, hi);
  return better(right, best) ? right : best;
}

// ---------------------------------------------------------------------------
// Exact queries.
// ---------------------------------------------------------------------------

size_t Fst::memory_bytes() const {
  return sizeof(*this) + states_.size() * sizeof(State) + labels_.size() +
         (targets_.size() + out_.size() + weights_.size() + rmq_.size() + rmq_level_.size()) *
             sizeof(uint32_t);
}

uint32_t Fst::lower_bound_label(const State& s, uint8_t c) const {
  const uint8_t* labels = labels_.data() + s.first_transition;
  const uint32_t n = s.num_transitions;
  if (n <= kLinearScanMax) {
    uint32_t i = 0;
    while (i < n && labels[i] < c) ++i;
    return i;
  }
  return static_cast<uint32_t>(std::lower_bound(labels, labels + n, c) - labels);
}

std::optional<uint32_t> Fst::lookup(std::string_view key) const {
  if (num_keys_ == 0) return std::nullopt;
  uint32_t st = root_, ord = 0;
  for (char ch : key) {
    const uint8_t c = static_cast<uint8_t>(ch);
    const State& s = states_[st];
    const uint32_t i = lower_bound_label(s, c);
    const uint32_t t = s.first_transition + i;
    if (i == s.num_transitions || labels_[t] != c) return std::nullopt;
    ord += out_[t];
    st = targets_[t];
  }
  if (!states_[st].is_final) return std::nullopt;
  return ord;  // the key itself sorts first among the keys accepted from st
}

std::string Fst::key_at(uint32_t ord) const {
  DC_CHECK(ord < num_keys_, "Fst::key_at: ordinal out of range");
  std::string key;
  uint32_t st = root_, r = ord;  // r = rank among keys accepted from st
  while (true) {
    const State& s = states_[st];
    if (s.is_final && r == 0) break;
    // Last transition whose subtree starts at or before rank r.
    const uint32_t* out = out_.data() + s.first_transition;
    const uint32_t n = s.num_transitions;
    uint32_t i;
    if (n <= kLinearScanMax) {
      i = 0;
      while (i + 1 < n && out[i + 1] <= r) ++i;
    } else {
      i = static_cast<uint32_t>(std::upper_bound(out, out + n, r) - out) - 1;
    }
    const uint32_t t = s.first_transition + i;
    key.push_back(static_cast<char>(labels_[t]));
    r -= out[i];
    st = targets_[t];
  }
  return key;
}

uint32_t Fst::weight_at(uint32_t ord) const {
  DC_CHECK(ord < num_keys_, "Fst::weight_at: ordinal out of range");
  return weights_[ord];
}

std::pair<uint32_t, uint32_t> Fst::prefix_range(std::string_view prefix) const {
  if (num_keys_ == 0) return {0, 0};
  uint32_t st = root_, ord = 0;
  for (char ch : prefix) {
    const uint8_t c = static_cast<uint8_t>(ch);
    const State& s = states_[st];
    const uint32_t i = lower_bound_label(s, c);
    const uint32_t t = s.first_transition + i;
    if (i == s.num_transitions || labels_[t] != c) {
      // No key continues with c: everything from st below c sorts before.
      ord += (i < s.num_transitions) ? out_[t] : s.count;
      return {ord, ord};
    }
    ord += out_[t];
    st = targets_[t];
  }
  return {ord, ord + states_[st].count};
}

// ---------------------------------------------------------------------------
// Ranking: best `limit` ordinals over a union of ranges.
// ---------------------------------------------------------------------------

std::vector<Fst::Ranked> Fst::rank_ranges(std::vector<RangeHit>& hits, uint32_t limit) const {
  std::vector<Ranked> out;
  if (limit == 0 || hits.empty()) return out;
  std::sort(hits.begin(), hits.end(),
            [](const RangeHit& a, const RangeHit& b) { return a.distance < b.distance; });

  // Each heap entry is a sub-range together with its argmax. Popping the best
  // entry emits its argmax and splits the range around it, so producing k
  // results costs k pops and at most 2k RMQ queries.
  struct Cand {
    uint32_t best, lo, hi;
  };
  auto lower_priority = [this](const Cand& a, const Cand& b) { return better(b.best, a.best); };
  std::vector<Cand> heap;
  size_t i = 0;
  while (i < hits.size() && out.size() < limit) {
    const uint32_t d = hits[i].distance;
    heap.clear();
    for (; i < hits.size() && hits[i].distance == d; ++i)
      if (hits[i].lo < hits[i].hi)
        heap.push_back({range_argmax(hits[i].lo, hits[i].hi), hits[i].lo, hits[i].hi});
    std::make_heap(heap.begin(), heap.end(), lower_priority);
    while (!heap.empty() && out.size() < limit) {
      std::pop_heap(heap.begin(), heap.end(), lower_priority);
      const Cand c = heap.back();
      heap.pop_back();
      out.push_back({c.best, d});
      if (c.lo < c.best) {
        heap.push_back({range_argmax(c.lo, c.best), c.lo, c.best});
        std::push_heap(heap.begin(), heap.end(), lower_priority);
      }
      if (c.best + 1 < c.hi) {
        heap.push_back({range_argmax(c.best + 1, c.hi), c.best + 1, c.hi});
        std::push_heap(heap.begin(), heap.end(), lower_priority);
      }
    }
  }
  return out;
}

std::vector<Completion> Fst::top_k_prefix(std::string_view prefix, uint32_t k) const {
  std::vector<Completion> result;
  const auto [lo, hi] = prefix_range(prefix);
  if (lo == hi || k == 0) return result;
  std::vector<RangeHit> hits{{lo, hi, 0}};
  for (const Ranked& r : rank_ranges(hits, k))
    result.push_back({key_at(r.ord), weights_[r.ord], r.ord});
  return result;
}

std::vector<FuzzyMatch> Fst::to_matches(const std::vector<Ranked>& ranked) const {
  std::vector<FuzzyMatch> result;
  result.reserve(ranked.size());
  for (const Ranked& r : ranked) result.push_back({key_at(r.ord), r.distance, weights_[r.ord], r.ord});
  return result;
}

// ---------------------------------------------------------------------------
// Fuzzy queries: Levenshtein-automaton simulation over the DAWG.
// ---------------------------------------------------------------------------
//
// Row i of the DP table holds D(i, j) = distance between the i bytes spelled
// on the current path and term[0, j). Only cells with |i - j| <= K can be
// within budget, so each row is stored as a band of 2K+1 cells plus a margin
// cell on each side; cell (i, j) lives at rows[i * S + (j - i + K + 1)].
// Values are clamped at K + 1 ("too far"), which keeps the recurrence exact
// for every value <= K and lets cells fit in a byte. Margins and cells beyond
// the term are never written and stay K + 1, so the recurrence needs no edge
// cases except column 0. The band of a row depends only on its depth, so
// siblings reuse the same storage.
//
// Pruning: min(row i+1) >= min(row i) — every cell derives from a cell of the
// previous row (or two rows up for a transposition, which is >= a cell of the
// previous row) with a non-negative cost — so once min(row) > K no extension
// of the path can match.
//
// prefix_mode: a key's prefix distance is the minimum of D(i, m) over the
// prefixes of the key. Each frame carries `best` = min D(i, m) along its path.
// When min(row) >= best, deeper prefixes cannot improve on it, so the whole
// subtree — an ordinal range — matches at distance `best` and is emitted
// without being walked.
void Fst::levenshtein_walk(std::string_view term, uint32_t max_edits, bool transpositions,
                           bool prefix_mode, std::vector<RangeHit>& hits) const {
  if (num_keys_ == 0) return;
  const uint8_t* q = reinterpret_cast<const uint8_t*>(term.data());
  const uint32_t m = static_cast<uint32_t>(term.size());
  const uint32_t K = max_edits;
  const uint8_t kFar = static_cast<uint8_t>(K + 1);
  const uint32_t S = 2 * K + 3;
  const uint32_t max_depth = m + K;  // rows deeper than this have an empty band

  std::vector<uint8_t> rows(static_cast<size_t>(max_depth + 1) * S, kFar);
  std::vector<uint8_t> path(max_depth + 1, 0);  // path[i] = label leading to depth i
  auto row = [&](uint32_t i) { return rows.data() + static_cast<size_t>(i) * S; };
  auto last_cell = [&](uint32_t i) -> uint8_t {  // D(i, m)
    if (m + K + 1 < i || m + K + 1 - i >= S) return kFar;
    return row(i)[m + K + 1 - i];
  };

  struct Frame {
    uint32_t state, next, end, ord;
    uint8_t best;  // prefix_mode only
  };
  std::vector<Frame> stack;
  stack.reserve(max_depth + 1);

  // Decides, from the row alone, whether a node is dead: nothing in its
  // subtree matches. Most computed rows end here, so this runs before the
  // child's target/output/state are loaded. Also yields the node's `best`.
  auto dead = [&](uint32_t depth, uint8_t row_min, uint8_t parent_best, uint8_t& best) {
    if (!prefix_mode) return row_min > K;
    best = std::min(parent_best, last_cell(depth));
    return row_min >= best && best > K;
  };
  // Handles a live node: emits matches and pushes it if its subtree still
  // needs to be walked.
  auto enter = [&](uint32_t st, uint32_t ord, uint32_t depth, uint8_t row_min, uint8_t best) {
    const State& s = states_[st];
    if (prefix_mode) {
      if (row_min >= best) {  // no deeper prefix can do better: emit the subtree
        hits.push_back({ord, ord + s.count, best});
        return;
      }
      if (s.is_final && best <= K) hits.push_back({ord, ord + 1, best});
    } else {
      const uint8_t d = last_cell(depth);
      if (s.is_final && d <= K) hits.push_back({ord, ord + 1, d});
    }
    if (s.num_transitions > 0 && depth < max_depth)
      stack.push_back({st, s.first_transition, s.first_transition + s.num_transitions, ord, best});
  };

  // Row 0: D(0, j) = j.
  for (uint32_t j = 0; j <= std::min(m, K); ++j) row(0)[j + K + 1] = static_cast<uint8_t>(j);
  uint8_t root_best = kFar;
  if (!dead(0, 0, kFar, root_best)) enter(root_, 0, 0, 0, root_best);

  while (!stack.empty()) {
    Frame& f = stack.back();
    if (f.next == f.end) {
      stack.pop_back();
      continue;
    }
    const uint32_t t = f.next++;
    const uint32_t i = static_cast<uint32_t>(stack.size());  // depth of child
    const uint8_t c = labels_[t];
    path[i] = c;

    uint8_t* cur = row(i);
    const uint8_t* prev = row(i - 1);
    const bool transpose = transpositions && i >= 2;
    const uint8_t* pp = transpose ? row(i - 2) : nullptr;
    const uint8_t prev_label = path[i - 1];
    const uint32_t lo = i > K ? i - K : 0;
    const uint32_t hi = std::min(m, i + K);
    uint8_t row_min = kFar;
    for (uint32_t j = lo; j <= hi; ++j) {
      const uint32_t b = j + K + 1 - i;  // band offset of (i, j); (i-1, j) is b+1
      uint32_t v;
      if (j == 0) {
        v = i;
      } else {
        v = std::min({prev[b + 1] + 1u,                       // delete path byte
                      cur[b - 1] + 1u,                        // insert term byte
                      prev[b] + (q[j - 1] != c ? 1u : 0u)});  // match / substitute
        if (transpose && j >= 2 && c == q[j - 2] && prev_label == q[j - 1])
          v = std::min(v, pp[b] + 1u);  // swap of two adjacent bytes
      }
      cur[b] = static_cast<uint8_t>(std::min<uint32_t>(v, kFar));
      row_min = std::min(row_min, cur[b]);
    }
    uint8_t best = kFar;
    if (dead(i, row_min, f.best, best)) continue;
    enter(targets_[t], f.ord + out_[t], i, row_min, best);  // may invalidate f
  }
}

std::vector<FuzzyMatch> Fst::fuzzy(std::string_view term, uint32_t max_edits, uint32_t limit,
                                   bool transpositions) const {
  DC_CHECK(max_edits <= kMaxEdits, "Fst::fuzzy: max_edits too large");
  if (limit == 0) return {};
  std::vector<RangeHit> hits;
  levenshtein_walk(term, max_edits, transpositions, /*prefix_mode=*/false, hits);
  return to_matches(rank_ranges(hits, limit));
}

std::vector<FuzzyMatch> Fst::fuzzy_prefix(std::string_view prefix, uint32_t max_edits,
                                          uint32_t limit, bool transpositions) const {
  DC_CHECK(max_edits <= kMaxEdits, "Fst::fuzzy_prefix: max_edits too large");
  if (limit == 0) return {};
  std::vector<RangeHit> hits;
  levenshtein_walk(prefix, max_edits, transpositions, /*prefix_mode=*/true, hits);
  return to_matches(rank_ranges(hits, limit));
}

// ---------------------------------------------------------------------------
// Serialization.
//
//   u32 magic, u32 version, u32 num_keys, u32 num_states, u32 num_transitions,
//   u32 root,
//   num_states x {u16 num_transitions, u8 is_final}   (transitions are stored
//                                                      contiguously in state order)
//   u8  labels[num_transitions]
//   u32 targets[num_transitions]
//   u32 weights[num_keys]
// ---------------------------------------------------------------------------

void Fst::serialize(ByteWriter& out) const {
  out.put_u32(kMagic);
  out.put_u32(kVersion);
  out.put_u32(num_keys_);
  out.put_u32(static_cast<uint32_t>(states_.size()));
  out.put_u32(static_cast<uint32_t>(labels_.size()));
  out.put_u32(root_);
  for (const State& s : states_) {
    out.put_u16(s.num_transitions);
    out.put_u8(s.is_final);
  }
  out.put_bytes(labels_.data(), labels_.size());
  for (uint32_t t : targets_) out.put_u32(t);
  for (uint32_t w : weights_) out.put_u32(w);
}

Fst Fst::load(ByteReader& in) {
  DC_CHECK(in.get_u32() == kMagic, "Fst::load: bad magic");
  DC_CHECK(in.get_u32() == kVersion, "Fst::load: unsupported version");
  Fst f;
  f.num_keys_ = in.get_u32();
  const uint32_t num_states = in.get_u32();
  const uint32_t num_trans = in.get_u32();
  f.root_ = in.get_u32();
  if (num_states == 0) {
    DC_CHECK(f.num_keys_ == 0 && num_trans == 0 && f.root_ == 0, "Fst::load: corrupt header");
    return f;
  }
  DC_CHECK(f.root_ < num_states, "Fst::load: root out of range");
  DC_CHECK(f.num_keys_ > 0 || num_trans == 0, "Fst::load: transitions without keys");
  const uint64_t need = uint64_t{num_states} * 3 + uint64_t{num_trans} * 5 + uint64_t{f.num_keys_} * 4;
  DC_CHECK(need <= in.remaining(), "Fst::load: truncated");

  f.states_.resize(num_states);
  uint64_t first = 0;
  for (State& s : f.states_) {
    s.num_transitions = in.get_u16();
    s.is_final = in.get_u8();
    DC_CHECK(s.num_transitions <= 256 && s.is_final <= 1, "Fst::load: corrupt state");
    s.first_transition = static_cast<uint32_t>(first);
    first += s.num_transitions;
    DC_CHECK(first <= num_trans, "Fst::load: transition count mismatch");
  }
  DC_CHECK(first == num_trans, "Fst::load: transition count mismatch");
  f.labels_.resize(num_trans);
  if (num_trans) std::memcpy(f.labels_.data(), in.get_bytes(num_trans), num_trans);
  f.targets_.resize(num_trans);
  for (uint32_t& t : f.targets_) t = in.get_u32();
  f.weights_.resize(f.num_keys_);
  for (uint32_t& w : f.weights_) w = in.get_u32();

  // Recompute counts in id order. Targets must point to smaller ids (children
  // are frozen first), which also proves the graph is acyclic.
  for (uint32_t id = 0; id < num_states; ++id) {
    State& s = f.states_[id];
    uint64_t count = s.is_final;
    for (uint32_t t = s.first_transition; t < s.first_transition + s.num_transitions; ++t) {
      DC_CHECK(f.targets_[t] < id, "Fst::load: transition does not point to an earlier state");
      DC_CHECK(t == s.first_transition || f.labels_[t] > f.labels_[t - 1],
               "Fst::load: labels not strictly increasing");
      count += f.states_[f.targets_[t]].count;
      DC_CHECK(count <= f.num_keys_, "Fst::load: state count exceeds key count");
    }
    DC_CHECK(count > 0 || f.num_keys_ == 0, "Fst::load: dead state");
    s.count = static_cast<uint32_t>(count);
  }
  DC_CHECK(f.states_[f.root_].count == f.num_keys_, "Fst::load: root count mismatch");
  f.build_derived();
  return f;
}

}  // namespace dc::fst
