// Term dictionary as a minimal acyclic finite-state transducer (FST).
//
// Used for autocomplete (top-k completions of a prefix by popularity) and
// typo-tolerant lookup (Levenshtein / optimal-string-alignment distance).
//
// Design
// ------
// * Automaton. A minimal acyclic deterministic automaton (a DAWG) over the
//   UTF-8 bytes of the key set. It is built incrementally from sorted input
//   with a register of canonical states: Daciuk, Mihov, Watson & Watson,
//   "Incremental Construction of Minimal Acyclic Finite-State Automata",
//   Computational Linguistics 26(1), 2000 — the "sorted data" algorithm.
//   Because keys arrive sorted, once the next key diverges from the previous
//   one at depth p, every state on the previous key's path deeper than p can
//   never change again; those states are frozen bottom-up and replaced by an
//   equivalent already-registered state if one exists. The result is minimal
//   after every add(), and memory during construction is O(minimal automaton +
//   longest key).
//
// * Ordinals (perfect hashing). Every state stores `count` = number of keys
//   accepted from it. A key's ordinal is its lexicographic rank: walking the
//   key, add the counts of all transitions skipped (smaller labels) at each
//   state, plus one whenever a final state is passed (the shorter key sorts
//   first). key_at(ord) runs the same walk in reverse. Outputs are a function
//   of each state's right language, not attached per key, so minimisation
//   stays valid — this is what makes the DAWG an FST mapping key -> rank.
//   The per-state partial sums are cached in a derived per-transition array
//   (`out_`) so a step costs one search instead of O(fanout) random reads.
//
// * Weights. Popularity lives in a separate array indexed by ordinal. All
//   keys sharing a prefix occupy a contiguous ordinal range, so top-k
//   completion is top-k over weights[lo, hi): a range-maximum structure plus a
//   priority queue of sub-ranges yields the k best in O(k log k) after the
//   O(|prefix|) walk. The RMQ is a sparse table over blocks of 32 weights
//   (in-block scans at the two ends), which costs O(n/32 · log n) words instead
//   of the O(n log n) of a plain sparse table — a plain table would be larger
//   than the automaton itself.
//
// * Fuzzy lookup. A depth-first walk of the DAWG simulates the Levenshtein
//   automaton of the query: each visited state carries the dynamic-programming
//   row of edit distances between the path spelled so far and every prefix of
//   the query (plus the row two levels up, for adjacent transpositions =
//   optimal-string-alignment distance). A branch is pruned once min(row) >
//   max_edits; rows are banded (only cells with |i - j| <= max_edits can be
//   within budget) so each visited state costs O(max_edits), not O(|query|).
//   The DAWG shares suffixes, so one state can be reached through different
//   prefixes with different rows — the walk never memoises on state alone.
//   Distances are measured in bytes: a substituted multi-byte UTF-8 character
//   costs one edit per differing byte.
//
// Frozen representation: states are {first_transition, count,
// num_transitions, is_final} (12 bytes); transitions are parallel arrays
// label (u8) / target (u32) / out (u32, derived), sorted by label within a
// state (linear scan for small fanout, binary search otherwise). Children are
// always frozen before their parents, so every transition points to a smaller
// state id; load() relies on this to reject cyclic or corrupt input.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "util/bytes.h"
#include "util/common.h"

namespace dc::fst {

struct Completion {
  std::string key;
  uint32_t weight;
  uint32_t ord;
};

struct FuzzyMatch {
  std::string key;
  uint32_t distance;
  uint32_t weight;
  uint32_t ord;
};

// Largest edit budget accepted by fuzzy()/fuzzy_prefix().
inline constexpr uint32_t kMaxEdits = 3;

class FstBuilder;

class Fst {
 public:
  Fst() = default;

  uint32_t size() const { return num_keys_; }  // number of keys
  uint32_t num_states() const { return static_cast<uint32_t>(states_.size()); }
  uint32_t num_transitions() const { return static_cast<uint32_t>(labels_.size()); }
  size_t memory_bytes() const;

  // Ordinal (lexicographic rank) of `key`, or nullopt if absent.
  std::optional<uint32_t> lookup(std::string_view key) const;
  // Inverse of lookup(). Requires ord < size().
  std::string key_at(uint32_t ord) const;
  uint32_t weight_at(uint32_t ord) const;

  // Ordinals [lo, hi) of all keys starting with `prefix`. When no key has the
  // prefix the range is empty with lo == hi == the prefix's insertion point
  // (number of keys lexicographically smaller than it).
  std::pair<uint32_t, uint32_t> prefix_range(std::string_view prefix) const;

  // The k heaviest keys starting with `prefix`: weight desc, ties by ord asc.
  std::vector<Completion> top_k_prefix(std::string_view prefix, uint32_t k) const;

  // All keys within max_edits (<= kMaxEdits) of `term`, best first: distance
  // asc, then weight desc, then ord asc; at most `limit`. With
  // `transpositions` the metric is optimal-string-alignment distance (an
  // adjacent swap costs 1), otherwise plain Levenshtein.
  std::vector<FuzzyMatch> fuzzy(std::string_view term, uint32_t max_edits, uint32_t limit,
                                bool transpositions = true) const;

  // Typo-tolerant autocomplete: keys having some prefix within max_edits of
  // `prefix` (prefix edit distance = min over prefixes p of key of
  // dist(p, prefix)). Ranked distance asc, weight desc, ord asc; at most
  // `limit`. Whole subtrees whose distance can no longer improve are emitted
  // as ordinal ranges and ranked with the same RMQ machinery as top_k_prefix.
  std::vector<FuzzyMatch> fuzzy_prefix(std::string_view prefix, uint32_t max_edits,
                                       uint32_t limit, bool transpositions = true) const;

  // Only the automaton and weights are written; counts, per-transition
  // outputs and the RMQ table are derived and rebuilt (and validated) on load.
  void serialize(ByteWriter& out) const;
  static Fst load(ByteReader& in);

 private:
  friend class FstBuilder;

  struct State {
    uint32_t first_transition = 0;  // index into labels_/targets_/out_
    uint32_t count = 0;             // keys accepted from this state
    uint16_t num_transitions = 0;   // <= 256
    uint8_t is_final = 0;
  };

  // A run of ordinals [lo, hi) that all match at the same distance.
  struct RangeHit {
    uint32_t lo, hi, distance;
  };
  struct Ranked {
    uint32_t ord, distance;
  };

  // Index (relative to s.first_transition) of the first label >= c.
  uint32_t lower_bound_label(const State& s, uint8_t c) const;
  // Weight order used everywhere: heavier first, then smaller ordinal.
  bool better(uint32_t a, uint32_t b) const {
    return weights_[a] > weights_[b] || (weights_[a] == weights_[b] && a < b);
  }
  uint32_t scan_argmax(uint32_t lo, uint32_t hi) const;
  uint32_t range_argmax(uint32_t lo, uint32_t hi) const;  // requires lo < hi
  void build_derived();
  void build_rmq();
  void levenshtein_walk(std::string_view term, uint32_t max_edits, bool transpositions,
                        bool prefix_mode, std::vector<RangeHit>& hits) const;
  // Best `limit` ordinals over the union of hits: distance asc, then better().
  std::vector<Ranked> rank_ranges(std::vector<RangeHit>& hits, uint32_t limit) const;
  std::vector<FuzzyMatch> to_matches(const std::vector<Ranked>& ranked) const;

  std::vector<State> states_;
  std::vector<uint8_t> labels_;
  std::vector<uint32_t> targets_;
  std::vector<uint32_t> out_;  // derived: keys ranked before this transition's subtree
  std::vector<uint32_t> weights_;
  std::vector<uint32_t> rmq_;         // derived: block sparse table of argmax ordinals
  std::vector<uint32_t> rmq_level_;   // derived: offset of each level inside rmq_
  uint32_t root_ = 0;
  uint32_t num_keys_ = 0;
};

class FstBuilder {
 public:
  FstBuilder();

  // Keys must arrive in strictly increasing byte-wise order (DC_CHECK
  // otherwise). Empty key not allowed.
  void add(std::string_view key, uint32_t weight);
  // Freezes the remaining path and returns the automaton. The builder is reset
  // and can be reused.
  Fst finish();
  size_t size() const { return fst_.weights_.size(); }

 private:
  // A state on the path of the most recently added key, not yet frozen. Its
  // last transition points at the next path node (kPending) until frozen.
  struct PendingNode {
    std::vector<uint8_t> labels;
    std::vector<uint32_t> targets;
    bool is_final = false;
    void clear() {
      labels.clear();
      targets.clear();
      is_final = false;
    }
  };

  void freeze_tail(size_t depth);  // freeze path nodes deeper than `depth`
  uint32_t freeze(const PendingNode& node);
  uint32_t append_state(const PendingNode& node);
  uint64_t state_hash(uint32_t id) const;
  void grow_register();
  void reset();

  Fst fst_;                        // frozen states, transitions and weights
  std::vector<PendingNode> path_;  // path_[d] = node reached by prev_[0, d)
  size_t path_len_ = 1;
  std::string prev_;
  // Register of canonical states: open-addressing hash set of state ids keyed
  // by (is_final, labels, targets) — two states are equivalent iff these are
  // equal, because their children are already canonical.
  std::vector<uint32_t> register_;
  size_t registered_ = 0;
};

}  // namespace dc::fst
