// Tests for the term FST: exact queries vs a std::map reference, fuzzy queries
// vs brute-force edit distance, minimality vs a Myhill–Nerode reference, and
// serialization.
#include <algorithm>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "fst/fst.h"
#include "testing.h"

using dc::ByteReader;
using dc::ByteWriter;
using dc::fst::Completion;
using dc::fst::Fst;
using dc::fst::FstBuilder;
using dc::fst::FuzzyMatch;

namespace {

using KeyMap = std::map<std::string, uint32_t>;

Fst build(const KeyMap& m) {
  FstBuilder b;
  for (const auto& [k, w] : m) b.add(k, w);
  return b.finish();
}

std::vector<std::string> keys_of(const KeyMap& m) {
  std::vector<std::string> v;
  for (const auto& kv : m) v.push_back(kv.first);
  return v;
}

// Strings of 1..max_len bytes assembled from units that create shared
// prefixes, shared suffixes and multi-byte UTF-8 characters.
std::string random_string(std::mt19937& rng, size_t max_len) {
  static const std::vector<std::string> kUnits = {
      "a", "b", "c", "a", "b", "c", "ab", "ing", "\xC3\xA9" /* é */, "\xC3\xB1" /* ñ */,
      "\xE8\xBE\xA3" /* 辣 */, "\xF0\x9F\x8D\x95" /* 🍕 */, " "};
  std::uniform_int_distribution<size_t> len_d(1, max_len);
  std::uniform_int_distribution<size_t> unit_d(0, kUnits.size() - 1);
  const size_t target = len_d(rng);
  std::string s;
  for (int tries = 0; s.size() < target && tries < 64; ++tries) {
    const std::string& u = kUnits[unit_d(rng)];
    if (s.size() + u.size() <= target) s += u;
  }
  if (s.empty()) s = "a";
  return s;
}

KeyMap random_keymap(std::mt19937& rng, size_t n, size_t max_len = 20, uint32_t max_weight = 50) {
  KeyMap m;
  std::uniform_int_distribution<uint32_t> w_d(0, max_weight);
  std::uniform_int_distribution<int> coin(0, 9);
  for (size_t i = 0; i < n; ++i) {
    std::string s = random_string(rng, max_len);
    m[s] = w_d(rng);
    // Also insert some proper prefixes and suffix variants of existing keys.
    if (coin(rng) == 0 && s.size() > 1) m[s.substr(0, s.size() / 2)] = w_d(rng);
    if (coin(rng) == 0 && s.size() + 3 <= max_len) m[s + "ing"] = w_d(rng);
  }
  return m;
}

// Apply `edits` random byte edits (insert / delete / substitute / swap).
std::string mutate(std::mt19937& rng, std::string s, int edits) {
  static const std::string kAlpha = "abcgin \xC3";
  for (int e = 0; e < edits; ++e) {
    std::uniform_int_distribution<int> op_d(0, 3);
    std::uniform_int_distribution<size_t> ch_d(0, kAlpha.size() - 1);
    int op = op_d(rng);
    if (s.empty()) op = 0;
    std::uniform_int_distribution<size_t> pos_d(0, s.empty() ? 0 : s.size() - 1);
    size_t pos = pos_d(rng);
    switch (op) {
      case 0: s.insert(s.begin() + static_cast<long>(std::uniform_int_distribution<size_t>(0, s.size())(rng)), kAlpha[ch_d(rng)]); break;
      case 1: s.erase(pos, 1); break;
      case 2: s[pos] = kAlpha[ch_d(rng)]; break;
      case 3:
        if (pos + 1 < s.size()) std::swap(s[pos], s[pos + 1]);
        break;
    }
  }
  return s;
}

// Reference edit distance: Levenshtein, or optimal string alignment (adjacent
// transpositions, no edits inside a transposed pair).
std::vector<std::vector<uint32_t>> edit_table(std::string_view a, std::string_view b,
                                              bool transpositions) {
  const size_t n = a.size(), m = b.size();
  std::vector<std::vector<uint32_t>> d(n + 1, std::vector<uint32_t>(m + 1));
  for (size_t i = 0; i <= n; ++i) d[i][0] = static_cast<uint32_t>(i);
  for (size_t j = 0; j <= m; ++j) d[0][j] = static_cast<uint32_t>(j);
  for (size_t i = 1; i <= n; ++i) {
    for (size_t j = 1; j <= m; ++j) {
      d[i][j] = std::min({d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (a[i - 1] != b[j - 1] ? 1u : 0u)});
      if (transpositions && i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1])
        d[i][j] = std::min(d[i][j], d[i - 2][j - 2] + 1);
    }
  }
  return d;
}

uint32_t edit_distance(std::string_view a, std::string_view b, bool transpositions) {
  return edit_table(a, b, transpositions)[a.size()][b.size()];
}

// min over prefixes p of key of dist(p, q).
uint32_t prefix_distance(std::string_view key, std::string_view q, bool transpositions) {
  auto d = edit_table(key, q, transpositions);
  uint32_t best = UINT32_MAX;
  for (size_t i = 0; i <= key.size(); ++i) best = std::min(best, d[i][q.size()]);
  return best;
}

struct Expected {
  uint32_t distance, weight, ord;
};

std::vector<Expected> brute_fuzzy(const std::vector<std::string>& keys, const KeyMap& m,
                                  std::string_view q, uint32_t k, uint32_t limit,
                                  bool transpositions, bool prefix) {
  std::vector<Expected> v;
  for (uint32_t i = 0; i < keys.size(); ++i) {
    uint32_t d = prefix ? prefix_distance(keys[i], q, transpositions)
                        : edit_distance(keys[i], q, transpositions);
    if (d <= k) v.push_back({d, m.at(keys[i]), i});
  }
  std::sort(v.begin(), v.end(), [](const Expected& a, const Expected& b) {
    return std::tie(a.distance, b.weight, a.ord) < std::tie(b.distance, a.weight, b.ord);
  });
  if (v.size() > limit) v.resize(limit);
  return v;
}

void expect_matches(const std::vector<FuzzyMatch>& got, const std::vector<Expected>& want,
                    const std::vector<std::string>& keys) {
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_EQ(got[i].ord, want[i].ord);
    EXPECT_EQ(got[i].distance, want[i].distance);
    EXPECT_EQ(got[i].weight, want[i].weight);
    EXPECT_EQ(got[i].key, keys[want[i].ord]);
  }
}

// Number of states of the minimal (trimmed) DFA for the key set via
// Myhill–Nerode: one state per distinct right language {w : p·w ∈ L} over
// all prefixes p of keys. Also returns the trie size (= number of prefixes).
std::pair<size_t, size_t> reference_state_counts(const std::vector<std::string>& sorted_keys) {
  std::set<std::string> prefixes;
  for (const auto& k : sorted_keys)
    for (size_t i = 0; i <= k.size(); ++i) prefixes.insert(k.substr(0, i));
  std::set<std::string> languages;
  for (const auto& p : prefixes) {
    std::string sig;  // length-prefixed suffixes: injective for arbitrary bytes
    for (auto it = std::lower_bound(sorted_keys.begin(), sorted_keys.end(), p);
         it != sorted_keys.end() && it->compare(0, p.size(), p) == 0; ++it) {
      sig += std::to_string(it->size() - p.size());
      sig += ':';
      sig.append(*it, p.size());
    }
    languages.insert(sig);
  }
  return {languages.size(), prefixes.size()};
}

void check_exact_queries(const Fst& fst, const KeyMap& m, std::mt19937& rng) {
  const std::vector<std::string> keys = keys_of(m);
  ASSERT_EQ(fst.size(), static_cast<uint32_t>(keys.size()));
  for (uint32_t i = 0; i < keys.size(); ++i) {
    auto ord = fst.lookup(keys[i]);
    ASSERT_TRUE(ord.has_value());
    EXPECT_EQ(*ord, i);
    EXPECT_EQ(fst.key_at(i), keys[i]);
    EXPECT_EQ(fst.weight_at(i), m.at(keys[i]));
  }

  // Probes: random strings, proper prefixes of keys, keys plus a byte.
  std::vector<std::string> probes = {"", "z", "\xFF", std::string(1, '\0')};
  for (int i = 0; i < 200; ++i) probes.push_back(random_string(rng, 22));
  for (const auto& k : keys) {
    probes.push_back(k.substr(0, k.size() / 2));
    probes.push_back(k.substr(0, k.size() - 1));
    probes.push_back(k + "a");
  }
  for (const auto& p : probes) {
    auto got = fst.lookup(p);
    auto it = m.find(p);
    EXPECT_EQ(got.has_value(), it != m.end());
    if (got && it != m.end()) EXPECT_EQ(*got, static_cast<uint32_t>(std::distance(m.begin(), it)));

    // prefix_range vs lower_bound + linear scan.
    const uint32_t lo =
        static_cast<uint32_t>(std::lower_bound(keys.begin(), keys.end(), p) - keys.begin());
    uint32_t hi = lo;
    while (hi < keys.size() && keys[hi].compare(0, p.size(), p) == 0) ++hi;
    auto [rlo, rhi] = fst.prefix_range(p);
    EXPECT_EQ(rlo, lo);
    EXPECT_EQ(rhi, hi);

    // top_k_prefix vs sort of the range.
    for (uint32_t k : {1u, 3u, 10u}) {
      std::vector<uint32_t> ords;
      for (uint32_t o = lo; o < hi; ++o) ords.push_back(o);
      std::sort(ords.begin(), ords.end(), [&](uint32_t a, uint32_t b) {
        return std::make_pair(-static_cast<int64_t>(m.at(keys[a])), a) <
               std::make_pair(-static_cast<int64_t>(m.at(keys[b])), b);
      });
      if (ords.size() > k) ords.resize(k);
      std::vector<Completion> got_k = fst.top_k_prefix(p, k);
      ASSERT_EQ(got_k.size(), ords.size());
      for (size_t i = 0; i < ords.size(); ++i) {
        EXPECT_EQ(got_k[i].ord, ords[i]);
        EXPECT_EQ(got_k[i].key, keys[ords[i]]);
        EXPECT_EQ(got_k[i].weight, m.at(keys[ords[i]]));
      }
    }
  }
}

}  // namespace

TEST(Fst, EmptyFst) {
  for (int pass = 0; pass < 2; ++pass) {
    Fst fst = pass == 0 ? Fst() : FstBuilder().finish();
    EXPECT_EQ(fst.size(), 0u);
    EXPECT_FALSE(fst.lookup("a").has_value());
    EXPECT_FALSE(fst.lookup("").has_value());
    EXPECT_EQ(fst.prefix_range("a").first, 0u);
    EXPECT_EQ(fst.prefix_range("a").second, 0u);
    EXPECT_EQ(fst.prefix_range("").second, 0u);
    EXPECT_TRUE(fst.top_k_prefix("", 5).empty());
    EXPECT_TRUE(fst.fuzzy("a", 2, 10).empty());
    EXPECT_TRUE(fst.fuzzy_prefix("", 1, 10).empty());
    EXPECT_THROW(fst.key_at(0));
    ByteWriter w;
    fst.serialize(w);
    ByteReader r(w.buffer());
    Fst back = Fst::load(r);
    EXPECT_EQ(back.size(), 0u);
    EXPECT_FALSE(back.lookup("a").has_value());
  }
}

TEST(Fst, BuilderRejectsBadInput) {
  FstBuilder b;
  EXPECT_THROW(b.add("", 1));
  b.add("b", 1);
  EXPECT_THROW(b.add("a", 1));  // out of order
  EXPECT_THROW(b.add("b", 1));  // duplicate
  b.add("ba", 2);
  b.add("\xC3\xA9", 3);  // bytes >= 0x80 sort after ASCII
  EXPECT_THROW(b.add("z", 1));
  EXPECT_EQ(b.size(), 3u);
  Fst fst = b.finish();
  EXPECT_EQ(fst.size(), 3u);
  EXPECT_EQ(b.size(), 0u);  // builder reset and reusable
  b.add("a", 1);
  EXPECT_EQ(b.finish().size(), 1u);
  EXPECT_THROW(fst.fuzzy("b", 4, 10));
}

TEST(Fst, TextbookMinimalAutomaton) {
  // {tap, taps, top, tops}: the trie has 8 states; the minimal automaton
  // merges ta/to, tap/top and taps/tops: root -t-> A -{a,o}-> B -p-> C(final)
  // -s-> D(final), i.e. 5 states and 5 transitions.
  KeyMap m{{"tap", 1}, {"taps", 2}, {"top", 3}, {"tops", 4}};
  Fst fst = build(m);
  EXPECT_EQ(fst.num_states(), 5u);
  EXPECT_EQ(fst.num_transitions(), 5u);
  EXPECT_EQ(reference_state_counts(keys_of(m)).first, 5u);
  EXPECT_EQ(reference_state_counts(keys_of(m)).second, 8u);
  // Ordinals still distinguish the merged paths.
  EXPECT_EQ(fst.lookup("tap").value_or(99), 0u);
  EXPECT_EQ(fst.lookup("taps").value_or(99), 1u);
  EXPECT_EQ(fst.lookup("top").value_or(99), 2u);
  EXPECT_EQ(fst.lookup("tops").value_or(99), 3u);
  EXPECT_EQ(fst.key_at(3), "tops");
  EXPECT_FALSE(fst.lookup("to").has_value());
}

TEST(Fst, MinimalOnRandomSets) {
  std::mt19937 rng(7);
  for (int iter = 0; iter < 30; ++iter) {
    KeyMap m = random_keymap(rng, 1 + rng() % 300);
    Fst fst = build(m);
    auto [minimal, trie] = reference_state_counts(keys_of(m));
    EXPECT_EQ(fst.num_states(), static_cast<uint32_t>(minimal));
    EXPECT_LE(fst.num_states(), static_cast<uint32_t>(trie));
  }
}

TEST(Fst, SharedSuffixesCollapse) {
  // 300 stems x 8 inflections: a trie stores every inflection separately per
  // stem; the DAWG stores the inflection sub-automaton once.
  std::mt19937 rng(11);
  const std::vector<std::string> suffixes = {"", "s", "ed", "er", "ers", "ing", "ings", "able"};
  KeyMap m;
  std::uniform_int_distribution<int> len_d(3, 7), ch_d('a', 'z');
  std::set<std::string> stems;
  while (stems.size() < 300) {
    std::string s;
    for (int i = len_d(rng); i > 0; --i) s.push_back(static_cast<char>(ch_d(rng)));
    stems.insert(s + "x");  // 'x' ending keeps stems from interacting with suffixes
  }
  for (const auto& s : stems)
    for (const auto& suf : suffixes) m[s + suf] = static_cast<uint32_t>(rng() % 1000);
  Fst fst = build(m);
  auto [minimal, trie] = reference_state_counts(keys_of(m));
  EXPECT_EQ(fst.num_states(), static_cast<uint32_t>(minimal));
  // Expect well under half of the trie (the stems' own trie dominates).
  EXPECT_LT(fst.num_states() * 2, static_cast<uint32_t>(trie));
  std::mt19937 rng2(3);
  check_exact_queries(fst, m, rng2);
}

TEST(Fst, ExactQueriesVsMap) {
  std::mt19937 rng(42);
  for (int iter = 0; iter < 40; ++iter) {
    const size_t n = iter == 0 ? 1 : 1 + rng() % 400;
    KeyMap m = random_keymap(rng, n);
    Fst fst = build(m);
    check_exact_queries(fst, m, rng);
  }
}

TEST(Fst, HighFanoutAndBinaryBytes) {
  // Every byte value (including NUL and 0x80..0xFF) as a key and as a second
  // byte: exercises the binary-search path and unsigned byte ordering.
  std::mt19937 rng(5);
  KeyMap m;
  for (int b = 0; b < 256; ++b) {
    m[std::string(1, static_cast<char>(b))] = static_cast<uint32_t>(rng() % 100);
    if (b % 3 == 0) m[std::string("q") + static_cast<char>(b)] = static_cast<uint32_t>(rng() % 100);
    if (b % 5 == 0) m[std::string(1, static_cast<char>(b)) + "zz"] = static_cast<uint32_t>(rng() % 100);
  }
  Fst fst = build(m);
  check_exact_queries(fst, m, rng);
  const auto keys = keys_of(m);
  for (std::string q : {std::string("q"), std::string("\xFF"), std::string(1, '\0') + "z"}) {
    for (uint32_t k = 0; k <= 2; ++k) {
      expect_matches(fst.fuzzy(q, k, 1000), brute_fuzzy(keys, m, q, k, 1000, true, false), keys);
      expect_matches(fst.fuzzy_prefix(q, k, 1000), brute_fuzzy(keys, m, q, k, 1000, true, true),
                     keys);
    }
  }
}

TEST(Fst, TopKTieBreaksByOrdinal) {
  KeyMap m{{"pa", 5}, {"pb", 9}, {"pc", 5}, {"pd", 9}, {"pe", 1}, {"q", 100}};
  Fst fst = build(m);
  auto top = fst.top_k_prefix("p", 4);
  ASSERT_EQ(top.size(), 4u);
  EXPECT_EQ(top[0].key, "pb");
  EXPECT_EQ(top[1].key, "pd");
  EXPECT_EQ(top[2].key, "pa");
  EXPECT_EQ(top[3].key, "pc");
  EXPECT_EQ(fst.top_k_prefix("p", 100).size(), 5u);
  EXPECT_TRUE(fst.top_k_prefix("p", 0).empty());
  EXPECT_TRUE(fst.top_k_prefix("x", 3).empty());
  const std::string heaviest = fst.top_k_prefix("", 1).at(0).key;
  EXPECT_EQ(heaviest, "q");
}

TEST(Fst, TopKLargeRanges) {
  // Ranges spanning many RMQ blocks (block size 32) with heavy ties.
  std::mt19937 rng(99);
  KeyMap m;
  for (int i = 0; i < 5000; ++i) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "k%05d", i);
    m[buf] = static_cast<uint32_t>(rng() % 20);
  }
  Fst fst = build(m);
  const auto keys = keys_of(m);
  for (const char* p : {"", "k", "k0", "k01", "k012", "k4999", "k49"}) {
    auto [lo, hi] = fst.prefix_range(p);
    std::vector<uint32_t> ords;
    for (uint32_t o = lo; o < hi; ++o) ords.push_back(o);
    std::stable_sort(ords.begin(), ords.end(),
                     [&](uint32_t a, uint32_t b) { return m[keys[a]] > m[keys[b]]; });
    for (uint32_t k : {1u, 10u, 50u, 333u}) {
      auto got = fst.top_k_prefix(p, k);
      const size_t want = std::min<size_t>(k, ords.size());
      ASSERT_EQ(got.size(), want);
      for (size_t i = 0; i < got.size(); ++i) EXPECT_EQ(got[i].ord, ords[i]);
    }
  }
}

TEST(Fst, FuzzyVsBruteForce) {
  std::mt19937 rng(1234);
  for (int iter = 0; iter < 25; ++iter) {
    KeyMap m = random_keymap(rng, 1 + rng() % 250, 12, 9);
    Fst fst = build(m);
    const auto keys = keys_of(m);
    std::vector<std::string> queries = {"", "a", "ab", "abc"};
    for (int i = 0; i < 12; ++i) {
      const std::string& base = keys[rng() % keys.size()];
      queries.push_back(mutate(rng, base, static_cast<int>(rng() % 4)));
      queries.push_back(random_string(rng, 10));
    }
    for (const auto& q : queries) {
      for (uint32_t k = 0; k <= 2; ++k) {
        for (bool tr : {true, false}) {
          for (uint32_t limit : {1000u, 3u}) {
            expect_matches(fst.fuzzy(q, k, limit, tr), brute_fuzzy(keys, m, q, k, limit, tr, false),
                           keys);
          }
        }
      }
    }
    // One max-budget query per set.
    const std::string q = mutate(rng, keys[rng() % keys.size()], 3);
    expect_matches(fst.fuzzy(q, 3, 1000), brute_fuzzy(keys, m, q, 3, 1000, true, false), keys);
  }
}

TEST(Fst, FuzzyPrefixVsBruteForce) {
  std::mt19937 rng(777);
  for (int iter = 0; iter < 25; ++iter) {
    KeyMap m = random_keymap(rng, 1 + rng() % 250, 12, 9);
    Fst fst = build(m);
    const auto keys = keys_of(m);
    std::vector<std::string> queries = {"", "a", "ba", "ca b"};
    for (int i = 0; i < 12; ++i) {
      const std::string& base = keys[rng() % keys.size()];
      queries.push_back(mutate(rng, base.substr(0, 1 + rng() % base.size()), static_cast<int>(rng() % 3)));
      queries.push_back(random_string(rng, 8));
    }
    for (const auto& q : queries) {
      for (uint32_t k = 0; k <= 2; ++k) {
        for (bool tr : {true, false}) {
          for (uint32_t limit : {1000u, 5u}) {
            expect_matches(fst.fuzzy_prefix(q, k, limit, tr),
                           brute_fuzzy(keys, m, q, k, limit, tr, true), keys);
          }
        }
      }
    }
  }
}

TEST(Fst, SerializeRoundtrip) {
  std::mt19937 rng(2024);
  KeyMap m = random_keymap(rng, 500);
  Fst fst = build(m);
  ByteWriter w;
  fst.serialize(w);
  ByteReader r(w.buffer());
  Fst back = Fst::load(r);
  EXPECT_TRUE(r.eof());
  EXPECT_EQ(back.size(), fst.size());
  EXPECT_EQ(back.num_states(), fst.num_states());
  EXPECT_EQ(back.num_transitions(), fst.num_transitions());
  EXPECT_EQ(back.memory_bytes(), fst.memory_bytes());
  check_exact_queries(back, m, rng);
  const auto keys = keys_of(m);
  for (int i = 0; i < 10; ++i) {
    std::string q = mutate(rng, keys[rng() % keys.size()], 1);
    expect_matches(back.fuzzy(q, 1, 20), brute_fuzzy(keys, m, q, 1, 20, true, false), keys);
  }
  // Re-serialising the loaded automaton is byte-identical.
  ByteWriter w2;
  back.serialize(w2);
  EXPECT_TRUE(w.buffer() == w2.buffer());
}

TEST(Fst, LoadRejectsCorruptInput) {
  KeyMap m{{"apple", 3}, {"apricot", 1}, {"banana", 2}, {"band", 5}};
  Fst fst = build(m);
  ByteWriter w;
  fst.serialize(w);
  const std::vector<uint8_t> good = w.buffer();

  auto load_bytes = [](const std::vector<uint8_t>& bytes) {
    ByteReader r(bytes);
    return Fst::load(r);
  };
  {  // truncated at every length
    for (size_t n = 0; n < good.size(); ++n) {
      std::vector<uint8_t> cut(good.begin(), good.begin() + static_cast<long>(n));
      EXPECT_THROW(load_bytes(cut));
    }
  }
  {  // bad magic
    auto bad = good;
    bad[0] ^= 0xFF;
    EXPECT_THROW(load_bytes(bad));
  }
  {  // a transition pointing forward (would allow cycles)
    auto bad = good;
    const size_t num_states = fst.num_states(), num_trans = fst.num_transitions();
    const size_t targets_at = 24 + num_states * 3 + num_trans;
    dc::ByteWriter patch;
    patch.put_u32(static_cast<uint32_t>(num_states - 1));
    std::copy(patch.buffer().begin(), patch.buffer().end(), bad.begin() + static_cast<long>(targets_at));
    EXPECT_THROW(load_bytes(bad));
  }
  {  // wrong key count
    auto bad = good;
    bad[8] += 1;
    EXPECT_THROW(load_bytes(bad));
  }
  EXPECT_EQ(load_bytes(good).size(), 4u);
}

TEST(Fst, FoodExample) {
  KeyMap m{{"biryani", 950},        {"burger", 800},         {"burrito", 400},
           {"butter chicken", 900}, {"chicken biryani", 870}, {"chicken burger", 500},
           {"chicken tikka masala", 880}};
  Fst fst = build(m);

  // Extra 'i' typed: one insertion away.
  auto hits = fst.fuzzy("biriyani", 2, 10);
  ASSERT_FALSE(hits.empty());
  EXPECT_EQ(hits[0].key, "biryani");
  EXPECT_EQ(hits[0].distance, 1u);

  // Swapped letters: one transposition under OSA, two edits under Levenshtein.
  hits = fst.fuzzy("briyani", 1, 10, /*transpositions=*/true);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].key, "biryani");
  EXPECT_EQ(hits[0].distance, 1u);
  EXPECT_TRUE(fst.fuzzy("briyani", 1, 10, /*transpositions=*/false).empty());
  hits = fst.fuzzy("briyani", 2, 10, /*transpositions=*/false);
  ASSERT_FALSE(hits.empty());
  EXPECT_EQ(hits[0].key, "biryani");
  EXPECT_EQ(hits[0].distance, 2u);

  // Autocomplete: the two heaviest "chicken ..." dishes.
  auto top = fst.top_k_prefix("chicken ", 2);
  ASSERT_EQ(top.size(), 2u);
  EXPECT_EQ(top[0].key, "chicken tikka masala");
  EXPECT_EQ(top[0].weight, 880u);
  EXPECT_EQ(top[1].key, "chicken biryani");
  EXPECT_EQ(top[1].weight, 870u);
  const std::string bu = fst.top_k_prefix("bu", 1).at(0).key;
  EXPECT_EQ(bu, "butter chicken");

  // Typo-tolerant autocomplete.
  auto ac = fst.fuzzy_prefix("chiken", 1, 3);
  ASSERT_EQ(ac.size(), 3u);
  EXPECT_EQ(ac[0].key, "chicken tikka masala");
  EXPECT_EQ(ac[1].key, "chicken biryani");
  EXPECT_EQ(ac[2].key, "chicken burger");
  for (const auto& a : ac) EXPECT_EQ(a.distance, 1u);
  ac = fst.fuzzy_prefix("buter", 1, 10);
  ASSERT_EQ(ac.size(), 1u);
  EXPECT_EQ(ac[0].key, "butter chicken");
  // Exact prefix matches outrank heavier typo matches.
  ac = fst.fuzzy_prefix("bur", 1, 10);
  ASSERT_GE(ac.size(), 3u);
  EXPECT_EQ(ac[0].key, "burger");
  EXPECT_EQ(ac[0].distance, 0u);
  EXPECT_EQ(ac[1].key, "burrito");
  EXPECT_EQ(ac[1].distance, 0u);
  EXPECT_EQ(ac[2].distance, 1u);
}
