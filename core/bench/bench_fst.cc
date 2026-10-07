// Benchmark for the term FST (dc::fst).
//
//   bench_fst --keys file.tsv      key<TAB>weight per line, any order (sorted here;
//                                  duplicate keys keep the max weight)
//   bench_fst --synthetic N [--seed S]
//                                  N unique dish-like phrases with Zipf weights
//
// Prints one JSON object per line: build stats (time, states vs a trie, memory
// vs raw key bytes), exact lookup latency, top-k completion latency per
// prefix length, and fuzzy / fuzzy-prefix latency per edit budget.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "fst/fst.h"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct Entry {
  std::string key;
  uint32_t weight;
};

std::vector<Entry> load_tsv(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<Entry> v;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const size_t tab = line.find('\t');
    std::string key = line.substr(0, tab);
    uint32_t w = 1;
    if (tab != std::string::npos) w = static_cast<uint32_t>(std::strtoul(line.c_str() + tab + 1, nullptr, 10));
    if (!key.empty()) v.push_back({std::move(key), w});
  }
  return v;
}

// Dish-like phrases: 1–3 words of 1–3 syllables, often ending in a common
// dish head word ("... masala", "... rice"), which gives the prefix and
// suffix sharing of a real menu vocabulary.
std::vector<Entry> synthetic(size_t n, uint64_t seed) {
  static const char* kSyl[] = {"chi", "cken", "bir", "ya",  "ni",  "pa",  "neer", "tik", "ka",
                               "ma",  "sa",   "la",  "bur", "ger", "ri",  "to",   "piz", "za",
                               "dal", "mak",  "ha",  "ro",  "ti",  "na",  "an",   "bu",  "tter",
                               "do",  "i",    "dli", "va",  "da",  "pav", "bha",  "ji",  "ko",
                               "fta", "ra",   "ita", "las", "si",  "pu",  "lao",  "nood", "les",
                               "so",  "up",   "sal", "ad",  "ta",  "co",  "su",   "shi", "men"};
  static const char* kHead[] = {"curry", "masala", "biryani", "rice",  "fry",     "roll",
                                "soup",  "salad",  "burger",  "pizza", "tikka",   "naan",
                                "dosa",  "noodles", "wrap",   "bowl",  "kebab",   "paratha"};
  constexpr size_t kNumSyl = sizeof(kSyl) / sizeof(kSyl[0]);
  constexpr size_t kNumHead = sizeof(kHead) / sizeof(kHead[0]);
  std::mt19937_64 rng(seed);
  auto pick = [&](size_t n_) { return static_cast<size_t>(rng() % n_); };
  auto word = [&] {
    const int syl = 1 + static_cast<int>(pick(10) < 3 ? 0 : (pick(10) < 7 ? 1 : 2));
    std::string w;
    for (int i = 0; i < syl; ++i) w += kSyl[pick(kNumSyl)];
    return w;
  };
  std::unordered_set<std::string> seen;
  std::vector<Entry> v;
  v.reserve(n);
  while (v.size() < n) {
    const size_t r = pick(100);
    const int words = r < 35 ? 1 : (r < 80 ? 2 : 3);
    std::string phrase;
    for (int i = 0; i < words; ++i) {
      if (i) phrase += ' ';
      phrase += (i == words - 1 && words > 1 && pick(10) < 5) ? std::string(kHead[pick(kNumHead)]) : word();
    }
    if (!seen.insert(phrase).second) continue;
    // Zipf-like popularity from a random rank.
    const double rank = 1.0 + static_cast<double>(pick(n));
    v.push_back({std::move(phrase), static_cast<uint32_t>(1e7 / rank) + static_cast<uint32_t>(pick(100))});
  }
  return v;
}

std::string mutate(std::mt19937& rng, std::string s) {
  static const char kAlpha[] = "abcdefghijklmnopqrstuvwxyz";
  if (s.size() < 2) return s + "a";
  const size_t pos = rng() % (s.size() - 1);
  switch (rng() % 4) {
    case 0: s.insert(s.begin() + static_cast<long>(pos), kAlpha[rng() % 26]); break;
    case 1: s.erase(pos, 1); break;
    case 2: s[pos] = kAlpha[rng() % 26]; break;
    default: std::swap(s[pos], s[pos + 1]); break;
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  std::string keys_path;
  size_t synthetic_n = 0;
  uint64_t seed = 42;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--keys") && i + 1 < argc) keys_path = argv[++i];
    else if (!std::strcmp(argv[i], "--synthetic") && i + 1 < argc) synthetic_n = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
    else {
      std::fprintf(stderr, "usage: %s --keys file.tsv | --synthetic N [--seed S]\n", argv[0]);
      return 2;
    }
  }
  if (keys_path.empty() && synthetic_n == 0) {
    std::fprintf(stderr, "usage: %s --keys file.tsv | --synthetic N [--seed S]\n", argv[0]);
    return 2;
  }

  std::vector<Entry> entries = keys_path.empty() ? synthetic(synthetic_n, seed) : load_tsv(keys_path);

  // ---- sort + dedupe -------------------------------------------------------
  auto t0 = Clock::now();
  std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.key < b.key; });
  std::vector<Entry> uniq;
  uniq.reserve(entries.size());
  for (auto& e : entries) {
    if (!uniq.empty() && uniq.back().key == e.key) uniq.back().weight = std::max(uniq.back().weight, e.weight);
    else uniq.push_back(std::move(e));
  }
  entries.swap(uniq);
  const double sort_ms = seconds_since(t0) * 1e3;

  size_t raw_bytes = 0, trie_states = 1;
  for (size_t i = 0; i < entries.size(); ++i) {
    const std::string& k = entries[i].key;
    raw_bytes += k.size();
    size_t lcp = 0;
    if (i) {
      const std::string& p = entries[i - 1].key;
      while (lcp < p.size() && lcp < k.size() && p[lcp] == k[lcp]) ++lcp;
    }
    trie_states += k.size() - lcp;
  }

  // ---- build (best of 3) ---------------------------------------------------
  dc::fst::Fst fst;
  double build_ms = 1e100;
  for (int rep = 0; rep < 3; ++rep) {
    t0 = Clock::now();
    dc::fst::FstBuilder b;
    for (const auto& e : entries) b.add(e.key, e.weight);
    fst = b.finish();
    build_ms = std::min(build_ms, seconds_since(t0) * 1e3);
  }
  dc::ByteWriter w;
  fst.serialize(w);
  std::printf(
      "{\"bench\":\"build\",\"keys\":%u,\"sort_ms\":%.2f,\"build_ms\":%.2f,\"states\":%u,"
      "\"transitions\":%u,\"trie_states\":%zu,\"raw_key_bytes\":%zu,\"memory_bytes\":%zu,"
      "\"serialized_bytes\":%zu,\"memory_over_raw\":%.3f,\"serialized_over_raw\":%.3f}\n",
      fst.size(), sort_ms, build_ms, fst.num_states(), fst.num_transitions(), trie_states, raw_bytes,
      fst.memory_bytes(), w.size(), double(fst.memory_bytes()) / double(raw_bytes),
      double(w.size()) / double(raw_bytes));
  if (fst.size() == 0) return 0;

  std::mt19937 rng(static_cast<uint32_t>(seed));
  uint64_t sink = 0;

  // ---- exact lookup / key_at -----------------------------------------------
  {
    std::vector<uint32_t> order(entries.size());
    for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), rng);
    const size_t ops = std::max<size_t>(order.size(), 1000000);
    t0 = Clock::now();
    for (size_t i = 0; i < ops; ++i) sink += fst.lookup(entries[order[i % order.size()]].key).value_or(0);
    const double lookup_ns = seconds_since(t0) * 1e9 / double(ops);
    t0 = Clock::now();
    for (size_t i = 0; i < ops; ++i) sink += fst.key_at(order[i % order.size()]).size();
    const double key_at_ns = seconds_since(t0) * 1e9 / double(ops);
    std::printf("{\"bench\":\"lookup\",\"ops\":%zu,\"ns_per_op\":%.1f}\n", ops, lookup_ns);
    std::printf("{\"bench\":\"key_at\",\"ops\":%zu,\"ns_per_op\":%.1f}\n", ops, key_at_ns);
  }

  // ---- top-k completion for random 1–4 byte prefixes of real keys -----------
  for (size_t len = 1; len <= 4; ++len) {
    std::vector<std::string> prefixes;
    while (prefixes.size() < 20000) {
      const std::string& k = entries[rng() % entries.size()].key;
      if (k.size() >= len) prefixes.push_back(k.substr(0, len));
    }
    size_t results = 0, range = 0;
    t0 = Clock::now();
    for (const auto& p : prefixes) {
      auto r = fst.top_k_prefix(p, 10);
      results += r.size();
      sink += r.empty() ? 0 : r[0].ord;
    }
    const double us = seconds_since(t0) * 1e6 / double(prefixes.size());
    for (const auto& p : prefixes) {
      auto [lo, hi] = fst.prefix_range(p);
      range += hi - lo;
    }
    std::printf(
        "{\"bench\":\"top_k_prefix\",\"k\":10,\"prefix_len\":%zu,\"ops\":%zu,\"us_per_op\":%.3f,"
        "\"avg_results\":%.2f,\"avg_range\":%.1f}\n",
        len, prefixes.size(), us, double(results) / double(prefixes.size()),
        double(range) / double(prefixes.size()));
  }

  // ---- fuzzy: whole keys with one random typo -------------------------------
  {
    std::vector<std::string> queries;
    for (int i = 0; i < 2000; ++i) queries.push_back(mutate(rng, entries[rng() % entries.size()].key));
    size_t qbytes = 0;
    for (const auto& q : queries) qbytes += q.size();
    for (uint32_t k : {1u, 2u}) {
      size_t results = 0;
      t0 = Clock::now();
      for (const auto& q : queries) {
        auto r = fst.fuzzy(q, k, 10);
        results += r.size();
        sink += r.empty() ? 0 : r[0].ord;
      }
      const double us = seconds_since(t0) * 1e6 / double(queries.size());
      std::printf(
          "{\"bench\":\"fuzzy\",\"max_edits\":%u,\"limit\":10,\"ops\":%zu,\"avg_query_len\":%.1f,"
          "\"us_per_op\":%.2f,\"avg_results\":%.2f}\n",
          k, queries.size(), double(qbytes) / double(queries.size()), us,
          double(results) / double(queries.size()));
    }
  }

  // ---- fuzzy_prefix: 3–6 byte prefixes with one random typo ----------------
  {
    std::vector<std::string> queries;
    while (queries.size() < 2000) {
      const std::string& k = entries[rng() % entries.size()].key;
      const size_t len = 3 + rng() % 4;
      if (k.size() >= len) queries.push_back(mutate(rng, k.substr(0, len)));
    }
    for (uint32_t k : {1u, 2u}) {
      size_t results = 0;
      t0 = Clock::now();
      for (const auto& q : queries) {
        auto r = fst.fuzzy_prefix(q, k, 10);
        results += r.size();
        sink += r.empty() ? 0 : r[0].ord;
      }
      const double us = seconds_since(t0) * 1e6 / double(queries.size());
      std::printf(
          "{\"bench\":\"fuzzy_prefix\",\"max_edits\":%u,\"limit\":10,\"ops\":%zu,\"us_per_op\":%.2f,"
          "\"avg_results\":%.2f}\n",
          k, queries.size(), us, double(results) / double(queries.size()));
    }
  }

  std::fprintf(stderr, "sink=%llu\n", static_cast<unsigned long long>(sink));
  return 0;
}
