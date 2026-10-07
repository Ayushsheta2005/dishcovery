#include "analysis/analyzer.h"

#include <utf8proc.h>

#include <cstdlib>
#include <map>
#include <mutex>
#include <unordered_set>

#include "analysis/porter2.h"
#include "util/common.h"

namespace dc::analysis {

namespace {

bool is_ascii(std::string_view s) {
  for (unsigned char c : s)
    if (c & 0x80) return false;
  return true;
}

bool is_apostrophe(utf8proc_int32_t cp) {
  return cp == '\'' || cp == 0x2019 || cp == 0x2018 || cp == '`' || cp == 0x00B4 || cp == 0x02BC;
}

bool is_word_cp(utf8proc_int32_t cp) {
  if (cp < 0x80) return (cp >= '0' && cp <= '9') || (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
  switch (utf8proc_category(cp)) {
    case UTF8PROC_CATEGORY_LU:
    case UTF8PROC_CATEGORY_LL:
    case UTF8PROC_CATEGORY_LT:
    case UTF8PROC_CATEGORY_LM:
    case UTF8PROC_CATEGORY_LO:
    case UTF8PROC_CATEGORY_ND:
    case UTF8PROC_CATEGORY_NL:
    case UTF8PROC_CATEGORY_NO:
    case UTF8PROC_CATEGORY_MN:
    case UTF8PROC_CATEGORY_MC:
      return true;
    default:
      return false;
  }
}

// Lucene's default English stop set.
const std::unordered_set<std::string_view>& stop_set() {
  static const std::unordered_set<std::string_view> s = {
      "a",    "an",   "and",  "are",  "as",    "at",    "be",   "but",  "by",   "for",  "if",
      "in",   "into", "is",   "it",   "no",    "not",   "of",   "on",   "or",   "such", "that",
      "the",  "their", "then", "there", "these", "they", "this", "to",   "was",  "will", "with"};
  return s;
}

}  // namespace

std::string normalize(std::string_view text) {
  if (is_ascii(text)) {
    std::string out;
    out.reserve(text.size());
    for (char c : text) {
      if (c == '\'' || c == '`') continue;  // join across apostrophes
      out.push_back((c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c);
    }
    return out;
  }
  utf8proc_uint8_t* mapped = nullptr;
  auto opts = static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPAT | UTF8PROC_COMPOSE |
                                             UTF8PROC_STRIPMARK | UTF8PROC_CASEFOLD | UTF8PROC_IGNORE);
  utf8proc_ssize_t n = utf8proc_map(reinterpret_cast<const utf8proc_uint8_t*>(text.data()),
                                    static_cast<utf8proc_ssize_t>(text.size()), &mapped, opts);
  if (n < 0) {
    // Invalid UTF-8: fall back to bytewise ASCII folding, dropping non-ASCII.
    std::string out;
    for (unsigned char c : text)
      if (c < 0x80 && c != '\'') out.push_back(static_cast<char>(std::tolower(c)));
    return out;
  }
  // Drop apostrophes (and their Unicode look-alikes) in a second pass.
  std::string out;
  out.reserve(static_cast<size_t>(n));
  const utf8proc_uint8_t* p = mapped;
  const utf8proc_uint8_t* end = mapped + n;
  while (p < end) {
    utf8proc_int32_t cp;
    utf8proc_ssize_t len = utf8proc_iterate(p, end - p, &cp);
    if (len <= 0) break;
    if (!is_apostrophe(cp)) out.append(reinterpret_cast<const char*>(p), static_cast<size_t>(len));
    p += len;
  }
  std::free(mapped);
  return out;
}

void tokenize(std::string_view s, std::vector<std::string>& out, uint32_t max_token_bytes) {
  const auto* p = reinterpret_cast<const utf8proc_uint8_t*>(s.data());
  const auto* end = p + s.size();
  std::string cur;
  auto flush = [&] {
    if (!cur.empty()) {
      if (cur.size() <= max_token_bytes) out.push_back(cur);
      cur.clear();
    }
  };
  while (p < end) {
    utf8proc_int32_t cp;
    utf8proc_ssize_t len;
    if (*p < 0x80) {
      cp = *p;
      len = 1;
    } else {
      len = utf8proc_iterate(p, end - p, &cp);
      if (len <= 0) {  // invalid byte: treat as separator
        flush();
        ++p;
        continue;
      }
    }
    if (is_word_cp(cp))
      cur.append(reinterpret_cast<const char*>(p), static_cast<size_t>(len));
    else
      flush();
    p += len;
  }
  flush();
}

bool is_stopword(std::string_view term) { return stop_set().count(term) > 0; }

void minimal_stem_inplace(std::string& w) {
  size_t n = w.size();
  if (n < 3 || w[n - 1] != 's') return;
  switch (w[n - 2]) {
    case 'u':
    case 's':
      return;
    case 'e':
      if (n > 3 && w[n - 3] == 'i' && w[n - 4] != 'a' && w[n - 4] != 'e') {
        w.resize(n - 3);
        w.push_back('y');
        return;
      }
      if (w[n - 3] == 'i' || w[n - 3] == 'a' || w[n - 3] == 'o' || w[n - 3] == 'e') return;
      [[fallthrough]];
    default:
      w.pop_back();
  }
}

uint32_t Analyzer::analyze(std::string_view text, std::vector<Token>& out, uint32_t pos) const {
  std::string norm = normalize(text);
  if (cfg_.keyword) {
    // Collapse internal separators to single spaces so "Domino's  Pizza" == "dominos pizza".
    std::vector<std::string> parts;
    tokenize(norm, parts, 4096);
    std::string joined;
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i) joined.push_back(' ');
      joined += parts[i];
    }
    if (!joined.empty()) out.push_back({std::move(joined), pos++});
    return pos;
  }
  std::vector<std::string> words;
  tokenize(norm, words, cfg_.max_token_bytes);
  for (auto& w : words) {
    uint32_t p = pos++;
    if (cfg_.stopwords && is_stopword(w)) continue;
    switch (cfg_.stemmer) {
      case Stemmer::kPorter2:
        porter2_stem_inplace(w);
        break;
      case Stemmer::kMinimal:
        minimal_stem_inplace(w);
        break;
      case Stemmer::kNone:
        break;
    }
    if (cfg_.edge_ngram_min > 0) {
      // Edge n-grams share the word's position. Only cut at UTF-8 boundaries.
      for (uint32_t len = cfg_.edge_ngram_min; len <= cfg_.edge_ngram_max && len <= w.size(); ++len) {
        if (len < w.size() && (static_cast<unsigned char>(w[len]) & 0xC0) == 0x80) continue;
        out.push_back({w.substr(0, len), p});
      }
      if (w.size() > cfg_.edge_ngram_max) out.push_back({w, p});
    } else {
      out.push_back({std::move(w), p});
    }
  }
  return pos;
}

std::vector<std::string> Analyzer::terms(std::string_view text) const {
  std::vector<Token> toks;
  analyze(text, toks, 0);
  std::vector<std::string> out;
  out.reserve(toks.size());
  for (auto& t : toks) out.push_back(std::move(t.term));
  return out;
}

const Analyzer& get_analyzer(std::string_view name) {
  static const std::map<std::string, Analyzer, std::less<>> registry = [] {
    std::map<std::string, Analyzer, std::less<>> m;
    auto add = [&](AnalyzerConfig c) { m.emplace(c.name, Analyzer(c)); };
    add({"standard", false, true, Stemmer::kPorter2, 0, 0, 64});
    add({"minimal", false, true, Stemmer::kMinimal, 0, 0, 64});
    add({"plain", false, false, Stemmer::kNone, 0, 0, 64});
    add({"keyword", true, false, Stemmer::kNone, 0, 0, 4096});
    add({"edge_ngram", false, false, Stemmer::kNone, 2, 15, 64});
    return m;
  }();
  auto it = registry.find(name);
  if (it == registry.end()) fail("unknown analyzer: " + std::string(name));
  return it->second;
}

}  // namespace dc::analysis
