// Snowball English (Porter2) stemmer.
//
// Mirrors english.sbl from Snowball 3.x rule for rule; step names follow
// https://snowballstem.org/algorithms/english/stemmer.html. Compared with the
// original 2001 Porter2 description, Snowball 3 adds: more R1 prefixes
// (emerg, inter, later, organ, past, univers), "ogist" and "lessli" in Step 2,
// "past" as a short syllable, the "ying" -> "ie" rule for dying/lying/tying,
// keeping the double in add/egg/off, and handling the old post-Step-1a
// exception words (inning, outing, ..., succeed, plus "evening") in Step 1b.
//
// Every edit is a suffix replacement on the caller's buffer, and region
// boundaries p1/p2 are indices into the prefix that never changes.

#include "analysis/porter2.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace dc::analysis {
namespace {

// ---- Character groupings ----------------------------------------------------

enum : uint8_t {
  kV = 1,        // v:        a e i o u y
  kVWXY = 2,     // v_WXY:    v + w x Y
  kValidLi = 4,  // valid_LI: c d e g h k m n r t
  kDouble = 8,   // double:   b d f g m n p r t
  kAeo = 16,     // a e o     (initial letter of add/egg/off-style words)
};

constexpr std::array<uint8_t, 256> make_classes() {
  std::array<uint8_t, 256> t{};
  for (char c : std::string_view("aeiouy")) t[static_cast<uint8_t>(c)] |= kV | kVWXY;
  for (char c : std::string_view("wxY")) t[static_cast<uint8_t>(c)] |= kVWXY;
  for (char c : std::string_view("cdeghkmnrt")) t[static_cast<uint8_t>(c)] |= kValidLi;
  for (char c : std::string_view("bdfgmnprt")) t[static_cast<uint8_t>(c)] |= kDouble;
  for (char c : std::string_view("aeo")) t[static_cast<uint8_t>(c)] |= kAeo;
  return t;
}
constexpr std::array<uint8_t, 256> kClasses = make_classes();

inline bool has(char c, uint8_t cls) { return kClasses[static_cast<uint8_t>(c)] & cls; }
inline bool is_v(char c) { return has(c, kV); }

// ---- Small string helpers ---------------------------------------------------

// Compares from the back so most mismatches exit on the first character.
inline bool ends_with(std::string_view w, std::string_view suffix) {
  if (suffix.size() > w.size()) return false;
  const char* a = w.data() + w.size();
  const char* b = suffix.data() + suffix.size();
  for (size_t i = 0; i < suffix.size(); ++i) {
    if (*--a != *--b) return false;
  }
  return true;
}

// Replaces w[pos, end) with `repl`.
inline void replace_tail(std::string& w, size_t pos, std::string_view repl) {
  w.resize(pos);
  w.append(repl);
}

// ---- Suffix rule tables (Snowball `among`) ----------------------------------

enum class Cond : uint8_t { kNone, kPrecededByL, kPrecededByValidLi, kInR2, kPrecededBySOrT };

struct Rule {
  std::string_view suffix;
  std::string_view replacement;
  Cond cond = Cond::kNone;
};

// `among` selects the longest matching suffix and never falls back to a
// shorter one if that rule's condition fails. Tables are therefore kept in
// order of non-increasing suffix length so the first hit is the longest.
template <size_t N>
constexpr bool longest_first(const Rule (&rules)[N]) {
  for (size_t i = 1; i < N; ++i) {
    if (rules[i].suffix.size() > rules[i - 1].suffix.size()) return false;
  }
  return true;
}

constexpr Rule kStep2Rules[] = {
    {"ization", "ize"}, {"ational", "ate"}, {"iveness", "ive"}, {"fulness", "ful"},
    {"ousness", "ous"}, {"biliti", "ble"},  {"lessli", "less"}, {"tional", "tion"},
    {"fulli", "ful"},   {"ousli", "ous"},   {"entli", "ent"},   {"aliti", "al"},
    {"iviti", "ive"},   {"alism", "al"},    {"ation", "ate"},   {"ogist", "og"},
    {"anci", "ance"},   {"enci", "ence"},   {"abli", "able"},   {"alli", "al"},
    {"izer", "ize"},    {"ator", "ate"},    {"bli", "ble"},
    {"ogi", "og", Cond::kPrecededByL},
    {"li", "", Cond::kPrecededByValidLi},
};

constexpr Rule kStep3Rules[] = {
    {"ational", "ate"}, {"tional", "tion"}, {"alize", "al"}, {"icate", "ic"},
    {"iciti", "ic"},    {"ative", "", Cond::kInR2},          {"ical", "ic"},
    {"ness", ""},       {"ful", ""},
};

constexpr Rule kStep4Rules[] = {
    {"ement", ""}, {"ance", ""}, {"ence", ""}, {"able", ""}, {"ible", ""}, {"ment", ""},
    {"ate", ""},   {"ive", ""},  {"ize", ""},  {"iti", ""},  {"ism", ""},
    {"ion", "", Cond::kPrecededBySOrT},
    {"ous", ""},   {"ant", ""},  {"ent", ""},  {"ic", ""},   {"al", ""},   {"er", ""},
};

static_assert(longest_first(kStep2Rules) && longest_first(kStep3Rules) &&
              longest_first(kStep4Rules));

template <size_t N>
const Rule* longest_match(std::string_view w, const Rule (&rules)[N]) {
  for (const Rule& r : rules) {
    if (ends_with(w, r.suffix)) return &r;
  }
  return nullptr;
}

// Shared body of Steps 2-4: replace the longest matching suffix if it starts
// at or after `region` (p1 for Steps 2-3, p2 for Step 4) and its extra
// condition holds.
template <size_t N>
void replace_suffix_in_region(std::string& w, const Rule (&rules)[N], size_t region, size_t p2) {
  const Rule* rule = longest_match(w, rules);
  if (rule == nullptr) return;
  const size_t s = w.size() - rule->suffix.size();
  if (s < region) return;
  const char prev = s > 0 ? w[s - 1] : '\0';
  switch (rule->cond) {
    case Cond::kNone:
      break;
    case Cond::kPrecededByL:
      if (prev != 'l') return;
      break;
    case Cond::kPrecededByValidLi:
      if (!has(prev, kValidLi)) return;
      break;
    case Cond::kInR2:
      if (s < p2) return;
      break;
    case Cond::kPrecededBySOrT:
      if (prev != 's' && prev != 't') return;
      break;
  }
  replace_tail(w, s, rule->replacement);
}

// ---- Exceptional forms ------------------------------------------------------

struct Exception {
  std::string_view word;
  std::string_view stem;
};

// Whole words handled before any other processing ("exception1").
constexpr Exception kExceptions[] = {
    {"skis", "ski"},   {"skies", "sky"}, {"idly", "idl"},   {"gently", "gentl"},
    {"ugly", "ugli"},  {"early", "earli"}, {"only", "onli"}, {"singly", "singl"},
    // Invariant forms.
    {"sky", "sky"},    {"news", "news"}, {"howe", "howe"},  {"atlas", "atlas"},
    {"cosmos", "cosmos"}, {"bias", "bias"}, {"andes", "andes"},
};

bool apply_exception(std::string& w) {
  for (const Exception& e : kExceptions) {
    if (w == e.word) {
      if (e.stem != e.word) w.assign(e.stem);
      return true;
    }
  }
  return false;
}

// ---- Prelude and regions ----------------------------------------------------

// Drops a leading apostrophe, then marks consonantal y as Y: an initial y, or
// a y after a vowel. Returns whether any Y was set (the postlude needs it).
bool prelude(std::string& w) {
  if (!w.empty() && w[0] == '\'') w.erase(0, 1);
  bool y_found = false;
  if (!w.empty() && w[0] == 'y') {
    w[0] = 'Y';
    y_found = true;
  }
  for (size_t i = 1; i < w.size(); ++i) {
    if (w[i] == 'y' && is_v(w[i - 1])) {
      w[i] = 'Y';
      y_found = true;
    }
  }
  return y_found;
}

struct Regions {
  size_t p1;  // R1 = w[p1, end)
  size_t p2;  // R2 = w[p2, end)
};

// Position just past the first non-vowel that follows a vowel, searching from
// `from`; w.size() if there is none.
size_t after_vowel_then_consonant(std::string_view w, size_t from) {
  size_t i = from;
  while (i < w.size() && !is_v(w[i])) ++i;
  while (i < w.size() && is_v(w[i])) ++i;
  return i < w.size() ? i + 1 : w.size();
}

// Words starting with these get R1 right after the prefix, so e.g.
// "generous" and "general" are not conflated via "gen".
constexpr std::string_view kR1Prefixes[] = {
    "arsen", "commun", "emerg", "gener", "inter", "later", "organ", "past", "univers",
};

Regions mark_regions(std::string_view w) {
  size_t p1 = std::string_view::npos;
  for (std::string_view prefix : kR1Prefixes) {
    if (w.starts_with(prefix)) {
      p1 = prefix.size();
      break;
    }
  }
  if (p1 == std::string_view::npos) p1 = after_vowel_then_consonant(w, 0);
  return {p1, after_vowel_then_consonant(w, p1)};
}

// A short syllable ending at `pos`: non-vowel, vowel, non-vowel other than
// w/x/Y; or vowel, non-vowel at the very start of the word; or "past".
bool ends_in_short_syllable(std::string_view w, size_t pos) {
  if (pos >= 3 && !has(w[pos - 1], kVWXY) && is_v(w[pos - 2]) && !is_v(w[pos - 3])) return true;
  if (pos == 2 && !is_v(w[1]) && is_v(w[0])) return true;
  return pos >= 4 && w.substr(pos - 4, 4) == "past";
}

// ---- Steps ------------------------------------------------------------------

// Step 0 (folded into 1a by english.sbl): strip the longest of ' 's 's'.
// Step 1a: sses -> ss; ied/ies -> i (or ie after a single letter);
// drop s when a vowel occurs before the preceding letter; leave ss/us alone.
void step_1a(std::string& w) {
  if (ends_with(w, "'s'")) {
    w.resize(w.size() - 3);
  } else if (ends_with(w, "'s")) {
    w.resize(w.size() - 2);
  } else if (ends_with(w, "'")) {
    w.pop_back();
  }

  const size_t n = w.size();
  if (ends_with(w, "sses")) {
    w.resize(n - 2);
  } else if (ends_with(w, "ied") || ends_with(w, "ies")) {
    w.resize(n - 3 >= 2 ? n - 2 : n - 1);  // ties -> tie, cries -> cri
  } else if (ends_with(w, "ss") || ends_with(w, "us")) {
    // unchanged
  } else if (ends_with(w, "s")) {
    for (size_t i = 0; i + 2 < n; ++i) {
      if (is_v(w[i])) {
        w.pop_back();  // gaps -> gap, but gas -> gas
        break;
      }
    }
  }
}

// Step 1b: eed/eedly -> ee in R1; otherwise delete ed/edly/ing/ingly when the
// preceding part has a vowel, then repair the stem (add e, undouble, ...).
void step_1b(std::string& w, size_t p1) {
  enum class Kind : uint8_t { kEed, kIng, kDelete };
  struct Suffix {
    std::string_view text;
    Kind kind;
  };
  static constexpr Suffix kSuffixes[] = {
      {"eedly", Kind::kEed}, {"ingly", Kind::kDelete}, {"edly", Kind::kDelete},
      {"eed", Kind::kEed},   {"ing", Kind::kIng},      {"ed", Kind::kDelete},
  };

  const Suffix* match = nullptr;
  for (const Suffix& sfx : kSuffixes) {
    if (ends_with(w, sfx.text)) {
      match = &sfx;
      break;
    }
  }
  if (match == nullptr) return;
  const size_t s = w.size() - match->text.size();
  const std::string_view stem(w.data(), s);

  if (match->kind == Kind::kEed) {
    // proceed/exceed/succeed keep their "eed".
    if (s >= p1 && stem != "proc" && stem != "exc" && stem != "succ") replace_tail(w, s, "ee");
    return;
  }
  if (match->kind == Kind::kIng) {
    // A lone consonant + "ying": dying -> die, lying -> lie, tying -> tie.
    if (s == 2 && stem[1] == 'y' && !is_v(stem[0])) {
      replace_tail(w, 1, "ie");
      return;
    }
    // inning, outing, canning, herring, earring, evening stay as they are.
    if (stem == "inn" || stem == "out" || stem == "cann" || stem == "herr" || stem == "earr" ||
        stem == "even") {
      return;
    }
  }

  if (std::none_of(stem.begin(), stem.end(), is_v)) return;
  w.resize(s);

  if (ends_with(w, "at") || ends_with(w, "bl") || ends_with(w, "iz")) {
    w.push_back('e');  // luxuriated -> luxuriate
  } else if (s >= 2 && w[s - 1] == w[s - 2] && has(w[s - 1], kDouble)) {
    if (!(s == 3 && has(w[0], kAeo))) w.pop_back();  // hopped -> hop, but added -> add
  } else if (s == p1 && ends_in_short_syllable(w, s)) {
    w.push_back('e');  // hoping -> hope (the word is "short")
  }
}

// Step 1c: final y/Y -> i after a non-vowel that is not the first letter.
void step_1c(std::string& w) {
  const size_t n = w.size();
  if (n >= 3 && (w[n - 1] == 'y' || w[n - 1] == 'Y') && !is_v(w[n - 2])) w[n - 1] = 'i';
}

// Step 5: drop final e in R2, or in R1 when not preceded by a short
// syllable; drop the second l of a final "ll" in R2.
void step_5(std::string& w, const Regions& r) {
  if (w.empty()) return;
  const size_t s = w.size() - 1;
  if (w[s] == 'e') {
    if (s >= r.p2 || (s >= r.p1 && !ends_in_short_syllable(w, s))) w.pop_back();
  } else if (w[s] == 'l') {
    if (s >= r.p2 && s > 0 && w[s - 1] == 'l') w.pop_back();
  }
}

}  // namespace

void porter2_stem_inplace(std::string& word) {
  if (word.size() <= 2) return;
  for (char c : word) {
    if (static_cast<uint8_t>(c) >= 0x80) return;
  }
  if (apply_exception(word)) return;

  const bool y_found = prelude(word);
  const Regions r = mark_regions(word);
  step_1a(word);
  step_1b(word, r.p1);
  step_1c(word);
  replace_suffix_in_region(word, kStep2Rules, r.p1, r.p2);  // Step 2
  replace_suffix_in_region(word, kStep3Rules, r.p1, r.p2);  // Step 3
  replace_suffix_in_region(word, kStep4Rules, r.p2, r.p2);  // Step 4
  step_5(word, r);

  // Postlude: turn the Y markers back into y.
  if (y_found) std::replace(word.begin(), word.end(), 'Y', 'y');
}

std::string porter2_stem(std::string_view word) {
  std::string out(word);
  porter2_stem_inplace(out);
  return out;
}

}  // namespace dc::analysis
