// Text analysis pipeline: Unicode normalisation → case folding → tokenisation
// → stopwords → stemming → optional edge n-grams.
//
// Analyzers are configured per field (schema.json) because dish names, review
// text and keyword filters want different treatment:
//   standard   normalise, tokenise, stopwords, Porter2           (descriptions, reviews)
//   minimal    normalise, tokenise, stopwords, plural-only stem  (dish names)
//   plain      normalise, tokenise                               (exact-match field)
//   keyword    whole normalised string as one token              (filters: chain, city)
//   edge_ngram normalise, tokenise, prefixes of length [2, 15]   (prefix matching)
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace dc::analysis {

struct Token {
  std::string term;
  uint32_t position;
};

enum class Stemmer { kNone, kPorter2, kMinimal };

struct AnalyzerConfig {
  std::string name;
  bool keyword = false;       // emit the whole normalised input as one token
  bool stopwords = false;
  Stemmer stemmer = Stemmer::kNone;
  uint32_t edge_ngram_min = 0;  // 0 disables n-grams
  uint32_t edge_ngram_max = 0;
  uint32_t max_token_bytes = 64;
};

// Unicode NFKC + case fold + strip combining marks (utf8proc). Pure-ASCII input
// takes a fast path. Apostrophes are removed ("domino's" -> "dominos").
std::string normalize(std::string_view text);

// Split normalised text into alphanumeric runs (any script). Appends to `out`.
void tokenize(std::string_view normalized, std::vector<std::string>& out, uint32_t max_token_bytes = 64);

bool is_stopword(std::string_view term);

// Lucene's EnglishMinimalStemmer: only plural endings (ies→y, es→e, s→"").
void minimal_stem_inplace(std::string& w);

class Analyzer {
 public:
  explicit Analyzer(AnalyzerConfig cfg) : cfg_(std::move(cfg)) {}

  // Analyse `text`, numbering tokens from `first_position`. Positions advance
  // for removed stopwords too, so phrase matching respects gaps. Returns the
  // position after the last token (callers add a gap between field values).
  uint32_t analyze(std::string_view text, std::vector<Token>& out, uint32_t first_position = 0) const;
  std::vector<Token> analyze(std::string_view text) const {
    std::vector<Token> out;
    analyze(text, out, 0);
    return out;
  }
  // Convenience: just the terms.
  std::vector<std::string> terms(std::string_view text) const;

  const AnalyzerConfig& config() const { return cfg_; }
  const std::string& name() const { return cfg_.name; }

 private:
  AnalyzerConfig cfg_;
};

// Named analyzer registry (built-ins listed above). Throws on unknown names.
const Analyzer& get_analyzer(std::string_view name);

// Position gap inserted between values of a multi-valued field so phrases do
// not match across values ("thai" | "noodles" must not match "thai noodles").
inline constexpr uint32_t kMultiValueGap = 100;

}  // namespace dc::analysis
