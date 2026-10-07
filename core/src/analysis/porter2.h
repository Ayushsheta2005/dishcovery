// Snowball English ("Porter2") stemmer.
//
// A faithful port of the reference english.sbl (Snowball 3.x), verified
// against the Snowball project's Python implementation in
// core/tests/test_porter2.cc. Operates on single lowercase ASCII tokens.
#pragma once

#include <string>
#include <string_view>

namespace dc::analysis {

// Stem one lowercase token with the Snowball English (Porter2) algorithm.
// Tokens containing any non-ASCII byte, or of length <= 2, are returned unchanged.
// Digits are allowed (treated as non-vowels, i.e. like consonants) — e.g. "7up" stays sensible.
std::string porter2_stem(std::string_view word);

// In-place variant that avoids allocation when possible (used in the hot analyzer path).
// Edits are made in the existing buffer and never grow it by more than one byte, so
// tokens that fit the small-string buffer never touch the heap.
void porter2_stem_inplace(std::string& word);

}  // namespace dc::analysis
