#include "analysis/analyzer.h"
#include "testing.h"

using namespace dc::analysis;

TEST(Analysis, NormalizeFoldsCaseAccentsApostrophes) {
  EXPECT_EQ(normalize("Domino's Pizza"), "dominos pizza");
  EXPECT_EQ(normalize("Domino\xE2\x80\x99s"), "dominos");          // U+2019
  EXPECT_EQ(normalize("Jalape\xC3\xB1o Caf\xC3\xA9"), "jalapeno cafe");
  EXPECT_EQ(normalize("\xEF\xBC\xA2\xEF\xBC\xA9\xEF\xBC\xB2"), "bir");  // fullwidth BIR (NFKC)
}

TEST(Analysis, TokenizeSplitsOnPunctuation) {
  std::vector<std::string> out;
  tokenize(normalize("gluten-free, chicken_tikka (spicy!) 7up"), out);
  std::vector<std::string> want = {"gluten", "free", "chicken", "tikka", "spicy", "7up"};
  EXPECT_TRUE(out == want);
}

TEST(Analysis, StandardAnalyzerStemsAndDropsStopwords) {
  auto toks = get_analyzer("standard").analyze("The Spicy Noodles with Dumplings");
  ASSERT_EQ(toks.size(), size_t(3));
  EXPECT_EQ(toks[0].term, "spici");
  EXPECT_EQ(toks[0].position, 1u);  // "the" consumed position 0
  EXPECT_EQ(toks[1].term, "noodl");
  EXPECT_EQ(toks[2].term, "dumpl");
  EXPECT_EQ(toks[2].position, 4u);
}

TEST(Analysis, MinimalStemmerOnlyStripsPlurals) {
  auto t = get_analyzer("minimal").terms("momos tacos fries biryanis potatoes glass");
  std::vector<std::string> want = {"momo", "taco", "fry", "biryani", "potatoes", "glass"};
  EXPECT_TRUE(t == want);
}

TEST(Analysis, KeywordAnalyzerKeepsWholeValue) {
  auto t = get_analyzer("keyword").terms("  Domino's   Pizza ");
  ASSERT_EQ(t.size(), size_t(1));
  EXPECT_EQ(t[0], "dominos pizza");
}

TEST(Analysis, EdgeNgrams) {
  auto t = get_analyzer("edge_ngram").terms("pizza");
  std::vector<std::string> want = {"pi", "piz", "pizz", "pizza"};
  EXPECT_TRUE(t == want);
}

TEST(Analysis, PlainAnalyzerKeepsEverything) {
  auto t = get_analyzer("plain").terms("Mac and Cheese");
  std::vector<std::string> want = {"mac", "and", "cheese"};
  EXPECT_TRUE(t == want);
}
