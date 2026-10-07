#include "analysis/porter2.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "testing.h"

using dc::analysis::porter2_stem;
using dc::analysis::porter2_stem_inplace;

TEST(Porter2, ExceptionalForms) {
  EXPECT_EQ(porter2_stem("skis"), "ski");
  EXPECT_EQ(porter2_stem("skies"), "sky");
  EXPECT_EQ(porter2_stem("dying"), "die");
  EXPECT_EQ(porter2_stem("lying"), "lie");
  EXPECT_EQ(porter2_stem("tying"), "tie");
  EXPECT_EQ(porter2_stem("idly"), "idl");
  EXPECT_EQ(porter2_stem("gently"), "gentl");
  EXPECT_EQ(porter2_stem("ugly"), "ugli");
  EXPECT_EQ(porter2_stem("early"), "earli");
  EXPECT_EQ(porter2_stem("only"), "onli");
  EXPECT_EQ(porter2_stem("singly"), "singl");
  for (const char* w : {"sky", "news", "howe", "atlas", "cosmos", "bias", "andes"}) {
    EXPECT_EQ(porter2_stem(w), w);
  }
}

TEST(Porter2, Step1bExceptionWords) {
  EXPECT_EQ(porter2_stem("innings"), "inning");
  EXPECT_EQ(porter2_stem("outing"), "outing");
  EXPECT_EQ(porter2_stem("canning"), "canning");
  EXPECT_EQ(porter2_stem("herring"), "herring");
  EXPECT_EQ(porter2_stem("earrings"), "earring");
  EXPECT_EQ(porter2_stem("proceed"), "proceed");
  EXPECT_EQ(porter2_stem("exceed"), "exceed");
  EXPECT_EQ(porter2_stem("succeeded"), "succeed");
}

TEST(Porter2, RegionsAndSuffixSteps) {
  EXPECT_EQ(porter2_stem("generously"), "generous");  // gener- prefix sets R1
  EXPECT_EQ(porter2_stem("communism"), "communism");
  EXPECT_EQ(porter2_stem("arsenal"), "arsenal");
  EXPECT_EQ(porter2_stem("pasted"), "paste");
  EXPECT_EQ(porter2_stem("added"), "add");
  EXPECT_EQ(porter2_stem("hopped"), "hop");
  EXPECT_EQ(porter2_stem("hoping"), "hope");
  EXPECT_EQ(porter2_stem("caresses"), "caress");
  EXPECT_EQ(porter2_stem("ties"), "tie");
  EXPECT_EQ(porter2_stem("cried"), "cri");
  EXPECT_EQ(porter2_stem("gas"), "gas");
  EXPECT_EQ(porter2_stem("gaps"), "gap");
  EXPECT_EQ(porter2_stem("sayings"), "say");  // y after a vowel is a consonant
  EXPECT_EQ(porter2_stem("hopefulness"), "hope");
  EXPECT_EQ(porter2_stem("vietnamization"), "vietnam");
  EXPECT_EQ(porter2_stem("geologist"), "geolog");
  EXPECT_EQ(porter2_stem("replacement"), "replac");
  EXPECT_EQ(porter2_stem("adoption"), "adopt");
}

TEST(Porter2, FoodWords) {
  EXPECT_EQ(porter2_stem("biryanis"), "biryani");
  EXPECT_EQ(porter2_stem("noodles"), "noodl");
  EXPECT_EQ(porter2_stem("dumplings"), "dumpl");
  EXPECT_EQ(porter2_stem("tacos"), "taco");
  EXPECT_EQ(porter2_stem("potatoes"), "potato");
  EXPECT_EQ(porter2_stem("tomatoes"), "tomato");
  EXPECT_EQ(porter2_stem("chilies"), "chili");
  EXPECT_EQ(porter2_stem("momos"), "momo");
  EXPECT_EQ(porter2_stem("fries"), porter2_stem("fried"));
  EXPECT_EQ(porter2_stem("spicy"), "spici");
  EXPECT_EQ(porter2_stem("grilled"), "grill");
  EXPECT_EQ(porter2_stem("roasted"), "roast");
  EXPECT_EQ(porter2_stem("kebabs"), "kebab");
  EXPECT_EQ(porter2_stem("enchiladas"), "enchilada");
  EXPECT_EQ(porter2_stem("pastries"), "pastri");
  EXPECT_EQ(porter2_stem("curries"), porter2_stem("curried"));
  EXPECT_EQ(porter2_stem("samosas"), "samosa");
  EXPECT_EQ(porter2_stem("pasta"), "pasta");
  EXPECT_EQ(porter2_stem("hummus"), "hummus");
  EXPECT_EQ(porter2_stem("caramelized"), "caramel");
}

TEST(Porter2, ShortNonAsciiDigitsAndApostrophes) {
  for (const char* w : {"", "a", "is", "us", "ys"}) EXPECT_EQ(porter2_stem(w), w);
  EXPECT_EQ(porter2_stem("caf\xc3\xa9s"), "caf\xc3\xa9s");  // "cafés": non-ASCII, untouched
  EXPECT_EQ(porter2_stem("cr\xc3\xa8mes"), "cr\xc3\xa8mes");
  EXPECT_EQ(porter2_stem("7up"), "7up");
  EXPECT_EQ(porter2_stem("1990s"), "1990s");
  EXPECT_EQ(porter2_stem("covid19"), "covid19");
  EXPECT_EQ(porter2_stem("dog's"), "dog");
  EXPECT_EQ(porter2_stem("dogs'"), "dog");
  EXPECT_EQ(porter2_stem("'yes"), "yes");
  EXPECT_EQ(porter2_stem("''s"), "");
}

TEST(Porter2, InPlaceMatchesCopy) {
  for (const char* w : {"generalizations", "biryanis", "skies", "eying", "x", "dying"}) {
    std::string s = w;
    porter2_stem_inplace(s);
    EXPECT_EQ(s, porter2_stem(w));
  }
}

// Golden data produced by the Snowball project's reference implementation
// (Python `snowballstemmer`); see the header of the TSV. Requires 100% agreement.
TEST(Porter2, GoldenFileMatchesSnowballReference) {
  const auto path = std::filesystem::path(__FILE__).parent_path() / "data" / "porter2_golden.tsv";
  std::ifstream in(path);
  if (!in) {
    ::dctest::report_failure(__FILE__, __LINE__, "cannot open " + path.string());
    return;
  }

  int total = 0, mismatches = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) {
      ::dctest::report_failure(__FILE__, __LINE__, "malformed golden line: " + line);
      ++mismatches;
      continue;
    }
    const std::string word = line.substr(0, tab);
    const std::string expected = line.substr(tab + 1);
    std::string got = word;
    porter2_stem_inplace(got);
    ++total;
    if (got != expected || porter2_stem(word) != expected) {
      if (++mismatches <= 20) {
        ::dctest::report_failure(__FILE__, __LINE__,
                                 word + ": expected '" + expected + "', got '" + got + "'");
      }
    }
  }
  EXPECT_GE(total, 2900);
  EXPECT_EQ(mismatches, 0);
  std::cout << "    porter2 golden: " << (total - mismatches) << "/" << total << " agree\n";
}
