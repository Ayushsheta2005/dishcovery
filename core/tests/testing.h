// Minimal dependency-free unit test harness.
//
//   TEST(Suite, Name) { EXPECT_EQ(a, b); ASSERT_TRUE(x); }
//
// All test files are linked into one binary with test_main.cc. Run with an
// optional substring filter: ./dishcovery_tests roaring
#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace dctest {

struct TestCase {
  std::string suite;
  std::string name;
  std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

struct Registrar {
  Registrar(const char* suite, const char* name, std::function<void()> fn) {
    registry().push_back({suite, name, std::move(fn)});
  }
};

// Thrown by ASSERT_* to abort the current test.
struct AssertionAbort {};

inline int& failures_in_current_test() {
  static int n = 0;
  return n;
}

inline void report_failure(const char* file, int line, const std::string& msg) {
  ++failures_in_current_test();
  std::cerr << "    " << file << ":" << line << ": " << msg << "\n";
}

template <typename A, typename B>
std::string format_cmp(const char* ea, const char* eb, const A& a, const B& b, const char* op) {
  std::ostringstream os;
  os << "expected " << ea << " " << op << " " << eb << " (got " << a << " vs " << b << ")";
  return os.str();
}

}  // namespace dctest

#define DCT_CONCAT2(a, b) a##b
#define DCT_CONCAT(a, b) DCT_CONCAT2(a, b)

#define TEST(suite, name)                                                         \
  static void DCT_CONCAT(dct_test_, DCT_CONCAT(suite, DCT_CONCAT(_, name)))();    \
  static ::dctest::Registrar DCT_CONCAT(dct_reg_, DCT_CONCAT(suite, DCT_CONCAT(_, name)))( \
      #suite, #name, &DCT_CONCAT(dct_test_, DCT_CONCAT(suite, DCT_CONCAT(_, name))));      \
  static void DCT_CONCAT(dct_test_, DCT_CONCAT(suite, DCT_CONCAT(_, name)))()

#define DCT_CMP(a, b, op, fatal)                                                          \
  do {                                                                                    \
    const auto dct_a = (a);                                                               \
    const auto dct_b = (b);                                                               \
    if (!(dct_a op dct_b)) {                                                              \
      ::dctest::report_failure(__FILE__, __LINE__,                                        \
                               ::dctest::format_cmp(#a, #b, dct_a, dct_b, #op));          \
      if (fatal) throw ::dctest::AssertionAbort{};                                        \
    }                                                                                     \
  } while (0)

#define EXPECT_EQ(a, b) DCT_CMP(a, b, ==, false)
#define EXPECT_NE(a, b) DCT_CMP(a, b, !=, false)
#define EXPECT_LT(a, b) DCT_CMP(a, b, <, false)
#define EXPECT_LE(a, b) DCT_CMP(a, b, <=, false)
#define EXPECT_GT(a, b) DCT_CMP(a, b, >, false)
#define EXPECT_GE(a, b) DCT_CMP(a, b, >=, false)
#define ASSERT_EQ(a, b) DCT_CMP(a, b, ==, true)
#define ASSERT_NE(a, b) DCT_CMP(a, b, !=, true)
#define ASSERT_LT(a, b) DCT_CMP(a, b, <, true)
#define ASSERT_LE(a, b) DCT_CMP(a, b, <=, true)
#define ASSERT_GT(a, b) DCT_CMP(a, b, >, true)
#define ASSERT_GE(a, b) DCT_CMP(a, b, >=, true)

#define EXPECT_TRUE(c)                                                          \
  do {                                                                          \
    if (!(c)) ::dctest::report_failure(__FILE__, __LINE__, "expected true: " #c); \
  } while (0)
#define EXPECT_FALSE(c) EXPECT_TRUE(!(c))
#define ASSERT_TRUE(c)                                                             \
  do {                                                                             \
    if (!(c)) {                                                                    \
      ::dctest::report_failure(__FILE__, __LINE__, "expected true: " #c);          \
      throw ::dctest::AssertionAbort{};                                            \
    }                                                                              \
  } while (0)
#define ASSERT_FALSE(c) ASSERT_TRUE(!(c))

#define EXPECT_NEAR(a, b, tol)                                                          \
  do {                                                                                  \
    double dct_x = static_cast<double>(a), dct_y = static_cast<double>(b);              \
    if (std::fabs(dct_x - dct_y) > (tol)) {                                             \
      std::ostringstream dct_os;                                                        \
      dct_os << "expected |" #a " - " #b "| <= " << (tol) << " (got " << dct_x << " vs " \
             << dct_y << ")";                                                           \
      ::dctest::report_failure(__FILE__, __LINE__, dct_os.str());                       \
    }                                                                                   \
  } while (0)

#define EXPECT_THROW(stmt)                                                              \
  do {                                                                                  \
    bool dct_threw = false;                                                             \
    try {                                                                               \
      stmt;                                                                             \
    } catch (...) {                                                                     \
      dct_threw = true;                                                                 \
    }                                                                                   \
    if (!dct_threw) ::dctest::report_failure(__FILE__, __LINE__, "expected throw: " #stmt); \
  } while (0)
