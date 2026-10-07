#include <chrono>
#include <cstring>

#include "testing.h"

int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int passed = 0, failed = 0, skipped = 0;
  for (auto& tc : dctest::registry()) {
    std::string full = tc.suite + "." + tc.name;
    if (filter && full.find(filter) == std::string::npos) {
      ++skipped;
      continue;
    }
    dctest::failures_in_current_test() = 0;
    auto t0 = std::chrono::steady_clock::now();
    try {
      tc.fn();
    } catch (const dctest::AssertionAbort&) {
      // failure already recorded
    } catch (const std::exception& e) {
      dctest::report_failure(__FILE__, __LINE__, std::string("uncaught exception: ") + e.what());
    } catch (...) {
      dctest::report_failure(__FILE__, __LINE__, "uncaught non-std exception");
    }
    auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (dctest::failures_in_current_test() == 0) {
      ++passed;
      std::cout << "[  OK  ] " << full << " (" << ms << " ms)\n";
    } else {
      ++failed;
      std::cout << "[ FAIL ] " << full << "\n";
    }
  }
  std::cout << "\n" << passed << " passed, " << failed << " failed";
  if (skipped) std::cout << ", " << skipped << " filtered out";
  std::cout << "\n";
  return failed == 0 ? 0 : 1;
}
