#pragma once

#include <atomic>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <sstream>
#include <string>

namespace dc {

enum class LogLevel { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3 };

inline std::atomic<int>& log_level() {
  static std::atomic<int> lvl{static_cast<int>(LogLevel::kInfo)};
  return lvl;
}

inline void log_write(LogLevel lvl, const std::string& msg) {
  static std::mutex mu;
  static const char* names[] = {"DEBUG", "INFO", "WARN", "ERROR"};
  std::time_t t = std::time(nullptr);
  char ts[32];
  std::strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", std::localtime(&t));
  std::lock_guard<std::mutex> lk(mu);
  std::fprintf(stderr, "%s [%s] %s\n", ts, names[static_cast<int>(lvl)], msg.c_str());
}

}  // namespace dc

#define DC_LOG(level, expr)                                                    \
  do {                                                                         \
    if (static_cast<int>(::dc::LogLevel::level) >= ::dc::log_level().load()) { \
      std::ostringstream dc_log_os;                                            \
      dc_log_os << expr;                                                       \
      ::dc::log_write(::dc::LogLevel::level, dc_log_os.str());                 \
    }                                                                          \
  } while (0)

#define LOG_DEBUG(expr) DC_LOG(kDebug, expr)
#define LOG_INFO(expr) DC_LOG(kInfo, expr)
#define LOG_WARN(expr) DC_LOG(kWarn, expr)
#define LOG_ERROR(expr) DC_LOG(kError, expr)
