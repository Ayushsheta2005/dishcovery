// Shared primitive types and assertion macros for the Dishcovery engine.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace dc {

// Segment-local document id. Every segment numbers its docs 0..N-1.
using DocId = uint32_t;
inline constexpr DocId kNoMoreDocs = 0xFFFFFFFFu;

// Sentinel used by iterators before the first call to next()/advance().
inline constexpr DocId kUnpositioned = 0xFFFFFFFEu;

class DcError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[noreturn]] inline void fail(const std::string& msg) { throw DcError(msg); }

}  // namespace dc

#if defined(__GNUC__) || defined(__clang__)
#define DC_LIKELY(x) __builtin_expect(!!(x), 1)
#define DC_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define DC_LIKELY(x) (x)
#define DC_UNLIKELY(x) (x)
#endif

#define DC_STR2(x) #x
#define DC_STR(x) DC_STR2(x)

// Always-on invariant check. Index corruption must never be silently ignored.
#define DC_CHECK(cond, msg)                                                   \
  do {                                                                        \
    if (DC_UNLIKELY(!(cond)))                                                 \
      ::dc::fail(std::string(__FILE__ ":" DC_STR(__LINE__) " check failed: ") \
                 + #cond + " — " + (msg));                                    \
  } while (0)
