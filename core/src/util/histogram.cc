#include "util/histogram.h"

#include <cstdio>

namespace dc {

std::string Histogram::to_json() const {
  char buf[256];
  std::snprintf(buf, sizeof buf,
                "{\"count\":%llu,\"mean\":%.1f,\"p50\":%llu,\"p90\":%llu,\"p95\":%llu,\"p99\":%llu,"
                "\"p999\":%llu,\"max\":%llu}",
                static_cast<unsigned long long>(count()), mean(),
                static_cast<unsigned long long>(percentile(0.50)),
                static_cast<unsigned long long>(percentile(0.90)),
                static_cast<unsigned long long>(percentile(0.95)),
                static_cast<unsigned long long>(percentile(0.99)),
                static_cast<unsigned long long>(percentile(0.999)), static_cast<unsigned long long>(max()));
  return buf;
}

}  // namespace dc
