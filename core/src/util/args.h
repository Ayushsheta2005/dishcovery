// Tiny "--key value" / "--flag" command-line parser for the tools.
#pragma once

#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "util/common.h"

namespace dc {

class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("--", 0) == 0) {
        std::string key = a.substr(2);
        auto eq = key.find('=');
        if (eq != std::string::npos) {
          kv_[key.substr(0, eq)] = key.substr(eq + 1);
        } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
          kv_[key] = argv[++i];
        } else {
          kv_[key] = "1";
        }
      } else {
        positional_.push_back(a);
      }
    }
  }
  bool has(const std::string& k) const { return kv_.count(k) > 0; }
  std::string str(const std::string& k, const std::string& def = "") const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : it->second;
  }
  std::string required(const std::string& k) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) fail("missing required argument --" + k);
    return it->second;
  }
  long long integer(const std::string& k, long long def) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : std::atoll(it->second.c_str());
  }
  double real(const std::string& k, double def) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : std::atof(it->second.c_str());
  }
  bool flag(const std::string& k, bool def = false) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) return def;
    return it->second != "0" && it->second != "false" && it->second != "no";
  }
  const std::vector<std::string>& positional() const { return positional_; }

 private:
  std::map<std::string, std::string> kv_;
  std::vector<std::string> positional_;
};

}  // namespace dc
