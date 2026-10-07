// dishcovery-suggest: autocomplete + fuzzy term service backed by the FST.
//
//   dishcovery-suggest build --input suggestions.tsv --out suggest.fst
//   dishcovery-suggest serve --fst suggest.fst --port 7100
//
// suggestions.tsv: "<phrase>\t<weight>" (unsorted, duplicates keep the max weight).
//
//   GET /suggest?q=chick&k=10[&fuzzy=1]  top-k completions by popularity; with
//                                        fuzzy=1, typo-tolerant prefix matching
//                                        fills in when exact completions run short
//   GET /fuzzy?q=biriyani&max_edits=2&k=5   Levenshtein-automaton lookup
#include <httplib.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include "analysis/analyzer.h"
#include "fst/fst.h"
#include "util/args.h"
#include "util/fs.h"
#include "util/json.h"
#include "util/log.h"
#include "util/timer.h"

using namespace dc;

static int build(const Args& a) {
  std::ifstream in(a.required("input"));
  DC_CHECK(in.good(), "cannot open " + a.str("input"));
  std::map<std::string, uint32_t> entries;
  std::string line;
  while (std::getline(in, line)) {
    auto tab = line.rfind('\t');
    if (tab == std::string::npos) continue;
    // Normalise exactly like the analyzer so queries match keys.
    auto toks = analysis::get_analyzer("keyword").terms(line.substr(0, tab));
    if (toks.empty()) continue;
    uint32_t w = static_cast<uint32_t>(std::max(0.0, std::atof(line.c_str() + tab + 1)));
    auto& slot = entries[toks[0]];
    slot = std::max(slot, w);
  }
  fst::FstBuilder b;
  for (auto& [k, w] : entries) b.add(k, w);
  fst::Fst f = b.finish();
  ByteWriter w;
  f.serialize(w);
  write_file_atomic(a.required("out"), std::string_view(reinterpret_cast<const char*>(w.data()), w.size()));
  json report = {{"keys", f.size()}, {"states", f.num_states()}, {"bytes", w.size()}, {"memory", f.memory_bytes()}};
  std::cout << report.dump() << std::endl;
  return 0;
}

static int serve(const Args& a) {
  std::string bytes = read_file(a.required("fst"));
  ByteReader r(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
  fst::Fst f = fst::Fst::load(r);
  LOG_INFO("loaded FST with " << f.size() << " keys, " << f.num_states() << " states");
  httplib::Server svr;
  int threads = static_cast<int>(a.integer("threads", 4));
  svr.new_task_queue = [threads] { return new httplib::ThreadPool(threads); };
  svr.set_socket_options([](socket_t sock) {  // no SO_REUSEPORT: a busy port is an error
    int yes = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof yes);
  });

  svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content(json{{"ok", true}, {"keys", f.size()}}.dump(), "application/json");
  });
  svr.Get("/suggest", [&](const httplib::Request& req, httplib::Response& res) {
    Stopwatch sw;
    std::string q = analysis::normalize(req.get_param_value("q"));
    uint32_t k = req.has_param("k") ? static_cast<uint32_t>(std::stoul(req.get_param_value("k"))) : 10;
    bool fuzzy = req.has_param("fuzzy") && req.get_param_value("fuzzy") != "0";
    JsonWriter w;
    w.begin_object().kv("q", q).key("completions").begin_array();
    std::vector<std::string> seen;
    for (auto& c : f.top_k_prefix(q, k)) {
      w.begin_object().kv("text", c.key).kv("weight", c.weight).kv("distance", 0).end_object();
      seen.push_back(c.key);
    }
    if (fuzzy && seen.size() < k && q.size() >= 4) {
      uint32_t edits = q.size() >= 8 ? 2 : 1;
      for (auto& m : f.fuzzy_prefix(q, edits, k)) {
        if (seen.size() >= k) break;
        if (std::find(seen.begin(), seen.end(), m.key) != seen.end()) continue;
        w.begin_object().kv("text", m.key).kv("weight", m.weight).kv("distance", m.distance).end_object();
        seen.push_back(m.key);
      }
    }
    w.end_array().kv("time_us", sw.elapsed_us()).end_object();
    res.set_content(w.take(), "application/json");
  });
  svr.Get("/fuzzy", [&](const httplib::Request& req, httplib::Response& res) {
    Stopwatch sw;
    std::string q = analysis::normalize(req.get_param_value("q"));
    uint32_t k = req.has_param("k") ? static_cast<uint32_t>(std::stoul(req.get_param_value("k"))) : 5;
    uint32_t e = req.has_param("max_edits") ? static_cast<uint32_t>(std::stoul(req.get_param_value("max_edits"))) : 2;
    JsonWriter w;
    w.begin_object().kv("q", q).key("matches").begin_array();
    for (auto& m : f.fuzzy(q, std::min(e, 3u), k))
      w.begin_object().kv("text", m.key).kv("distance", m.distance).kv("weight", m.weight).end_object();
    w.end_array().kv("time_us", sw.elapsed_us()).end_object();
    res.set_content(w.take(), "application/json");
  });
  int port = static_cast<int>(a.integer("port", 7100));
  LOG_INFO("suggest service listening on " << port);
  if (!svr.listen(a.str("host", "0.0.0.0"), port)) fail("cannot listen on port " + std::to_string(port));
  return 0;
}

int main(int argc, char** argv) {
  Args a(argc, argv);
  try {
    if (!a.positional().empty() && a.positional()[0] == "build") return build(a);
    if (!a.positional().empty() && a.positional()[0] == "serve") return serve(a);
  } catch (const std::exception& e) {
    LOG_ERROR(e.what());
    return 1;
  }
  std::cerr << "usage: dishcovery-suggest build --input suggestions.tsv --out suggest.fst\n"
               "       dishcovery-suggest serve --fst suggest.fst [--port 7100] [--threads 4]\n";
  return 2;
}
