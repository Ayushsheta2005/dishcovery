// dishcovery-geostat: choose H3 resolutions from the data.
//
// For each resolution it reports (one JSON object per line):
//   * point index: docs per non-empty cell (mean/p50/p90/p99/max) — what a
//     radius query unions per cell, and how many cells a 3 km disk touches;
//   * delivery covers: cells per restaurant zone before/after compaction and
//     the docs-per-key distribution of the cover index.
// The chosen resolutions go into configs/schema.json (geo.point_res / cover_res).
//
//   dishcovery-geostat --docs data/docs.jsonl [--min-res 5 --max-res 10] [--limit N]
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include "geo/geo.h"
#include "util/args.h"
#include "util/json.h"
#include "util/timer.h"

using namespace dc;

namespace {
json dist(std::vector<double> v) {
  if (v.empty()) return json::object();
  std::sort(v.begin(), v.end());
  auto q = [&](double p) { return v[static_cast<size_t>(p * (v.size() - 1))]; };
  double sum = 0;
  for (double x : v) sum += x;
  return {{"n", v.size()}, {"mean", sum / v.size()}, {"p50", q(0.5)}, {"p90", q(0.9)}, {"p99", q(0.99)}, {"max", v.back()}};
}
}  // namespace

int main(int argc, char** argv) {
  Args a(argc, argv);
  if (!a.has("docs")) {
    std::cerr << "usage: dishcovery-geostat --docs FILE [--min-res 5] [--max-res 10] [--limit N]\n";
    return 2;
  }
  int min_res = static_cast<int>(a.integer("min-res", 5)), max_res = static_cast<int>(a.integer("max-res", 10));
  uint64_t limit = static_cast<uint64_t>(a.integer("limit", 0));
  struct Doc {
    double lat, lon;
    bool restaurant;
    std::vector<float> poly;
  };
  std::vector<Doc> docs;
  std::ifstream in(a.str("docs"));
  std::string line;
  while (std::getline(in, line)) {
    if (limit && docs.size() >= limit) break;
    json j = json::parse(line, nullptr, false);
    if (j.is_discarded() || !j.contains("lat") || !j["lat"].is_number()) continue;
    Doc d{j["lat"].get<double>(), j["lon"].get<double>(), j.value("type", std::string()) == "restaurant", {}};
    if (d.restaurant && j.contains("delivery_polygon"))
      for (auto& p : j["delivery_polygon"]) {
        d.poly.push_back(p[0].get<float>());
        d.poly.push_back(p[1].get<float>());
      }
    docs.push_back(std::move(d));
  }
  std::cerr << "loaded " << docs.size() << " docs\n";
  for (int res = min_res; res <= max_res; ++res) {
    Stopwatch sw;
    std::unordered_map<uint64_t, uint32_t> point_counts;
    for (auto& d : docs) ++point_counts[geo::cell_of(d.lat, d.lon, res)];
    std::vector<double> per_cell;
    for (auto& [c, n] : point_counts) per_cell.push_back(n);
    double edge = geo::edge_length_km(res);
    size_t disk3 = geo::disk_cells(39.95, -75.16, 3.0, res).size();

    std::vector<double> cover_raw, cover_compact;
    std::unordered_map<uint64_t, uint32_t> cover_counts;
    size_t sampled = 0;
    for (auto& d : docs) {
      if (!d.restaurant || d.poly.size() < 6) continue;
      if (++sampled > 20000) break;  // covers are expensive at fine resolutions
      auto raw = geo::polygon_cover(d.poly.data(), d.poly.size() / 2, res, d.lat, d.lon, false);
      auto cmp = geo::polygon_cover(d.poly.data(), d.poly.size() / 2, res, d.lat, d.lon, true);
      cover_raw.push_back(raw.size());
      cover_compact.push_back(cmp.size());
      for (uint64_t c : cmp) ++cover_counts[c];
    }
    std::vector<double> per_cover_key;
    for (auto& [c, n] : cover_counts) per_cover_key.push_back(n);
    json out = {{"res", res},
                {"edge_km", edge},
                {"nonempty_point_cells", point_counts.size()},
                {"docs_per_point_cell", dist(per_cell)},
                {"cells_in_3km_disk", disk3},
                {"cover_cells_raw", dist(cover_raw)},
                {"cover_cells_compacted", dist(cover_compact)},
                {"restaurants_per_cover_key", dist(per_cover_key)},
                {"seconds", sw.elapsed_s()}};
    std::cout << out.dump() << std::endl;
  }
  return 0;
}
