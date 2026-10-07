#include "geo/geo.h"

#include <h3api.h>

#include <algorithm>
#include <cmath>

#include "util/common.h"

namespace dc::geo {

namespace {
constexpr double kDegToRad = M_PI / 180.0;
}

double haversine_km(double lat1, double lon1, double lat2, double lon2) {
  double p1 = lat1 * kDegToRad, p2 = lat2 * kDegToRad;
  double dp = (lat2 - lat1) * kDegToRad, dl = (lon2 - lon1) * kDegToRad;
  double a = std::sin(dp / 2) * std::sin(dp / 2) + std::cos(p1) * std::cos(p2) * std::sin(dl / 2) * std::sin(dl / 2);
  return 2 * kEarthRadiusKm * std::asin(std::min(1.0, std::sqrt(a)));
}

bool point_in_polygon(double lat, double lon, const float* v, size_t n) {
  if (n < 3) return false;
  bool inside = false;
  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    double yi = v[2 * i], xi = v[2 * i + 1];
    double yj = v[2 * j], xj = v[2 * j + 1];
    bool crosses = (yi > lat) != (yj > lat);
    if (crosses && lon < (xj - xi) * (lat - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

std::vector<float> circle_polygon(double lat, double lon, double radius_km, int n) {
  std::vector<float> out;
  out.reserve(2 * n);
  double dlat = radius_km / 111.32;
  double dlon = radius_km / (111.32 * std::max(0.05, std::cos(lat * kDegToRad)));
  for (int i = 0; i < n; ++i) {
    double t = 2 * M_PI * i / n;
    out.push_back(static_cast<float>(lat + dlat * std::sin(t)));
    out.push_back(static_cast<float>(lon + dlon * std::cos(t)));
  }
  return out;
}

uint64_t cell_of(double lat, double lon, int res) {
  LatLng g{lat * kDegToRad, lon * kDegToRad};
  H3Index out = 0;
  if (latLngToCell(&g, res, &out) != E_SUCCESS) return 0;
  return out;
}

uint64_t parent_of(uint64_t cell, int res) {
  H3Index p = 0;
  if (cellToParent(cell, res, &p) != E_SUCCESS) return 0;
  return p;
}

int resolution_of(uint64_t cell) { return getResolution(cell); }

std::vector<uint64_t> grid_disk(uint64_t cell, int k) {
  int64_t max = 0;
  if (maxGridDiskSize(k, &max) != E_SUCCESS) return {cell};
  std::vector<H3Index> out(static_cast<size_t>(max), 0);
  if (gridDisk(cell, k, out.data()) != E_SUCCESS) return {cell};
  std::vector<uint64_t> cells;
  cells.reserve(out.size());
  for (H3Index c : out)
    if (c) cells.push_back(c);
  return cells;
}

double edge_length_km(int res) {
  double km = 0;
  getHexagonEdgeLengthAvgKm(res, &km);
  return km;
}

std::vector<uint64_t> polygon_cover(const float* v, size_t n, int res, double anchor_lat, double anchor_lon,
                                    bool compact) {
  std::vector<uint64_t> cells;
  if (n >= 3) {
    std::vector<LatLng> verts(n);
    for (size_t i = 0; i < n; ++i) verts[i] = {v[2 * i] * kDegToRad, v[2 * i + 1] * kDegToRad};
    GeoPolygon poly{};
    poly.geoloop.numVerts = static_cast<int>(n);
    poly.geoloop.verts = verts.data();
    poly.numHoles = 0;
    poly.holes = nullptr;
    int64_t max = 0;
    if (maxPolygonToCellsSize(&poly, res, 0, &max) == E_SUCCESS && max > 0) {
      std::vector<H3Index> out(static_cast<size_t>(max), 0);
      if (polygonToCells(&poly, res, 0, out.data()) == E_SUCCESS)
        for (H3Index c : out)
          if (c) cells.push_back(c);
    }
  }
  if (!std::isnan(anchor_lat)) {
    uint64_t a = cell_of(anchor_lat, anchor_lon, res);
    if (a) cells.push_back(a);
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  if (!compact || cells.size() < 7) return cells;
  std::vector<H3Index> compacted(cells.size(), 0);
  if (compactCells(cells.data(), compacted.data(), static_cast<int64_t>(cells.size())) != E_SUCCESS) return cells;
  std::vector<uint64_t> out;
  for (H3Index c : compacted)
    if (c) out.push_back(c);
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<uint64_t> disk_cells(double lat, double lon, double radius_km, int res) {
  uint64_t center = cell_of(lat, lon, res);
  if (!center) return {};
  // Hex centres in ring k are ~k * sqrt(3) * edge apart; +1 ring covers the
  // boundary cells that intersect the disk.
  double step = std::sqrt(3.0) * edge_length_km(res);
  int k = static_cast<int>(std::ceil(radius_km / std::max(step, 1e-6))) + 1;
  k = std::min(k, 200);
  return grid_disk(center, k);
}

}  // namespace dc::geo
