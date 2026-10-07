// Geospatial helpers: great-circle distance, point-in-polygon, and H3 wrappers.
//
// Delivery zones are indexed as *compacted* H3 cover sets: a polygon is filled
// at `cover_res`, then compacted (7 complete children -> parent), so a 5 km
// zone costs ~10-20 cells instead of ~100. A query for "who delivers to me"
// looks up the user's cell and all its ancestors — at most cover_res+1
// bitmap lookups, unioned.
#pragma once

#include <cstdint>
#include <vector>

namespace dc::geo {

inline constexpr double kEarthRadiusKm = 6371.0088;

double haversine_km(double lat1, double lon1, double lat2, double lon2);

// Ray casting over a simple polygon given as interleaved (lat, lon) floats.
bool point_in_polygon(double lat, double lon, const float* latlon, size_t num_vertices);

// Circle approximated by an n-gon (lat, lon interleaved).
std::vector<float> circle_polygon(double lat, double lon, double radius_km, int n = 16);

uint64_t cell_of(double lat, double lon, int res);
uint64_t parent_of(uint64_t cell, int res);
int resolution_of(uint64_t cell);
std::vector<uint64_t> grid_disk(uint64_t cell, int k);
double edge_length_km(int res);

// Compacted H3 cover of a polygon (interleaved lat/lon). Always contains the
// cell of (anchor_lat, anchor_lon) so tiny zones are never empty.
std::vector<uint64_t> polygon_cover(const float* latlon, size_t num_vertices, int res, double anchor_lat,
                                    double anchor_lon, bool compact = true);

// Cells whose union covers a disk of radius_km around (lat, lon) at `res`.
std::vector<uint64_t> disk_cells(double lat, double lon, double radius_km, int res);

}  // namespace dc::geo
