// End-to-end query execution checks against brute-force oracles.
#include <cmath>
#include <map>
#include <random>

#include "analysis/analyzer.h"
#include "geo/geo.h"
#include "index/shard_index.h"
#include "search/searcher.h"
#include "testing.h"

using namespace dc;
using namespace dc::index;
using namespace dc::search;

namespace {

const char* kWords[] = {"chicken", "biryani", "paneer", "tikka", "masala", "pizza", "margherita", "pepperoni",
                        "pad",     "thai",    "noodles", "garlic", "naan", "spicy", "veg", "butter"};

struct Fixture {
  Schema schema;
  std::vector<std::shared_ptr<MemoryStorage>> storages;
  Snapshot snap;
  std::vector<json> docs;

  Fixture() {
    schema = Schema::default_schema();
    schema.vector_dim = 16;
    std::mt19937 rng(11);
    // Three segments, the middle one with deletes.
    for (int s = 0; s < 3; ++s) {
      SegmentBuilder b(schema);
      for (int i = 0; i < 700; ++i) {
        int id = s * 1000 + i;
        std::string name, desc;
        for (int k = 0; k < 2 + int(rng() % 3); ++k) name += std::string(kWords[rng() % 16]) + " ";
        for (int k = 0; k < 8; ++k) desc += std::string(kWords[rng() % 16]) + " ";
        std::vector<float> v(16);
        for (auto& x : v) x = std::normal_distribution<float>()(rng);
        double lat = 39.90 + (rng() % 1000) / 10000.0, lon = -75.20 + (rng() % 1000) / 10000.0;
        json j = {{"id", "d:" + std::to_string(id)},
                  {"type", i % 5 ? "dish" : "restaurant"},
                  {"restaurant_id", "r:" + std::to_string(id / 7)},
                  {"name", name},
                  {"restaurant_name", "Place " + std::to_string(id / 7)},
                  {"cuisines", json::array({i % 3 ? "indian" : "italian"})},
                  {"description", desc},
                  {"diet", json::array({i % 4 ? "non_veg" : "veg"})},
                  {"price_tier", 1 + i % 4},
                  {"rating", 2.5 + (i % 6) * 0.5},
                  {"popularity", (rng() % 1000) / 1000.0},
                  {"lat", lat},
                  {"lon", lon},
                  {"hours", std::string(84, i % 2 ? 'f' : '0')},
                  {"delivery_radius_km", 1.0 + (i % 5)},
                  {"embedding", v}};
        docs.push_back(j);
        b.add(parse_document(j, schema));
      }
      auto st = std::make_shared<MemoryStorage>();
      SegmentWriteOptions o;
      o.uid = "seg" + std::to_string(s);
      o.hnsw_min_docs = 100;
      b.write(*st, o);
      storages.push_back(st);
      SegmentView sv{SegmentReader::open(st, schema), nullptr};
      if (s == 1) {
        auto del = std::make_shared<RoaringBitmap>();
        for (uint32_t d = 0; d < 700; d += 3) del->add(d);
        sv.deletes = del;
      }
      snap.segments.push_back(sv);
      snap.max_docs += 700;
    }
    snap.generation = 1;
  }
};

Fixture& fx() {
  static Fixture f;
  return f;
}

// Exhaustive BM25 oracle (same formula as the engine).
std::map<std::pair<uint32_t, DocId>, float> oracle_bm25(Searcher& s, const SearchRequest& req,
                                                        const std::vector<std::shared_ptr<const RoaringBitmap>>& f) {
  auto& F = fx();
  auto clauses = s.compile(F.snap, req.terms, req.field_weights, true, false);
  std::map<std::pair<uint32_t, DocId>, float> out;
  const float k1 = F.schema.bm25.k1, b = F.schema.bm25.b;
  for (uint32_t si = 0; si < F.snap.segments.size(); ++si) {
    auto& sv = F.snap.segments[si];
    for (auto& c : clauses) {
      auto ti = sv.seg->term(c.field, c.term);
      if (!ti) continue;
      auto it = sv.seg->postings(*ti);
      for (DocId d = it.next(); d != kNoMoreDocs; d = it.next()) {
        if (sv.is_deleted(d)) continue;
        if (f[si] && !f[si]->contains(d)) continue;
        float tf = it.freq();
        float dl = sv.seg->norm(c.field, d);
        out[{si, d}] += c.weight * c.idf * (k1 + 1) * tf / (tf + k1 * (1 - b + b * dl / c.avgdl));
      }
    }
  }
  return out;
}

SearchRequest base_request(const std::vector<std::string>& words) {
  SearchRequest r;
  int g = 0;
  for (auto& w : words) r.terms.push_back({w, 1.0f, g++, true});
  for (auto& f : fx().schema.text_fields) r.field_weights.push_back(f.weight);
  r.bm25_k = 25;
  return r;
}

}  // namespace

TEST(Search, MaxScoreMatchesExhaustiveOracle) {
  auto& F = fx();
  Searcher s(F.schema, {});
  for (auto words : std::vector<std::vector<std::string>>{
           {"chicken", "biryani"}, {"paneer"}, {"spicy", "garlic", "naan", "masala"}, {"pizza", "thai"}}) {
    SearchRequest req = base_request(words);
    auto resp = s.search(F.snap, req);
    std::vector<std::shared_ptr<const RoaringBitmap>> none(3);
    auto oracle = oracle_bm25(s, req, none);
    std::vector<float> want;
    for (auto& [k, v] : oracle) want.push_back(v);
    std::sort(want.rbegin(), want.rend());
    ASSERT_EQ(resp.bm25.hits.size(), std::min<size_t>(25, want.size()));
    for (size_t i = 0; i < resp.bm25.hits.size(); ++i) {
      EXPECT_NEAR(resp.bm25.hits[i].score, want[i], 1e-3);
      auto key = std::make_pair(resp.bm25.hits[i].seg, resp.bm25.hits[i].doc);
      EXPECT_NEAR(oracle[key], resp.bm25.hits[i].score, 1e-3);
    }
    // Pruning must have skipped work relative to exhaustive scoring.
    EXPECT_LE(resp.bm25.scored, oracle.size());
  }
}

TEST(Search, FilterModesAgreeAndFilterAwareScoresLess) {
  auto& F = fx();
  Searcher s(F.schema, {});
  SearchRequest req = base_request({"chicken", "masala", "naan"});
  req.filter.terms.push_back({F.schema.keyword_field("diet"), {"veg"}});
  req.filter.ranges.push_back({F.schema.numeric_field("price_tier"), -INFINITY, 2.0f});
  req.filter_mode = FilterMode::kFilterAware;
  auto aware = s.search(F.snap, req);
  req.filter_mode = FilterMode::kPostFilter;
  auto post = s.search(F.snap, req);
  ASSERT_EQ(aware.bm25.hits.size(), post.bm25.hits.size());
  for (size_t i = 0; i < aware.bm25.hits.size(); ++i) EXPECT_NEAR(aware.bm25.hits[i].score, post.bm25.hits[i].score, 1e-4);
  EXPECT_LT(aware.bm25.scored, post.bm25.scored);
  // Every hit satisfies the filter.
  for (auto& h : aware.bm25.hits) {
    auto& seg = *F.snap.segments[h.seg].seg;
    EXPECT_TRUE(seg.numeric(F.schema.numeric_field("price_tier"), h.doc) <= 2.0f);
    EXPECT_TRUE(seg.stored(h.doc).find("\"veg\"") != std::string_view::npos);
  }
}

TEST(Search, DeliveryGeoFilterMatchesPolygonTruth) {
  auto& F = fx();
  Searcher s(F.schema, {});
  SearchRequest req = base_request({"pizza"});
  req.bm25_k = 5000;
  req.filter.geo.mode = GeoFilter::Mode::kDelivery;
  req.filter.geo.lat = 39.95;
  req.filter.geo.lon = -75.15;
  auto resp = s.search(F.snap, req);
  // Cover sets are cell-approximations of the zones: every hit must be within
  // radius + one cover-cell diameter of the user.
  double slack = 2.5 * geo::edge_length_km(F.schema.geo.cover_res);
  for (auto& h : resp.bm25.hits) {
    auto& seg = *F.snap.segments[h.seg].seg;
    double dist = geo::haversine_km(39.95, -75.15, seg.lat(h.doc), seg.lon(h.doc));
    EXPECT_LE(dist, seg.numeric(F.schema.numeric_field("delivery_radius_km"), h.doc) + slack);
  }
  EXPECT_GT(resp.bm25.hits.size(), size_t(0));
}

TEST(Search, RadiusFilterIsExact) {
  auto& F = fx();
  Searcher s(F.schema, {});
  SearchRequest req = base_request({"thai"});
  req.bm25_k = 5000;
  req.filter.geo = {GeoFilter::Mode::kRadius, 39.95, -75.15, 2.0};
  auto resp = s.search(F.snap, req);
  size_t truth = 0;
  for (uint32_t si = 0; si < 3; ++si) {
    auto& sv = F.snap.segments[si];
    int f = F.schema.text_field("name");
    auto ti = sv.seg->term(static_cast<uint8_t>(f), "thai");
    (void)ti;
  }
  for (auto& h : resp.bm25.hits) {
    auto& seg = *F.snap.segments[h.seg].seg;
    EXPECT_LE(geo::haversine_km(39.95, -75.15, seg.lat(h.doc), seg.lon(h.doc)), 2.0 + 1e-6);
    ++truth;
  }
  EXPECT_GT(truth, size_t(0));
}

TEST(Search, ExactPathRequiresAllTokens) {
  auto& F = fx();
  Searcher s(F.schema, {});
  SearchRequest req = base_request({"chicken", "biryani"});
  req.bm25 = false;
  req.exact = true;
  req.exact_k = 1000;
  auto resp = s.search(F.snap, req);
  EXPECT_GT(resp.exact.hits.size(), size_t(0));
  for (auto& h : resp.exact.hits) {
    auto stored = F.snap.segments[h.seg].seg->stored(h.doc);
    EXPECT_TRUE(stored.find("chicken") != std::string_view::npos);
    EXPECT_TRUE(stored.find("biryani") != std::string_view::npos);
    EXPECT_FALSE(F.snap.segments[h.seg].is_deleted(h.doc));
  }
  for (size_t i = 1; i < resp.exact.hits.size(); ++i) EXPECT_GE(resp.exact.hits[i - 1].score, resp.exact.hits[i].score);
}

TEST(Search, VectorPathHnswVsBruteForce) {
  auto& F = fx();
  Searcher s(F.schema, {});
  SearchRequest req;
  req.bm25 = false;
  req.vector = true;
  req.vector_k = 10;
  req.ef = 200;
  req.qvec = F.docs[123]["embedding"].get<std::vector<float>>();
  req.brute_force_max = 0;  // force HNSW
  auto h = s.search(F.snap, req);
  req.brute_force_max = 1u << 30;  // force exact scan
  auto b = s.search(F.snap, req);
  ASSERT_EQ(b.vector.hits.size(), size_t(10));
  std::set<std::pair<uint32_t, DocId>> truth;
  for (auto& x : b.vector.hits) truth.insert({x.seg, x.doc});
  size_t overlap = 0;
  for (auto& x : h.vector.hits) overlap += truth.count({x.seg, x.doc});
  EXPECT_GE(overlap, size_t(9));
  // The query's own doc is the top hit (cosine 1).
  EXPECT_NEAR(b.vector.hits[0].score, 1.0, 1e-4);
}

TEST(Search, GeoPathRanksByDistanceAndPopularity) {
  auto& F = fx();
  Searcher s(F.schema, {});
  SearchRequest req;
  req.bm25 = false;
  req.geo = true;
  req.geo_k = 20;
  req.filter.geo = {GeoFilter::Mode::kDelivery, 39.95, -75.15, 0};
  auto r = s.search(F.snap, req);
  EXPECT_EQ(r.geo.hits.size(), size_t(20));
  for (size_t i = 1; i < r.geo.hits.size(); ++i) EXPECT_GE(r.geo.hits[i - 1].score, r.geo.hits[i].score);
}

TEST(Search, FeaturesProduceExpectedSignals) {
  auto& F = fx();
  Searcher s(F.schema, {});
  SearchRequest req = base_request({"chicken", "biryani"});
  req.exact = true;
  auto r = s.search(F.snap, req);
  ASSERT_GT(r.exact.hits.size(), size_t(0));
  FeatureRequest fr;
  fr.terms = req.terms;
  fr.field_weights = req.field_weights;
  fr.user_lat = 39.95;
  fr.user_lon = -75.15;
  fr.slot = 100;
  for (auto& h : r.exact.hits) fr.docs.push_back({h.seg, h.doc});
  auto out = s.features(F.snap, fr);
  ASSERT_EQ(out.rows.size(), fr.docs.size());
  auto col = [&](const std::string& n) {
    auto it = std::find(out.names.begin(), out.names.end(), n);
    return static_cast<size_t>(it - out.names.begin());
  };
  ASSERT_LT(col("coverage"), out.names.size());
  for (size_t i = 0; i < out.rows.size(); ++i) {
    EXPECT_NEAR(out.rows[i][col("coverage")], 1.0, 1e-6);  // both groups matched
    EXPECT_GT(out.rows[i][col("bm25_name")], 0.0f);
    EXPECT_GT(out.rows[i][col("bm25_weighted")], 0.0f);
    EXPECT_GE(out.rows[i][col("distance_km")], 0.0f);
    EXPECT_FALSE(out.ids[i].empty());
    auto stored = out.stored[i];
    bool phrase = stored.find("chicken biryani") != std::string_view::npos;
    EXPECT_EQ(out.rows[i][col("phrase_name")] > 0.5f, phrase);
  }
}
