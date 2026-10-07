#include "search/searcher.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <queue>

#include "analysis/analyzer.h"
#include "geo/geo.h"
#include "util/timer.h"
#include "vector/distance.h"
#include "vector/hnsw.h"

namespace dc::search {

// ------------------------------------------------------------------ TermCursor

DocId TermCursor::next() {
  if (!cached_) return stream_.next();
  if (doc_ == kNoMoreDocs) return doc_;
  if (doc_ != kUnpositioned) ++idx_;
  return doc_ = idx_ < cached_->docs.size() ? cached_->docs[idx_] : kNoMoreDocs;
}

DocId TermCursor::advance(DocId target) {
  if (!cached_) return stream_.advance(target);
  if (doc_ == kNoMoreDocs) return doc_;
  if (doc_ != kUnpositioned && doc_ >= target) return doc_;
  const auto& d = cached_->docs;
  // Galloping search from the current position.
  size_t lo = idx_, step = 1, hi = idx_;
  while (hi < d.size() && d[hi] < target) {
    lo = hi;
    hi += step;
    step <<= 1;
  }
  hi = std::min(hi, d.size());
  idx_ = static_cast<size_t>(std::lower_bound(d.begin() + lo, d.begin() + hi, target) - d.begin());
  return doc_ = idx_ < d.size() ? d[idx_] : kNoMoreDocs;
}

// ------------------------------------------------------------------ helpers

namespace {

struct HitWorse {
  // Heap "greater" = better hit; the top of a min-heap is the worst kept hit.
  bool operator()(const Hit& a, const Hit& b) const {
    if (a.score != b.score) return a.score > b.score;
    if (a.seg != b.seg) return a.seg < b.seg;
    return a.doc < b.doc;
  }
};

class TopK {
 public:
  explicit TopK(uint32_t k) : k_(k) {}
  float threshold() const { return heap_.size() < k_ ? -INFINITY : heap_.front().score; }
  bool full() const { return heap_.size() >= k_; }
  void push(const Hit& h) {
    if (k_ == 0) return;
    if (heap_.size() < k_) {
      heap_.push_back(h);
      std::push_heap(heap_.begin(), heap_.end(), HitWorse());
    } else if (HitWorse()(h, heap_.front())) {
      std::pop_heap(heap_.begin(), heap_.end(), HitWorse());
      heap_.back() = h;
      std::push_heap(heap_.begin(), heap_.end(), HitWorse());
    }
  }
  std::vector<Hit> sorted() {
    std::vector<Hit> out = heap_;
    std::sort(out.begin(), out.end(), HitWorse());
    return out;
  }

 private:
  uint32_t k_;
  std::vector<Hit> heap_;
};

inline float bm25_tf(float tf, float dl, float c1, float c2) { return tf / (tf + c1 + c2 * dl); }

}  // namespace

// ------------------------------------------------------------------ Searcher

Searcher::Searcher(const index::Schema& schema, SearcherOptions opt)
    : schema_(schema), opt_(opt), filters_(opt.filter_cache_bytes), postings_cache_(opt.postings_cache_bytes) {
  f_popularity_ = schema_.numeric_field("popularity");
  f_rating_ = schema_.numeric_field("rating");
  f_name_exact_ = schema_.text_field("name_exact");
  f_name_ = schema_.text_field("name");
}

void Searcher::retire_segment(const std::string& uid) {
  filters_.invalidate_segment(uid);
  postings_cache_.invalidate_prefix(uid + "|");
}

json Searcher::cache_stats() const {
  auto f = const_cast<FilterEngine&>(filters_).cache().stats();
  auto p = postings_cache_.stats();
  auto j = [](const auto& s) {
    return json{{"hits", s.hits},       {"misses", s.misses}, {"evictions", s.evictions},
                {"entries", s.entries}, {"bytes", s.bytes},   {"capacity", s.capacity},
                {"invalidations", s.invalidations}};
  };
  return {{"filter_cache", j(f)}, {"postings_cache", j(p)}};
}

std::vector<Clause> Searcher::compile(const index::Snapshot& snap, const std::vector<QueryTerm>& terms,
                                      const std::vector<float>& field_weights, bool merge_duplicates,
                                      bool include_zero_weight) const {
  std::vector<Clause> clauses;
  std::map<std::pair<uint8_t, std::string>, size_t> seen;
  for (const auto& qt : terms) {
    for (size_t f = 0; f < schema_.text_fields.size(); ++f) {
      float fw = f < field_weights.size() ? field_weights[f] : schema_.text_fields[f].weight;
      if (fw <= 0 && !include_zero_weight) continue;
      auto toks = analysis::get_analyzer(schema_.text_fields[f].analyzer).terms(qt.text);
      if (toks.empty()) continue;
      // A multi-token alternative ("cottage cheese") splits its weight.
      float w = qt.weight * std::max(fw, 0.0f) / static_cast<float>(toks.size());
      for (auto& t : toks) {
        auto key = std::make_pair(static_cast<uint8_t>(f), t);
        if (merge_duplicates) {
          auto it = seen.find(key);
          if (it != seen.end()) {
            clauses[it->second].weight += w;
            clauses[it->second].primary |= qt.primary;
            continue;
          }
          seen[key] = clauses.size();
        }
        Clause c;
        c.field = static_cast<uint8_t>(f);
        c.term = t;
        c.weight = w;
        c.group = qt.group;
        c.primary = qt.primary;
        clauses.push_back(std::move(c));
      }
    }
  }
  // Snapshot-global statistics: N, df and average field length.
  const double N = static_cast<double>(std::max<uint64_t>(snap.max_docs, 1));
  std::vector<double> sum_dl(schema_.text_fields.size(), 0.0);
  for (const auto& sv : snap.segments)
    for (size_t f = 0; f < sum_dl.size(); ++f) sum_dl[f] += static_cast<double>(sv.seg->meta().field_sum_dl[f]);
  for (auto& c : clauses) {
    for (const auto& sv : snap.segments)
      if (auto ti = sv.seg->term(c.field, c.term)) c.df += ti->df;
    double df = static_cast<double>(c.df);
    c.idf = static_cast<float>(std::log(1.0 + (N - df + 0.5) / (df + 0.5)));
    c.avgdl = static_cast<float>(std::max(sum_dl[c.field] / N, 1e-3));
  }
  return clauses;
}

bool Searcher::open_cursor(const index::SegmentReader& seg, const Clause& c, bool use_cache, TermCursor& cur,
                           index::TermInfo* info_out) {
  auto ti = seg.term(c.field, c.term);
  if (!ti) return false;
  if (info_out) *info_out = *ti;
  if (use_cache && seg.on_disk() && ti->df >= opt_.postings_cache_min_df && postings_cache_.capacity() > 0) {
    std::string key = seg.uid() + "|" + std::to_string(c.field) + "|" + c.term;
    auto p = postings_cache_.get(key);
    if (!p) {
      auto dp = std::make_shared<DecodedPostings>();
      dp->docs.reserve(ti->df);
      dp->freqs.reserve(ti->df);
      auto it = seg.postings(*ti);
      for (DocId d = it.next(); d != kNoMoreDocs; d = it.next()) {
        dp->docs.push_back(d);
        dp->freqs.push_back(it.freq());
      }
      postings_cache_.put(key, dp, dp->bytes());
      p = dp;
    }
    cur.init_cached(std::move(p));
  } else {
    cur.init_stream(seg.postings(*ti));
  }
  return true;
}

SearchResponse Searcher::search(const index::Snapshot& snap, const SearchRequest& req) {
  Stopwatch total;
  SearchResponse resp;
  resp.generation = snap.generation;
  resp.segments = static_cast<uint32_t>(snap.segments.size());

  // Materialise filters once per segment; every path shares them.
  Stopwatch sw;
  std::vector<std::shared_ptr<const RoaringBitmap>> fsets(snap.segments.size());
  if (!req.filter.empty()) {
    std::string key = req.filter.cache_key(schema_);
    for (size_t s = 0; s < snap.segments.size(); ++s) {
      bool hit = false;
      fsets[s] = filters_.materialize(*snap.segments[s].seg, req.filter, key, req.use_filter_cache, &hit);
      resp.filter_cache_hits += hit;
      resp.filter_cardinality += fsets[s] ? fsets[s]->cardinality() : 0;
    }
  }
  resp.filter_us = sw.lap_us();

  if (req.bm25 && !req.terms.empty()) run_bm25(snap, req, fsets, resp.bm25);
  if (req.exact && !req.terms.empty()) run_exact(snap, req, fsets, resp.exact);
  if (req.vector && !req.qvec.empty()) run_vector(snap, req, fsets, resp.vector);
  if (req.geo && req.filter.geo.mode != GeoFilter::Mode::kNone) run_geo(snap, req, fsets, resp.geo);
  resp.total_us = total.elapsed_us();
  return resp;
}

// ------------------------------------------------------------------ BM25 + MaxScore

void Searcher::run_bm25(const index::Snapshot& snap, const SearchRequest& req,
                        const std::vector<std::shared_ptr<const RoaringBitmap>>& fsets, PathResult& out) {
  Stopwatch sw;
  std::vector<Clause> clauses = compile(snap, req.terms, req.field_weights, true, false);
  const float k1 = schema_.bm25.k1, b = schema_.bm25.b;
  const bool post_filter = req.filter_mode == FilterMode::kPostFilter;
  const bool post_topk = req.filter_mode == FilterMode::kPostFilterTopK;
  const uint32_t k = post_topk ? req.bm25_k * std::max(1u, req.oversample) : req.bm25_k;
  TopK top(k);

  struct Run {
    TermCursor cur;
    float w, c1, c2, ub;
    const uint16_t* norms;
  };
  std::vector<Run> runs;
  std::vector<float> cum;

  for (size_t s = 0; s < snap.segments.size(); ++s) {
    const auto& sv = snap.segments[s];
    const auto& seg = *sv.seg;
    const RoaringBitmap* F = fsets[s].get();
    if (F && F->empty() && !post_filter && !post_topk) continue;
    const bool filter_in_loop = F && req.filter_mode == FilterMode::kFilterAware;

    runs.clear();
    for (const auto& c : clauses) {
      if (c.weight <= 0) continue;
      Run r;
      index::TermInfo ti;
      if (!open_cursor(seg, c, req.use_postings_cache, r.cur, &ti)) continue;
      r.w = c.weight * c.idf * (k1 + 1.0f);
      r.c1 = k1 * (1.0f - b);
      r.c2 = k1 * b / c.avgdl;
      r.norms = seg.norms(c.field);
      r.ub = r.w * bm25_tf(static_cast<float>(ti.max_tf), static_cast<float>(ti.min_dl), r.c1, r.c2);
      runs.push_back(std::move(r));
    }
    if (runs.empty()) continue;
    std::sort(runs.begin(), runs.end(), [](const Run& a, const Run& b) { return a.ub < b.ub; });
    const size_t n = runs.size();
    cum.assign(n, 0.0f);
    for (size_t i = 0; i < n; ++i) cum[i] = runs[i].ub + (i ? cum[i - 1] : 0.0f);

    // Exhaustive scoring (no pruning) is the post-filter baseline.
    auto theta = [&]() -> float { return post_filter ? -INFINITY : top.threshold(); };
    size_t fe = 0;  // clauses [0, fe) are non-essential
    auto update_fe = [&] {
      float t = theta();
      while (fe < n && cum[fe] <= t) ++fe;
    };
    update_fe();
    for (size_t i = fe; i < n; ++i) runs[i].cur.next();
    std::optional<RoaringBitmap::Iterator> fit;
    if (filter_in_loop) fit.emplace(*F);

    while (fe < n) {
      DocId d = kNoMoreDocs;
      for (size_t i = fe; i < n; ++i) d = std::min(d, runs[i].cur.doc());
      if (d == kNoMoreDocs) break;

      if (filter_in_loop) {
        DocId f = fit->value() != kUnpositioned && fit->value() >= d ? fit->value() : fit->advance(d);
        if (f == kNoMoreDocs) break;
        if (f > d) {
          for (size_t i = fe; i < n; ++i)
            if (runs[i].cur.doc() < f) runs[i].cur.advance(f);
          continue;
        }
      }
      if (sv.is_deleted(d)) {
        for (size_t i = fe; i < n; ++i)
          if (runs[i].cur.doc() == d) runs[i].cur.next();
        continue;
      }
      float score = 0;
      for (size_t i = fe; i < n; ++i) {
        Run& r = runs[i];
        if (r.cur.doc() == d) {
          score += r.w * bm25_tf(static_cast<float>(r.cur.freq()), r.norms[d], r.c1, r.c2);
          r.cur.next();
        }
      }
      const float t = theta();
      for (size_t i = fe; i-- > 0;) {
        if (score + cum[i] <= t) break;  // remaining clauses cannot lift d into the top-k
        Run& r = runs[i];
        if (r.cur.doc() == kUnpositioned || r.cur.doc() < d) r.cur.advance(d);
        if (r.cur.doc() == d) score += r.w * bm25_tf(static_cast<float>(r.cur.freq()), r.norms[d], r.c1, r.c2);
      }
      ++out.scored;
      if (post_filter && F && !F->contains(d)) continue;
      ++out.matched;
      if (score > t || !top.full()) {
        top.push({static_cast<uint32_t>(s), d, score});
        update_fe();
      }
    }
  }
  out.hits = top.sorted();
  if (post_topk) {
    std::vector<Hit> kept;
    for (const auto& h : out.hits) {
      const RoaringBitmap* F = fsets[h.seg].get();
      if (!F || F->contains(h.doc)) kept.push_back(h);
      if (kept.size() >= req.bm25_k) break;
    }
    out.hits = std::move(kept);
  }
  out.time_us = sw.elapsed_us();
}

// ------------------------------------------------------------------ exact match

void Searcher::run_exact(const index::Snapshot& snap, const SearchRequest& req,
                         const std::vector<std::shared_ptr<const RoaringBitmap>>& fsets, PathResult& out) {
  Stopwatch sw;
  int field = f_name_exact_ >= 0 ? f_name_exact_ : f_name_;
  if (field < 0) return;
  const auto& an = analysis::get_analyzer(schema_.text_fields[field].analyzer);
  std::vector<std::string> toks;
  for (const auto& qt : req.terms)
    if (qt.primary)
      for (auto& t : an.terms(qt.text)) toks.push_back(t);
  std::sort(toks.begin(), toks.end());
  toks.erase(std::unique(toks.begin(), toks.end()), toks.end());
  if (toks.empty()) return;
  TopK top(req.exact_k);
  for (size_t s = 0; s < snap.segments.size(); ++s) {
    const auto& sv = snap.segments[s];
    const auto& seg = *sv.seg;
    const RoaringBitmap* F = fsets[s].get();
    if (F && F->empty()) continue;
    std::vector<index::PostingsIterator> its;
    bool missing = false;
    for (auto& t : toks) {
      auto ti = seg.term(static_cast<uint8_t>(field), t);
      if (!ti) {
        missing = true;
        break;
      }
      its.push_back(seg.postings(*ti));
    }
    if (missing) continue;
    std::sort(its.begin(), its.end(), [](const auto& a, const auto& b) { return a.cost() < b.cost(); });
    std::optional<RoaringBitmap::Iterator> fit;
    if (F) fit.emplace(*F);
    // Leapfrog intersection: the rarest list proposes, the rest (and the
    // filter) confirm via skip-list advance().
    DocId d = its[0].next();
    while (d != kNoMoreDocs) {
      DocId cand = d;
      for (size_t i = 1; i < its.size() && cand == d; ++i) {
        DocId x = its[i].advance(d);
        if (x != d) cand = x;
      }
      if (cand == d && fit) {
        DocId x = fit->advance(d);
        if (x != d) cand = x;
      }
      if (cand == kNoMoreDocs) break;
      if (cand != d) {
        d = its[0].advance(cand);
        continue;
      }
      ++out.matched;
      if (!sv.is_deleted(d)) {
        float score = f_popularity_ >= 0 ? seg.numeric(f_popularity_, d) : 0.0f;
        if (f_rating_ >= 0) score += 1e-3f * seg.numeric(f_rating_, d);
        ++out.scored;
        top.push({static_cast<uint32_t>(s), d, score});
      }
      d = its[0].next();
    }
  }
  out.hits = top.sorted();
  out.time_us = sw.elapsed_us();
}

// ------------------------------------------------------------------ vector

void Searcher::run_vector(const index::Snapshot& snap, const SearchRequest& req,
                          const std::vector<std::shared_ptr<const RoaringBitmap>>& fsets, PathResult& out) {
  Stopwatch sw;
  if (req.qvec.size() != schema_.vector_dim) return;
  std::vector<float> q = req.qvec;
  vec::normalize(q.data(), q.size());
  TopK top(req.vector_k);
  for (size_t s = 0; s < snap.segments.size(); ++s) {
    const auto& sv = snap.segments[s];
    const auto& seg = *sv.seg;
    if (!seg.vectors()) continue;
    const RoaringBitmap* F = fsets[s].get();
    if (F && F->empty()) continue;
    const uint32_t dim = schema_.vector_dim;
    uint64_t admissible = F ? F->cardinality() : seg.num_docs();
    if (!seg.hnsw() || admissible <= req.brute_force_max) {
      auto score_doc = [&](uint32_t d) {
        if (sv.is_deleted(d)) return;
        ++out.scored;
        top.push({static_cast<uint32_t>(s), d, vec::dot(q.data(), seg.vector(d), dim)});
      };
      if (F) F->for_each(score_doc);
      else
        for (uint32_t d = 0; d < seg.num_docs(); ++d) score_doc(d);
    } else {
      vec::IdFilter pred = [&](uint32_t id) { return (!F || F->contains(id)) && !sv.is_deleted(id); };
      vec::SearchStats st;
      bool filtered = F || sv.deletes;
      auto res = seg.hnsw()->search(q.data(), req.vector_k, std::max(req.ef, req.vector_k),
                                    filtered ? &pred : nullptr, &st);
      out.scored += st.distance_computations;
      for (auto& r : res) top.push({static_cast<uint32_t>(s), r.id, r.score});
    }
  }
  out.hits = top.sorted();
  out.matched = out.hits.size();
  out.time_us = sw.elapsed_us();
}

// ------------------------------------------------------------------ geo

void Searcher::run_geo(const index::Snapshot& snap, const SearchRequest& req,
                       const std::vector<std::shared_ptr<const RoaringBitmap>>& fsets, PathResult& out) {
  Stopwatch sw;
  TopK top(req.geo_k);
  const double lat = req.filter.geo.lat, lon = req.filter.geo.lon;
  const double tau = std::max(0.1f, req.geo_tau_km);
  for (size_t s = 0; s < snap.segments.size(); ++s) {
    const auto& sv = snap.segments[s];
    const auto& seg = *sv.seg;
    const RoaringBitmap* F = fsets[s].get();
    if (!F) continue;  // geo path only runs under a geo filter
    F->for_each([&](uint32_t d) {
      if (sv.is_deleted(d)) return;
      double dist = geo::haversine_km(lat, lon, seg.lat(d), seg.lon(d));
      double pop = f_popularity_ >= 0 ? seg.numeric(f_popularity_, d) : 0.5;
      double rating = f_rating_ >= 0 ? seg.numeric(f_rating_, d) / 5.0 : 0.5;
      float score = static_cast<float>((0.7 * pop + 0.3 * rating) * std::exp(-dist / tau));
      ++out.scored;
      top.push({static_cast<uint32_t>(s), d, score});
    });
  }
  out.hits = top.sorted();
  out.matched = out.scored;
  out.time_us = sw.elapsed_us();
}

// ------------------------------------------------------------------ features

namespace {

// Smallest window (in positions) containing at least one position from every
// group present. `items` are (position, group) pairs.
uint32_t min_cover_window(std::vector<std::pair<uint32_t, int>>& items, int groups_present) {
  if (groups_present < 2 || items.empty()) return 0;
  std::sort(items.begin(), items.end());
  std::map<int, int> counts;
  int covered = 0;
  uint32_t best = UINT32_MAX;
  size_t lo = 0;
  for (size_t hi = 0; hi < items.size(); ++hi) {
    if (counts[items[hi].second]++ == 0) ++covered;
    while (covered == groups_present) {
      best = std::min(best, items[hi].first - items[lo].first + 1);
      if (--counts[items[lo].second] == 0) --covered;
      ++lo;
    }
  }
  return best == UINT32_MAX ? 0 : best;
}

}  // namespace

FeatureResponse Searcher::features(const index::Snapshot& snap, const FeatureRequest& req) {
  Stopwatch sw;
  FeatureResponse resp;
  const size_t T = schema_.text_fields.size();
  const size_t NN = schema_.numeric_fields.size();

  // ---- feature schema ----
  std::vector<int> prox_fields;
  for (size_t f = 0; f < T; ++f) resp.names.push_back("bm25_" + schema_.text_fields[f].name);
  resp.names.push_back("bm25_weighted");
  for (size_t f = 0; f < T; ++f)
    if (schema_.text_fields[f].positions && static_cast<int>(f) != f_name_exact_) {
      prox_fields.push_back(static_cast<int>(f));
      resp.names.push_back("proximity_" + schema_.text_fields[f].name);
    }
  for (const char* n : {"coverage", "name_coverage", "exact_name", "phrase_name", "name_len", "cosine"})
    resp.names.push_back(n);
  for (size_t f = 0; f < NN; ++f) resp.names.push_back(schema_.numeric_fields[f].name);
  for (const char* n : {"is_dish", "distance_km", "inside_zone", "open_now", "meal_breakfast", "meal_lunch",
                        "meal_snack", "meal_dinner", "meal_late_night"})
    resp.names.push_back(n);
  const size_t F = resp.names.size();
  const size_t idx_weighted = T;
  const size_t idx_prox = T + 1;
  const size_t idx_cov = idx_prox + prox_fields.size();
  const size_t idx_num = idx_cov + 6;
  const size_t idx_misc = idx_num + NN;

  const size_t D = req.docs.size();
  resp.rows.assign(D, std::vector<float>(F, 0.0f));
  resp.ids.resize(D);
  resp.stored.resize(D);

  // Unit field weights: per-field features carry only the query-term weight;
  // the retrieval field weights are applied for bm25_weighted alone.
  std::vector<Clause> clauses = compile(snap, req.terms, std::vector<float>(T, 1.0f), false, true);
  int num_groups = 0;
  for (const auto& qt : req.terms) num_groups = std::max(num_groups, qt.group + 1);
  // Primary tokens of the exact-match field, in query order.
  std::vector<std::string> exact_tokens;
  if (f_name_exact_ >= 0) {
    std::vector<std::pair<int, std::string>> tmp;
    const auto& an = analysis::get_analyzer(schema_.text_fields[f_name_exact_].analyzer);
    for (const auto& qt : req.terms)
      if (qt.primary)
        for (auto& t : an.terms(qt.text)) tmp.emplace_back(qt.group, t);
    std::stable_sort(tmp.begin(), tmp.end(), [](auto& a, auto& b) { return a.first < b.first; });
    for (auto& [g, t] : tmp) exact_tokens.push_back(t);
  }
  const float k1 = schema_.bm25.k1, b = schema_.bm25.b;
  std::vector<float> fw = req.field_weights;
  fw.resize(T, 0.0f);

  // Group request docs by segment, sorted by doc id (postings are forward-only).
  std::map<uint32_t, std::vector<size_t>> by_seg;
  for (size_t i = 0; i < D; ++i)
    if (req.docs[i].first < snap.segments.size()) by_seg[req.docs[i].first].push_back(i);

  std::vector<uint32_t> pos;
  for (auto& [s, idxs] : by_seg) {
    const auto& seg = *snap.segments[s].seg;
    std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t c) { return req.docs[a].second < req.docs[c].second; });
    const size_t n = idxs.size();
    // Per (doc, group) match flags across all fields and on the name field.
    std::vector<uint64_t> any_group(n, 0), name_group(n, 0);
    // Positions per (doc, proximity field): (pos, group).
    std::vector<std::vector<std::vector<std::pair<uint32_t, int>>>> prox(
        n, std::vector<std::vector<std::pair<uint32_t, int>>>(prox_fields.size()));
    std::vector<uint64_t> prox_groups(n * prox_fields.size(), 0);

    for (const auto& c : clauses) {
      auto ti = seg.term(c.field, c.term);
      if (!ti) continue;
      auto it = seg.postings(*ti);
      const uint16_t* norms = seg.norms(c.field);
      const float c1 = k1 * (1 - b), c2 = k1 * b / c.avgdl;
      int pidx = -1;
      for (size_t p = 0; p < prox_fields.size(); ++p)
        if (prox_fields[p] == c.field) pidx = static_cast<int>(p);
      for (size_t j = 0; j < n; ++j) {
        DocId d = req.docs[idxs[j]].second;
        if (it.advance(d) == kNoMoreDocs) break;
        if (it.doc() != d) continue;
        float tf = static_cast<float>(it.freq());
        float s1 = c.weight * c.idf * (k1 + 1) * bm25_tf(tf, norms[d], c1, c2);
        auto& row = resp.rows[idxs[j]];
        row[c.field] += s1;
        row[idx_weighted] += s1 * fw[c.field];
        if (c.group < 64) {
          any_group[j] |= 1ull << c.group;
          if (static_cast<int>(c.field) == f_name_) name_group[j] |= 1ull << c.group;
        }
        if (pidx >= 0 && c.group < 64) {
          it.positions(pos);
          for (uint32_t p : pos) prox[j][pidx].emplace_back(p, c.group);
          prox_groups[j * prox_fields.size() + pidx] |= 1ull << c.group;
        }
      }
    }

    // Exact / phrase match on the un-stemmed name field via positions.
    std::vector<std::vector<std::vector<uint32_t>>> exact_pos(n, std::vector<std::vector<uint32_t>>(exact_tokens.size()));
    for (size_t t = 0; t < exact_tokens.size(); ++t) {
      auto ti = seg.term(static_cast<uint8_t>(f_name_exact_), exact_tokens[t]);
      if (!ti) continue;
      auto it = seg.postings(*ti);
      for (size_t j = 0; j < n; ++j) {
        DocId d = req.docs[idxs[j]].second;
        if (it.advance(d) == kNoMoreDocs) break;
        if (it.doc() == d) it.positions(exact_pos[j][t]);
      }
    }

    for (size_t j = 0; j < n; ++j) {
      const size_t i = idxs[j];
      const DocId d = req.docs[i].second;
      auto& row = resp.rows[i];
      for (size_t p = 0; p < prox_fields.size(); ++p) {
        int present = __builtin_popcountll(prox_groups[j * prox_fields.size() + p]);
        uint32_t win = min_cover_window(prox[j][p], present);
        row[idx_prox + p] = win ? static_cast<float>(present) / static_cast<float>(win) : 0.0f;
      }
      float ng = static_cast<float>(std::max(num_groups, 1));
      row[idx_cov + 0] = static_cast<float>(__builtin_popcountll(any_group[j])) / ng;
      row[idx_cov + 1] = static_cast<float>(__builtin_popcountll(name_group[j])) / ng;
      // phrase: tokens appear consecutively; exact: phrase starting at 0 covering the whole name.
      bool phrase = false, exact = false;
      if (!exact_tokens.empty() && !exact_pos[j][0].empty()) {
        for (uint32_t p0 : exact_pos[j][0]) {
          bool ok = true;
          for (size_t t = 1; t < exact_tokens.size() && ok; ++t) {
            auto& v = exact_pos[j][t];
            ok = std::binary_search(v.begin(), v.end(), p0 + static_cast<uint32_t>(t));
          }
          if (ok) {
            phrase = true;
            if (p0 == 0 && seg.norm(static_cast<uint32_t>(f_name_exact_), d) == exact_tokens.size()) exact = true;
          }
        }
      }
      row[idx_cov + 2] = exact;
      row[idx_cov + 3] = phrase;
      row[idx_cov + 4] = f_name_ >= 0 ? seg.norm(static_cast<uint32_t>(f_name_), d) : 0;
      if (!req.qvec.empty() && seg.vector(d) && req.qvec.size() == schema_.vector_dim) {
        std::vector<float> q = req.qvec;
        vec::normalize(q.data(), q.size());
        row[idx_cov + 5] = vec::dot(q.data(), seg.vector(d), schema_.vector_dim);
      }
      for (size_t f = 0; f < NN; ++f) row[idx_num + f] = seg.numeric(static_cast<uint32_t>(f), d);
      row[idx_misc + 0] = seg.type_code(d) == index::kTypeDish;
      if (!std::isnan(req.user_lat)) {
        row[idx_misc + 1] = static_cast<float>(geo::haversine_km(req.user_lat, req.user_lon, seg.lat(d), seg.lon(d)));
        size_t nv = 0;
        const float* poly = seg.polygon(d, &nv);
        row[idx_misc + 2] = poly ? geo::point_in_polygon(req.user_lat, req.user_lon, poly, nv) : 0.0f;
      }
      row[idx_misc + 3] = req.slot >= 0 ? index::hours_open(seg.hours(d), req.slot) : 1.0f;
      const uint8_t* meal = seg.meal(d);
      for (size_t m = 0; m < index::kMealBuckets; ++m) row[idx_misc + 4 + m] = meal[m] / 255.0f;
      resp.ids[i] = seg.ext_id(d);
      if (req.include_stored) resp.stored[i] = seg.stored(d);
    }
  }
  resp.time_us = sw.elapsed_us();
  return resp;
}

}  // namespace dc::search
