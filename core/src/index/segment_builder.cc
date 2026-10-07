#include <algorithm>
#include <cmath>
#include <numeric>

#include "geo/geo.h"
#include "index/segment.h"
#include "util/log.h"

namespace dc::index {

SegmentBuilder::SegmentBuilder(const Schema& schema) : schema_(schema) {
  fields_.resize(schema.text_fields.size());
  keyword_docs_.resize(schema.keyword_fields.size());
  numerics_.resize(schema.numeric_fields.size());
}

DocId SegmentBuilder::add(const Document& doc) {
  const DocId d = n_++;

  // ---- inverted index ----
  for (size_t f = 0; f < schema_.text_fields.size(); ++f) {
    const auto& def = schema_.text_fields[f];
    const auto& an = analysis::get_analyzer(def.analyzer);
    tok_scratch_.clear();
    uint32_t pos = 0;
    for (const auto& value : doc.text[f]) pos = an.analyze(value, tok_scratch_, pos) + analysis::kMultiValueGap;
    FieldIndex& fi = fields_[f];
    uint32_t dl = static_cast<uint32_t>(std::min<size_t>(tok_scratch_.size(), 65535));
    fi.norms.push_back(static_cast<uint16_t>(dl));
    if (tok_scratch_.empty()) continue;
    fi.sum_dl += dl;
    ++fi.docs_with_field;
    // Group tokens by term (stable sort keeps positions ascending).
    std::stable_sort(tok_scratch_.begin(), tok_scratch_.end(),
                     [](const analysis::Token& a, const analysis::Token& b) { return a.term < b.term; });
    for (size_t i = 0; i < tok_scratch_.size();) {
      size_t j = i;
      while (j < tok_scratch_.size() && tok_scratch_[j].term == tok_scratch_[i].term) ++j;
      auto [it, inserted] = fi.term_ids.try_emplace(tok_scratch_[i].term, static_cast<uint32_t>(fi.postings.size()));
      if (inserted) {
        fi.postings.emplace_back();
        fi.terms.push_back(tok_scratch_[i].term);
        ram_bytes_ += tok_scratch_[i].term.size() * 2 + 96;
      }
      PostingsBuffer& pb = fi.postings[it->second];
      uint32_t tf = static_cast<uint32_t>(j - i);
      pb.docs.push_back(d);
      pb.freqs.push_back(tf);
      pb.max_tf = std::max(pb.max_tf, tf);
      pb.min_dl = std::min(pb.min_dl, dl);
      ram_bytes_ += 8;
      if (def.positions) {
        for (size_t k = i; k < j; ++k) pb.positions.push_back(tok_scratch_[k].position);
        ram_bytes_ += 4 * tf;
      }
      i = j;
    }
  }

  // ---- filters ----
  for (size_t f = 0; f < schema_.keyword_fields.size(); ++f)
    for (const auto& v : doc.keywords[f]) {
      keyword_docs_[f][v].push_back(d);
      ram_bytes_ += 4;
    }
  if (!std::isnan(doc.lat)) {
    std::vector<float> zone = doc.polygon;
    if (zone.size() < 6) {
      int rf = schema_.numeric_field("delivery_radius_km");
      double radius = rf >= 0 ? doc.numerics[rf] : 5.0;
      zone = geo::circle_polygon(doc.lat, doc.lon, radius > 0 ? radius : 5.0);
    }
    for (uint64_t c : geo::polygon_cover(zone.data(), zone.size() / 2, schema_.geo.cover_res, doc.lat, doc.lon))
      cover_docs_[c].push_back(d);
    uint64_t pc = geo::cell_of(doc.lat, doc.lon, schema_.geo.point_res);
    if (pc) point_docs_[pc].push_back(d);
    ram_bytes_ += 64;
  }

  // ---- columns ----
  for (size_t f = 0; f < schema_.numeric_fields.size(); ++f) numerics_[f].push_back(doc.numerics[f]);
  latlon_.push_back(doc.lat);
  latlon_.push_back(doc.lon);
  hours_.insert(hours_.end(), doc.hours.begin(), doc.hours.end());
  meal_.insert(meal_.end(), doc.meal.begin(), doc.meal.end());
  types_.push_back(doc.type_code);
  stored_.push_back(doc.stored);
  ids_.push_back(doc.id);
  polygons_.push_back(doc.polygon);
  ram_bytes_ += 64 + doc.stored.size() + doc.id.size() + doc.polygon.size() * 4 + 4 * schema_.numeric_fields.size();
  if (schema_.vector_dim > 0) {
    if (doc.vec.empty()) {
      vectors_.resize(vectors_.size() + schema_.vector_dim, 0.0f);
    } else {
      vectors_.insert(vectors_.end(), doc.vec.begin(), doc.vec.end());
      any_vector_ = true;
    }
    ram_bytes_ += 4 * schema_.vector_dim;
  }
  return d;
}

SegmentMeta SegmentBuilder::write(SegmentStorage& out, const SegmentWriteOptions& opt) {
  SegmentMeta meta;
  meta.num_docs = n_;

  // ---- postings + term dictionary (keys sorted by field id, then term bytes) ----
  {
    auto post = out.create("postings.bin");
    auto pos = out.create("positions.bin");
    TermDictWriter dict(out.create("terms.dict"));
    PostingsWriter pw(post.get(), pos.get(), opt.codec);
    for (size_t f = 0; f < fields_.size(); ++f) {
      FieldIndex& fi = fields_[f];
      std::vector<uint32_t> order(fi.terms.size());
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return fi.terms[a] < fi.terms[b]; });
      for (uint32_t t : order) {
        TermInfo ti = pw.write(fi.postings[t], schema_.text_fields[f].positions);
        dict.add(make_key(static_cast<uint8_t>(f), fi.terms[t]), ti);
      }
      meta.field_sum_dl.push_back(fi.sum_dl);
      meta.field_docs.push_back(fi.docs_with_field);
    }
    meta.num_terms = dict.num_terms();
    dict.finish();
    post->close();
    pos->close();
  }

  // ---- filter bitmaps ----
  {
    auto data = out.create("filters.bin");
    TermDictWriter dict(out.create("filters.dict"));
    ByteWriter w;
    auto emit = [&](const std::string& key, const std::vector<uint32_t>& docs) {
      RoaringBitmap bm = RoaringBitmap::from_sorted(docs.data(), docs.size());
      bm.run_optimize();
      w.clear();
      bm.serialize(w);
      TermInfo ti;
      ti.df = static_cast<uint32_t>(docs.size());
      ti.post_off = data->offset();
      ti.post_len = w.size();
      data->write(w.data(), w.size());
      dict.add(key, ti);
    };
    for (size_t f = 0; f < keyword_docs_.size(); ++f) {
      std::vector<const std::string*> vals;
      for (auto& [v, _] : keyword_docs_[f]) vals.push_back(&v);
      std::sort(vals.begin(), vals.end(), [](auto* a, auto* b) { return *a < *b; });
      for (auto* v : vals) emit(make_key(static_cast<uint8_t>(f), *v), keyword_docs_[f][*v]);
    }
    for (auto* m : {&cover_docs_, &point_docs_}) {
      uint8_t field = m == &cover_docs_ ? kGeoCoverField : kGeoPointField;
      std::vector<uint64_t> cells;
      for (auto& [c, _] : *m) cells.push_back(c);
      std::sort(cells.begin(), cells.end());
      for (uint64_t c : cells) emit(make_cell_key(field, c), (*m)[c]);
    }
    meta.num_filter_keys = dict.num_terms();
    dict.finish();
    data->close();
  }

  // ---- columns ----
  meta.has_vectors = schema_.vector_dim > 0 && any_vector_;
  {
    ColumnsWriter cw(out, schema_, meta.has_vectors);
    std::vector<uint16_t> norms(fields_.size());
    std::vector<float> nums(numerics_.size());
    for (uint32_t d = 0; d < n_; ++d) {
      for (size_t f = 0; f < fields_.size(); ++f) norms[f] = fields_[f].norms[d];
      for (size_t f = 0; f < numerics_.size(); ++f) nums[f] = numerics_[f][d];
      ColumnsWriter::Row r{norms.data(),
                           nums.data(),
                           latlon_[2 * d],
                           latlon_[2 * d + 1],
                           hours_.data() + kHoursBytes * d,
                           meal_.data() + kMealBuckets * d,
                           types_[d],
                           stored_[d],
                           ids_[d],
                           polygons_[d].data(),
                           polygons_[d].size() / 2,
                           meta.has_vectors ? vectors_.data() + static_cast<size_t>(d) * schema_.vector_dim
                                            : nullptr};
      cw.add(r);
    }
    cw.finish();
  }
  finalize_segment(out, schema_, meta, opt);
  return meta;
}

}  // namespace dc::index
