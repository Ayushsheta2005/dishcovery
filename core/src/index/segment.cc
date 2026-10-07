#include "index/segment.h"

#include <algorithm>
#include <cstring>

#include "util/hash.h"
#include "util/log.h"
#include "util/thread_pool.h"
#include "util/timer.h"

namespace dc::index {

// ---------------------------------------------------------------- meta

json SegmentMeta::to_json() const {
  return json{{"uid", uid},
              {"num_docs", num_docs},
              {"codec", codec::codec_name(codec)},
              {"vector_dim", vector_dim},
              {"has_vectors", has_vectors},
              {"has_hnsw", has_hnsw},
              {"field_sum_dl", field_sum_dl},
              {"field_docs", field_docs},
              {"num_terms", num_terms},
              {"num_filter_keys", num_filter_keys},
              {"created_ms", created_ms},
              {"schema_fp", schema_fp},
              {"size_bytes", size_bytes},
              {"source", source}};
}

SegmentMeta SegmentMeta::from_json(const json& j) {
  SegmentMeta m;
  m.uid = j.at("uid").get<std::string>();
  m.num_docs = j.at("num_docs").get<uint32_t>();
  m.codec = codec::codec_from_name(j.at("codec").get<std::string>());
  m.vector_dim = j.value("vector_dim", 0u);
  m.has_vectors = j.value("has_vectors", false);
  m.has_hnsw = j.value("has_hnsw", false);
  m.field_sum_dl = j.at("field_sum_dl").get<std::vector<uint64_t>>();
  m.field_docs = j.at("field_docs").get<std::vector<uint32_t>>();
  m.num_terms = j.value("num_terms", 0ull);
  m.num_filter_keys = j.value("num_filter_keys", 0ull);
  m.created_ms = j.value("created_ms", 0ull);
  m.schema_fp = j.value("schema_fp", 0ull);
  m.size_bytes = j.value("size_bytes", 0ull);
  m.source = j.value("source", std::string());
  return m;
}

// ---------------------------------------------------------------- reader

SegmentReader::VarColumn SegmentReader::open_var(Span s) {
  VarColumn c;
  if (s.size < 8) return c;
  uint64_t n = load_u64(s.data + s.size - 8);
  DC_CHECK((n + 1) * 8 + 8 <= s.size, "var column corrupt");
  c.n = static_cast<uint32_t>(n);
  c.offsets = s.data + s.size - 8 - (n + 1) * 8;
  c.data = s.data;
  return c;
}

std::string_view SegmentReader::var_get(const VarColumn& c, DocId d) {
  if (!c.offsets || d >= c.n) return {};
  uint64_t a = load_u64(c.offsets + 8ull * d), b = load_u64(c.offsets + 8ull * (d + 1));
  return {reinterpret_cast<const char*>(c.data + a), static_cast<size_t>(b - a)};
}

std::shared_ptr<SegmentReader> SegmentReader::open(std::shared_ptr<SegmentStorage> storage, const Schema& schema) {
  std::shared_ptr<SegmentReader> r(new SegmentReader(std::move(storage), schema));
  SegmentStorage& st = *r->storage_;
  Span meta_span = st.open("meta.json");
  DC_CHECK(!meta_span.empty(), "segment has no meta.json at " + st.location());
  r->meta_ = SegmentMeta::from_json(
      json::parse(std::string_view(reinterpret_cast<const char*>(meta_span.data), meta_span.size)));
  DC_CHECK(r->meta_.schema_fp == schema.fingerprint(),
           "segment " + r->meta_.uid + " was built with a different schema");
  const uint32_t n = r->meta_.num_docs;

  r->terms_ = TermDictReader(st.open("terms.dict"));
  r->postings_ = st.open("postings.bin");
  r->positions_ = st.open("positions.bin");
  r->filters_ = TermDictReader(st.open("filters.dict"));
  r->filters_data_ = st.open("filters.bin");

  auto need = [&](const std::string& name, size_t bytes) -> const uint8_t* {
    Span s = st.open(name);
    if (n == 0) return s.data;
    DC_CHECK(s.size >= bytes, name + " truncated in segment " + r->meta_.uid);
    return s.data;
  };
  for (size_t f = 0; f < schema.text_fields.size(); ++f)
    r->norms_.push_back(reinterpret_cast<const uint16_t*>(need("norms_" + std::to_string(f) + ".bin", 2ull * n)));
  for (size_t f = 0; f < schema.numeric_fields.size(); ++f)
    r->numerics_.push_back(reinterpret_cast<const float*>(need("numeric_" + std::to_string(f) + ".bin", 4ull * n)));
  r->latlon_ = reinterpret_cast<const float*>(need("latlon.bin", 8ull * n));
  r->hours_ = need("hours.bin", kHoursBytes * n);
  r->meal_ = need("meal.bin", kMealBuckets * n);
  r->types_ = need("type.bin", n);
  r->stored_ = open_var(st.open("stored.bin"));
  r->ids_ = open_var(st.open("ids.bin"));
  r->polygons_ = open_var(st.open("polygons.bin"));
  r->idmap_ = need("idmap.bin", 12ull * n);
  if (r->meta_.has_vectors) {
    r->vectors_ = reinterpret_cast<const float*>(need("vectors.bin", 4ull * n * r->meta_.vector_dim));
    if (r->meta_.has_hnsw) {
      Span h = st.open("hnsw.bin");
      r->hnsw_ = std::make_unique<vec::HnswIndex>(
          vec::HnswIndex::load(h.data, h.size, r->vectors_, n, r->meta_.vector_dim));
    }
  }
  return r;
}

SegmentReader::~SegmentReader() {
  if (obsolete_.load()) {
    try {
      storage_->destroy();
    } catch (...) {
    }
  }
}

RoaringBitmap SegmentReader::filter_bitmap(const TermInfo& ti) const {
  DC_CHECK(ti.post_off + ti.post_len <= filters_data_.size, "filter bitmap out of range");
  ByteReader r(filters_data_.data + ti.post_off, ti.post_len);
  return RoaringBitmap::deserialize(r);
}

const float* SegmentReader::polygon(DocId d, size_t* num_vertices) const {
  auto s = var_get(polygons_, d);
  *num_vertices = s.size() / 8;
  return s.empty() ? nullptr : reinterpret_cast<const float*>(s.data());
}

std::optional<DocId> SegmentReader::find(std::string_view ext_id) const {
  const uint32_t n = meta_.num_docs;
  if (!idmap_ || n == 0) return std::nullopt;
  uint64_t h = fnv1a64(ext_id);
  // Binary search for the first entry with hash >= h; collisions are resolved
  // by comparing the stored id.
  uint32_t lo = 0, hi = n;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if (load_u64(idmap_ + 12ull * mid) < h) lo = mid + 1;
    else hi = mid;
  }
  for (uint32_t i = lo; i < n && load_u64(idmap_ + 12ull * i) == h; ++i) {
    DocId d = load_u32(idmap_ + 12ull * i + 8);
    if (this->ext_id(d) == ext_id) return d;
  }
  return std::nullopt;
}

// ---------------------------------------------------------------- columns writer

void ColumnsWriter::VarWriter::add(const void* p, size_t len) {
  if (len) out->write(p, len);
  offsets.push_back(offsets.back() + len);
}

void ColumnsWriter::VarWriter::finish() {
  out->align_to(8);
  // Offsets are relative to the file start; padding only follows the data.
  out->write(offsets.data(), offsets.size() * 8);
  uint64_t n = offsets.size() - 1;
  out->write(&n, 8);
  out->close();
}

ColumnsWriter::ColumnsWriter(SegmentStorage& out, const Schema& schema, bool with_vectors)
    : schema_(schema), with_vectors_(with_vectors), storage_(out) {
  for (size_t f = 0; f < schema.text_fields.size(); ++f)
    norms_.push_back(out.create("norms_" + std::to_string(f) + ".bin"));
  for (size_t f = 0; f < schema.numeric_fields.size(); ++f)
    numerics_.push_back(out.create("numeric_" + std::to_string(f) + ".bin"));
  latlon_ = out.create("latlon.bin");
  hours_ = out.create("hours.bin");
  meal_ = out.create("meal.bin");
  types_ = out.create("type.bin");
  stored_.out = out.create("stored.bin");
  ids_.out = out.create("ids.bin");
  polygons_.out = out.create("polygons.bin");
  if (with_vectors_) {
    vectors_ = out.create("vectors.bin");
    zero_vec_.assign(schema.vector_dim, 0.0f);
  }
}

void ColumnsWriter::add(const Row& r) {
  for (size_t f = 0; f < norms_.size(); ++f) norms_[f]->write(&r.norms[f], 2);
  for (size_t f = 0; f < numerics_.size(); ++f) numerics_[f]->write(&r.numerics[f], 4);
  float ll[2] = {r.lat, r.lon};
  latlon_->write(ll, 8);
  hours_->write(r.hours, kHoursBytes);
  meal_->write(r.meal, kMealBuckets);
  types_->write(&r.type_code, 1);
  stored_.add(r.stored.data(), r.stored.size());
  ids_.add(r.ext_id.data(), r.ext_id.size());
  polygons_.add(r.polygon, r.polygon_vertices * 8);
  if (with_vectors_) vectors_->write(r.vec ? r.vec : zero_vec_.data(), 4ull * schema_.vector_dim);
  idmap_.emplace_back(fnv1a64(r.ext_id), n_);
  ++n_;
}

void ColumnsWriter::finish() {
  for (auto& s : norms_) s->close();
  for (auto& s : numerics_) s->close();
  latlon_->close();
  hours_->close();
  meal_->close();
  types_->close();
  stored_.finish();
  ids_.finish();
  polygons_.finish();
  if (vectors_) vectors_->close();
  std::sort(idmap_.begin(), idmap_.end());
  auto out = storage_.create("idmap.bin");
  ByteWriter w(idmap_.size() * 12);
  for (auto& [h, d] : idmap_) {
    w.put_u64(h);
    w.put_u32(d);
  }
  out->write(w.data(), w.size());
  out->close();
}

void finalize_segment(SegmentStorage& out, const Schema& schema, SegmentMeta& meta, const SegmentWriteOptions& opt) {
  meta.uid = opt.uid;
  meta.codec = opt.codec;
  meta.schema_fp = schema.fingerprint();
  meta.created_ms = wall_ms();
  meta.source = opt.source;
  meta.vector_dim = schema.vector_dim;
  meta.has_hnsw = false;
  if (meta.has_vectors && opt.build_hnsw && meta.num_docs >= opt.hnsw_min_docs) {
    Span vs = out.open("vectors.bin");
    DC_CHECK(vs.size == 4ull * meta.num_docs * schema.vector_dim, "vectors.bin size mismatch");
    Stopwatch sw;
    vec::HnswIndex h(reinterpret_cast<const float*>(vs.data), meta.num_docs, schema.vector_dim, opt.hnsw);
    h.build(opt.threads);
    ByteWriter w;
    h.serialize(w);
    auto o = out.create("hnsw.bin");
    o->write(w.data(), w.size());
    o->close();
    meta.has_hnsw = true;
    LOG_INFO("segment " << opt.uid << ": HNSW over " << meta.num_docs << " vectors built in " << sw.elapsed_s()
                        << " s");
  }
  meta.size_bytes = out.total_bytes();
  std::string m = meta.to_json().dump(2);
  auto o = out.create("meta.json");
  o->write(m.data(), m.size());
  o->close();
  meta.size_bytes = out.total_bytes();
}

}  // namespace dc::index
