// Immutable index segments.
//
// A segment is a directory (or, for near-real-time RAM segments, an in-memory
// file map) of independently mmap'd files:
//
//   meta.json            doc count, codec, per-field length stats
//   terms.dict           front-coded term dictionary  -> postings.bin / positions.bin
//   filters.dict         (field, value) and H3-cell keys -> filters.bin (Roaring bitmaps)
//   norms_<f>.bin        u16 field length per doc, one file per text field
//   numeric_<f>.bin      f32 doc values, one file per numeric field
//   latlon.bin hours.bin meal.bin type.bin       built-in fixed-width columns
//   stored.bin ids.bin polygons.bin              variable-length, offsets at the end
//   idmap.bin            sorted (hash(id), doc) for delete-by-id
//   vectors.bin hnsw.bin unit embeddings + HNSW graph (level 0 used in place)
//
// Segments never change after being written; deletes are tombstone bitmaps
// owned by the shard's snapshot, not by the segment.
#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "analysis/analyzer.h"
#include "codec/block_codec.h"
#include "codec/roaring.h"
#include "index/document.h"
#include "index/postings.h"
#include "index/schema.h"
#include "index/term_dict.h"
#include "util/fs.h"
#include "util/json.h"
#include "vector/hnsw.h"

namespace dc::index {

struct SegmentMeta {
  std::string uid;
  uint32_t num_docs = 0;
  codec::CodecId codec = codec::CodecId::kPFor;
  uint32_t vector_dim = 0;
  bool has_vectors = false;
  bool has_hnsw = false;
  std::vector<uint64_t> field_sum_dl;   // per text field
  std::vector<uint32_t> field_docs;     // docs with >= 1 token in the field
  uint64_t num_terms = 0;
  uint64_t num_filter_keys = 0;
  uint64_t created_ms = 0;
  uint64_t schema_fp = 0;
  uint64_t size_bytes = 0;
  std::string source;                   // build | refresh | flush | merge
  json to_json() const;
  static SegmentMeta from_json(const json& j);
};

struct SegmentWriteOptions {
  std::string uid;
  codec::CodecId codec = codec::CodecId::kPFor;
  bool build_hnsw = true;
  uint32_t hnsw_min_docs = 2000;   // below this, vector search is a brute-force scan
  vec::HnswParams hnsw;
  unsigned threads = 1;
  std::string source = "build";
};

class SegmentReader {
 public:
  static std::shared_ptr<SegmentReader> open(std::shared_ptr<SegmentStorage> storage, const Schema& schema);
  ~SegmentReader();

  const SegmentMeta& meta() const { return meta_; }
  const std::string& uid() const { return meta_.uid; }
  uint32_t num_docs() const { return meta_.num_docs; }
  uint64_t size_bytes() const { return meta_.size_bytes; }
  bool on_disk() const { return storage_->on_disk(); }
  const Schema& schema() const { return schema_; }

  // --- inverted index ---
  std::optional<TermInfo> term(uint8_t field, std::string_view t) const { return terms_.lookup(make_key(field, t)); }
  PostingsIterator postings(const TermInfo& ti) const {
    return PostingsIterator(postings_, positions_, ti, meta_.codec);
  }
  const TermDictReader& terms() const { return terms_; }
  Span postings_span() const { return postings_; }
  Span positions_span() const { return positions_; }
  const uint16_t* norms(uint32_t field) const { return norms_[field]; }
  uint16_t norm(uint32_t field, DocId d) const { return norms_[field] ? norms_[field][d] : 0; }
  double avg_dl(uint32_t field) const {
    return meta_.num_docs ? static_cast<double>(meta_.field_sum_dl[field]) / meta_.num_docs : 0.0;
  }

  // --- filters (Roaring) ---
  const TermDictReader& filters() const { return filters_; }
  std::optional<TermInfo> filter_entry(std::string_view key) const { return filters_.lookup(key); }
  RoaringBitmap filter_bitmap(const TermInfo& ti) const;

  // --- columns ---
  float numeric(uint32_t field, DocId d) const { return numerics_[field][d]; }
  const float* numeric_column(uint32_t field) const { return numerics_[field]; }
  float lat(DocId d) const { return latlon_[2 * d]; }
  float lon(DocId d) const { return latlon_[2 * d + 1]; }
  const uint8_t* hours(DocId d) const { return hours_ + static_cast<size_t>(d) * kHoursBytes; }
  const uint8_t* meal(DocId d) const { return meal_ + static_cast<size_t>(d) * kMealBuckets; }
  uint8_t type_code(DocId d) const { return types_[d]; }
  std::string_view stored(DocId d) const { return var_get(stored_, d); }
  std::string_view ext_id(DocId d) const { return var_get(ids_, d); }
  const float* polygon(DocId d, size_t* num_vertices) const;
  const float* vector(DocId d) const {
    return vectors_ ? vectors_ + static_cast<size_t>(d) * meta_.vector_dim : nullptr;
  }
  const float* vectors() const { return vectors_; }
  const vec::HnswIndex* hnsw() const { return hnsw_.get(); }
  std::optional<DocId> find(std::string_view ext_id) const;

  // Obsolete segments delete their files when the last reference goes away.
  void mark_obsolete() { obsolete_.store(true); }
  SegmentStorage& storage() const { return *storage_; }

 private:
  struct VarColumn {
    const uint8_t* data = nullptr;
    const uint8_t* offsets = nullptr;  // (N+1) u64
    uint32_t n = 0;
  };
  static VarColumn open_var(Span s);
  static std::string_view var_get(const VarColumn& c, DocId d);

  SegmentReader(std::shared_ptr<SegmentStorage> storage, const Schema& schema)
      : storage_(std::move(storage)), schema_(schema) {}

  std::shared_ptr<SegmentStorage> storage_;
  Schema schema_;
  SegmentMeta meta_;
  TermDictReader terms_;
  TermDictReader filters_;
  Span postings_, positions_, filters_data_;
  std::vector<const uint16_t*> norms_;
  std::vector<const float*> numerics_;
  const float* latlon_ = nullptr;
  const uint8_t* hours_ = nullptr;
  const uint8_t* meal_ = nullptr;
  const uint8_t* types_ = nullptr;
  VarColumn stored_, ids_, polygons_;
  const uint8_t* idmap_ = nullptr;
  const float* vectors_ = nullptr;
  std::unique_ptr<vec::HnswIndex> hnsw_;
  std::atomic<bool> obsolete_{false};
};

// Streams the per-document columns of a segment being written (used by both
// the in-memory builder and the merger).
class ColumnsWriter {
 public:
  ColumnsWriter(SegmentStorage& out, const Schema& schema, bool with_vectors);
  struct Row {
    const uint16_t* norms;      // [text fields]
    const float* numerics;      // [numeric fields]
    float lat, lon;
    const uint8_t* hours;       // kHoursBytes
    const uint8_t* meal;        // kMealBuckets
    uint8_t type_code;
    std::string_view stored;
    std::string_view ext_id;
    const float* polygon;       // interleaved lat/lon
    size_t polygon_vertices;
    const float* vec;           // nullptr -> zero vector
  };
  void add(const Row& r);
  void finish();
  uint32_t num_docs() const { return n_; }

 private:
  struct VarWriter {
    std::unique_ptr<OutputStream> out;
    std::vector<uint64_t> offsets{0};
    void add(const void* p, size_t len);
    void finish();
  };
  const Schema& schema_;
  bool with_vectors_;
  std::vector<std::unique_ptr<OutputStream>> norms_, numerics_;
  std::unique_ptr<OutputStream> latlon_, hours_, meal_, types_, vectors_;
  VarWriter stored_, ids_, polygons_;
  std::vector<std::pair<uint64_t, uint32_t>> idmap_;
  SegmentStorage& storage_;
  uint32_t n_ = 0;
  std::vector<float> zero_vec_;
};

// Builds the HNSW graph (if requested) over vectors.bin and writes meta.json.
// Shared tail of SegmentBuilder::write and merge_segments.
void finalize_segment(SegmentStorage& out, const Schema& schema, SegmentMeta& meta, const SegmentWriteOptions& opt);

// In-memory accumulation of new documents into one segment.
class SegmentBuilder {
 public:
  explicit SegmentBuilder(const Schema& schema);
  DocId add(const Document& doc);
  uint32_t num_docs() const { return n_; }
  size_t ram_bytes() const { return ram_bytes_; }
  const std::string& ext_id(DocId d) const { return ids_[d]; }
  SegmentMeta write(SegmentStorage& out, const SegmentWriteOptions& opt);

 private:
  struct FieldIndex {
    std::unordered_map<std::string, uint32_t> term_ids;
    std::vector<std::string> terms;
    std::vector<PostingsBuffer> postings;
    std::vector<uint16_t> norms;
    uint64_t sum_dl = 0;
    uint32_t docs_with_field = 0;
  };
  const Schema& schema_;
  uint32_t n_ = 0;
  size_t ram_bytes_ = 0;
  std::vector<FieldIndex> fields_;
  std::vector<std::unordered_map<std::string, std::vector<uint32_t>>> keyword_docs_;
  std::unordered_map<uint64_t, std::vector<uint32_t>> cover_docs_, point_docs_;
  std::vector<std::vector<float>> numerics_;  // [field][doc]
  std::vector<float> latlon_;
  std::vector<uint8_t> hours_, meal_, types_;
  std::vector<std::string> stored_, ids_;
  std::vector<std::vector<float>> polygons_;
  std::vector<float> vectors_;
  bool any_vector_ = false;
  std::vector<analysis::Token> tok_scratch_;
};

struct MergeSource {
  std::shared_ptr<const SegmentReader> segment;
  std::shared_ptr<const RoaringBitmap> deletes;  // may be null
};

struct MergeResult {
  SegmentMeta meta;
  // doc_maps[i][old_doc] = new doc id, or -1 if the doc was deleted.
  std::vector<std::vector<int32_t>> doc_maps;
  uint64_t bytes_read = 0;
  uint64_t bytes_written = 0;
};

// Merge segments by streaming postings directly (no re-analysis): k-way merge
// of term dictionaries, docID remapping around deletes, bitmap unions for
// filters, column concatenation, and an HNSW rebuild over the merged vectors.
MergeResult merge_segments(const std::vector<MergeSource>& sources, const Schema& schema, SegmentStorage& out,
                           const SegmentWriteOptions& opt);

}  // namespace dc::index
