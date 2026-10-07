// Segment merging by direct postings merge.
//
// Re-analysing documents would be simpler but burns CPU on tokenisation and
// stemming; instead we stream each source's term dictionary in key order
// (k-way merge), decode postings, drop deleted docs, remap doc ids and
// re-encode. Merge cost is then dominated by decode/encode and I/O, which is
// what the merge-policy benchmarks measure.
#include <algorithm>
#include <queue>

#include "index/segment.h"
#include "util/log.h"
#include "util/timer.h"

namespace dc::index {

namespace {

// Min-heap over term-dictionary iterators keyed by current key.
struct DictCursor {
  TermDictReader::Iterator it;
  size_t source;
};

struct CursorGreater {
  bool operator()(const DictCursor* a, const DictCursor* b) const {
    int c = a->it.key().compare(b->it.key());
    return c > 0 || (c == 0 && a->source > b->source);
  }
};

using CursorHeap = std::priority_queue<DictCursor*, std::vector<DictCursor*>, CursorGreater>;

}  // namespace

MergeResult merge_segments(const std::vector<MergeSource>& sources, const Schema& schema, SegmentStorage& out,
                           const SegmentWriteOptions& opt) {
  Stopwatch sw;
  MergeResult res;
  const size_t S = sources.size();
  DC_CHECK(S > 0, "merge with no sources");

  // ---- doc maps ----
  res.doc_maps.resize(S);
  uint32_t next_doc = 0;
  for (size_t s = 0; s < S; ++s) {
    const auto& seg = *sources[s].segment;
    auto& map = res.doc_maps[s];
    map.assign(seg.num_docs(), -1);
    const RoaringBitmap* del = sources[s].deletes.get();
    for (DocId d = 0; d < seg.num_docs(); ++d)
      if (!del || !del->contains(d)) map[d] = static_cast<int32_t>(next_doc++);
    res.bytes_read += seg.size_bytes();
  }
  SegmentMeta meta;
  meta.num_docs = next_doc;
  const size_t T = schema.text_fields.size();
  meta.field_sum_dl.assign(T, 0);
  meta.field_docs.assign(T, 0);

  // ---- postings ----
  {
    auto post = out.create("postings.bin");
    auto pos = out.create("positions.bin");
    TermDictWriter dict(out.create("terms.dict"));
    PostingsWriter pw(post.get(), pos.get(), opt.codec);
    std::vector<DictCursor> cursors;
    cursors.reserve(S);
    for (size_t s = 0; s < S; ++s) cursors.push_back({sources[s].segment->terms().iterate(), s});
    CursorHeap heap;
    for (auto& c : cursors)
      if (c.it.valid()) heap.push(&c);
    PostingsBuffer buf;
    std::vector<uint32_t> positions;
    std::vector<DictCursor*> same;
    while (!heap.empty()) {
      std::string key = heap.top()->it.key();
      same.clear();
      while (!heap.empty() && heap.top()->it.key() == key) {
        same.push_back(heap.top());
        heap.pop();
      }
      std::sort(same.begin(), same.end(), [](auto* a, auto* b) { return a->source < b->source; });
      uint8_t field = static_cast<uint8_t>(key[0]);
      DC_CHECK(field < T, "term dict key with unknown field");
      bool with_pos = schema.text_fields[field].positions;
      buf.clear();
      for (DictCursor* c : same) {
        const auto& seg = *sources[c->source].segment;
        const auto& map = res.doc_maps[c->source];
        const uint16_t* norms = seg.norms(field);
        PostingsIterator it = seg.postings(c->it.info());
        for (DocId d = it.next(); d != kNoMoreDocs; d = it.next()) {
          int32_t nd = map[d];
          if (nd < 0) continue;
          uint32_t tf = it.freq();
          buf.docs.push_back(static_cast<uint32_t>(nd));
          buf.freqs.push_back(tf);
          buf.max_tf = std::max(buf.max_tf, tf);
          buf.min_dl = std::min<uint32_t>(buf.min_dl, norms[d]);
          if (with_pos) {
            it.positions(positions);
            buf.positions.insert(buf.positions.end(), positions.begin(), positions.end());
          }
        }
      }
      if (!buf.docs.empty()) dict.add(key, pw.write(buf, with_pos));
      for (DictCursor* c : same) {
        c->it.next();
        if (c->it.valid()) heap.push(c);
      }
    }
    meta.num_terms = dict.num_terms();
    dict.finish();
    post->close();
    pos->close();
  }

  // ---- filters ----
  {
    auto data = out.create("filters.bin");
    TermDictWriter dict(out.create("filters.dict"));
    std::vector<DictCursor> cursors;
    for (size_t s = 0; s < S; ++s) cursors.push_back({sources[s].segment->filters().iterate(), s});
    CursorHeap heap;
    for (auto& c : cursors)
      if (c.it.valid()) heap.push(&c);
    std::vector<uint32_t> docs;
    std::vector<DictCursor*> same;
    ByteWriter w;
    while (!heap.empty()) {
      std::string key = heap.top()->it.key();
      same.clear();
      while (!heap.empty() && heap.top()->it.key() == key) {
        same.push_back(heap.top());
        heap.pop();
      }
      std::sort(same.begin(), same.end(), [](auto* a, auto* b) { return a->source < b->source; });
      docs.clear();
      for (DictCursor* c : same) {
        const auto& map = res.doc_maps[c->source];
        RoaringBitmap bm = sources[c->source].segment->filter_bitmap(c->it.info());
        bm.for_each([&](uint32_t d) {
          if (map[d] >= 0) docs.push_back(static_cast<uint32_t>(map[d]));
        });
      }
      if (!docs.empty()) {
        // Sources are concatenated in order and maps are monotone, so `docs` is sorted.
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
      }
      for (DictCursor* c : same) {
        c->it.next();
        if (c->it.valid()) heap.push(c);
      }
    }
    meta.num_filter_keys = dict.num_terms();
    dict.finish();
    data->close();
  }

  // ---- columns ----
  bool any_vectors = false;
  for (auto& s : sources) any_vectors |= s.segment->meta().has_vectors;
  meta.has_vectors = any_vectors && schema.vector_dim > 0;
  {
    ColumnsWriter cw(out, schema, meta.has_vectors);
    std::vector<uint16_t> norms(T);
    std::vector<float> nums(schema.numeric_fields.size());
    for (size_t s = 0; s < S; ++s) {
      const auto& seg = *sources[s].segment;
      const auto& map = res.doc_maps[s];
      for (DocId d = 0; d < seg.num_docs(); ++d) {
        if (map[d] < 0) continue;
        for (size_t f = 0; f < T; ++f) {
          norms[f] = seg.norm(static_cast<uint32_t>(f), d);
          meta.field_sum_dl[f] += norms[f];
          if (norms[f]) ++meta.field_docs[f];
        }
        for (size_t f = 0; f < nums.size(); ++f) nums[f] = seg.numeric(static_cast<uint32_t>(f), d);
        size_t nv = 0;
        const float* poly = seg.polygon(d, &nv);
        ColumnsWriter::Row r{norms.data(), nums.data(), seg.lat(d),  seg.lon(d),   seg.hours(d),
                             seg.meal(d),  seg.type_code(d), seg.stored(d), seg.ext_id(d), poly,
                             nv,           meta.has_vectors ? seg.vector(d) : nullptr};
        cw.add(r);
      }
    }
    cw.finish();
  }

  finalize_segment(out, schema, meta, opt);
  res.meta = meta;
  res.bytes_written = meta.size_bytes;
  LOG_INFO("merged " << S << " segments -> " << meta.uid << " (" << meta.num_docs << " docs, "
                     << meta.size_bytes / (1 << 20) << " MiB) in " << sw.elapsed_s() << " s");
  return res;
}

}  // namespace dc::index
