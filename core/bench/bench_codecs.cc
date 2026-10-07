// bench_codecs: compression schemes on the real postings of a built shard.
//
//   bench_codecs --index work/yelp/index/shard_0 [--schema configs/schema.json]
//                [--min-df 64] [--max-terms 20000] [--pairs 2000]
//
// For every sampled term the doc-id list is re-encoded with raw / vbyte / pfor
// (128-int delta blocks, as in the index) and as a Roaring bitmap; we report
// bits per posting and decode throughput. Then AND-intersection of random
// term pairs is timed three ways: decode-everything + linear merge, the
// index's skip-list advance(), and Roaring AND. One JSON object per line.
#include <algorithm>
#include <iostream>
#include <random>

#include "codec/block_codec.h"
#include "codec/roaring.h"
#include "index/shard_index.h"
#include "util/args.h"
#include "util/json.h"
#include "util/timer.h"

using namespace dc;

namespace {

struct Encoded {
  ByteWriter bytes;
  size_t n;
};

Encoded encode_list(codec::CodecId c, const std::vector<uint32_t>& docs) {
  Encoded e{ByteWriter(docs.size() * 2), docs.size()};
  std::vector<uint32_t> tmp(128);
  for (size_t lo = 0; lo < docs.size(); lo += 128) {
    size_t n = std::min<size_t>(128, docs.size() - lo);
    std::copy(docs.begin() + lo, docs.begin() + lo + n, tmp.begin());
    codec::delta_encode(tmp.data(), n, lo ? docs[lo - 1] : 0);
    codec::encode_block(c, tmp.data(), n, e.bytes);
  }
  return e;
}

uint64_t decode_list(codec::CodecId c, const Encoded& e, std::vector<uint32_t>& out) {
  out.resize(e.n);
  const uint8_t* p = e.bytes.data();
  uint32_t base = 0;
  for (size_t lo = 0; lo < e.n; lo += 128) {
    size_t n = std::min<size_t>(128, e.n - lo);
    p = codec::decode_block(c, p, n, out.data() + lo);
    codec::prefix_sum(out.data() + lo, n, base);
    base = out[lo + n - 1];
  }
  return out.empty() ? 0 : out.back();
}

size_t linear_and(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
  size_t i = 0, j = 0, n = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] < b[j]) ++i;
    else if (a[i] > b[j]) ++j;
    else {
      ++n;
      ++i;
      ++j;
    }
  }
  return n;
}

}  // namespace

int main(int argc, char** argv) {
  Args a(argc, argv);
  if (!a.has("index")) {
    std::cerr << "usage: bench_codecs --index SHARD_DIR [--schema FILE] [--min-df 64] [--max-terms 20000] [--pairs 2000]\n";
    return 2;
  }
  index::Schema schema = a.has("schema") ? index::Schema::load(a.str("schema")) : index::Schema::default_schema();
  index::IndexOptions io;
  io.read_only = true;
  io.background = false;
  index::ShardIndex idx(a.str("index"), schema, io);
  idx.open();
  auto snap = idx.snapshot();
  DC_CHECK(!snap->segments.empty(), "empty index");
  // Largest segment.
  const index::SegmentReader* seg = nullptr;
  for (auto& sv : snap->segments)
    if (!seg || sv.seg->num_docs() > seg->num_docs()) seg = sv.seg.get();
  const uint32_t min_df = static_cast<uint32_t>(a.integer("min-df", 64));
  const size_t max_terms = static_cast<size_t>(a.integer("max-terms", 20000));

  std::vector<std::pair<std::string, index::TermInfo>> terms;
  for (auto it = seg->terms().iterate(); it.valid(); it.next())
    if (it.info().df >= min_df) terms.emplace_back(it.key(), it.info());
  std::mt19937_64 rng(7);
  std::shuffle(terms.begin(), terms.end(), rng);
  if (terms.size() > max_terms) terms.resize(max_terms);

  std::vector<std::vector<uint32_t>> lists;
  uint64_t postings = 0;
  for (auto& [k, ti] : terms) {
    std::vector<uint32_t> docs;
    auto it = seg->postings(ti);
    for (DocId d = it.next(); d != kNoMoreDocs; d = it.next()) docs.push_back(d);
    postings += docs.size();
    lists.push_back(std::move(docs));
  }
  std::cout << json{{"bench", "codecs"}, {"segment", seg->uid()}, {"segment_docs", seg->num_docs()},
                    {"index_codec", codec::codec_name(seg->meta().codec)}, {"terms", lists.size()},
                    {"postings", postings}}
                   .dump()
            << std::endl;

  std::vector<uint32_t> out;
  for (auto c : {codec::CodecId::kRaw, codec::CodecId::kVByte, codec::CodecId::kPFor}) {
    std::vector<Encoded> enc;
    uint64_t bytes = 0;
    Stopwatch esw;
    for (auto& l : lists) {
      enc.push_back(encode_list(c, l));
      bytes += enc.back().bytes.size();
    }
    double encode_s = esw.elapsed_s();
    uint64_t sink = 0;
    Stopwatch dsw;
    for (int rep = 0; rep < 3; ++rep)
      for (auto& e : enc) sink += decode_list(c, e, out);
    double decode_s = dsw.elapsed_s() / 3;
    std::cout << json{{"bench", "codec"},
                      {"codec", codec::codec_name(c)},
                      {"bytes", bytes},
                      {"bits_per_posting", 8.0 * bytes / std::max<uint64_t>(postings, 1)},
                      {"encode_mints_s", postings / 1e6 / encode_s},
                      {"decode_mints_s", postings / 1e6 / decode_s},
                      {"checksum", sink}}
                     .dump()
              << std::endl;
  }
  {
    uint64_t bytes = 0;
    std::vector<RoaringBitmap> bms;
    for (auto& l : lists) {
      bms.push_back(RoaringBitmap::from_sorted(l.data(), l.size()));
      bms.back().run_optimize();
      bytes += bms.back().serialized_size();
    }
    Stopwatch dsw;
    uint64_t sink = 0;
    for (auto& b : bms) b.for_each([&](uint32_t v) { sink += v; });
    double s = dsw.elapsed_s();
    std::cout << json{{"bench", "codec"}, {"codec", "roaring"}, {"bytes", bytes},
                      {"bits_per_posting", 8.0 * bytes / std::max<uint64_t>(postings, 1)},
                      {"decode_mints_s", postings / 1e6 / s}, {"checksum", sink}}
                     .dump()
              << std::endl;

    // Intersections: random pairs of a frequent and any term.
    const size_t pairs = static_cast<size_t>(a.integer("pairs", 2000));
    std::vector<size_t> by_df(lists.size());
    for (size_t i = 0; i < by_df.size(); ++i) by_df[i] = i;
    std::sort(by_df.begin(), by_df.end(), [&](size_t x, size_t y) { return lists[x].size() > lists[y].size(); });
    std::uniform_int_distribution<size_t> any(0, lists.size() - 1), top(0, std::min<size_t>(lists.size(), 200) - 1);
    std::vector<std::pair<size_t, size_t>> P;
    for (size_t i = 0; i < pairs; ++i) P.emplace_back(by_df[top(rng)], any(rng));
    std::vector<Encoded> enc_pfor;
    for (auto& l : lists) enc_pfor.push_back(encode_list(codec::CodecId::kPFor, l));
    std::vector<uint32_t> da, db;
    uint64_t n1 = 0, n2 = 0, n3 = 0;
    Stopwatch s1;
    for (auto [x, y] : P) {
      decode_list(codec::CodecId::kPFor, enc_pfor[x], da);
      decode_list(codec::CodecId::kPFor, enc_pfor[y], db);
      n1 += linear_and(da, db);
    }
    double t1 = s1.elapsed_s();
    Stopwatch s2;
    uint64_t blocks = 0;
    for (auto [x, y] : P) {
      auto ia = seg->postings(terms[x].second), ib = seg->postings(terms[y].second);
      auto& lead = ia.cost() <= ib.cost() ? ia : ib;
      auto& other = ia.cost() <= ib.cost() ? ib : ia;
      for (DocId d = lead.next(); d != kNoMoreDocs;) {
        DocId e = other.advance(d);
        if (e == kNoMoreDocs) break;
        if (e == d) {
          ++n2;
          d = lead.next();
        } else {
          d = lead.advance(e);
        }
      }
      blocks += ia.blocks_decoded() + ib.blocks_decoded();
    }
    double t2 = s2.elapsed_s();
    Stopwatch s3;
    for (auto [x, y] : P) n3 += bms[x].and_cardinality(bms[y]);
    double t3 = s3.elapsed_s();
    std::cout << json{{"bench", "intersection"},
                      {"pairs", P.size()},
                      {"matches", n1},
                      {"consistent", n1 == n2 && n2 == n3},
                      {"decode_all_linear_us", 1e6 * t1 / P.size()},
                      {"skiplist_advance_us", 1e6 * t2 / P.size()},
                      {"skiplist_blocks_decoded_per_pair", static_cast<double>(blocks) / P.size()},
                      {"roaring_and_us", 1e6 * t3 / P.size()}}
                     .dump()
              << std::endl;
  }
  return 0;
}
