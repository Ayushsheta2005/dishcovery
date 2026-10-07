// Postings, skip lists, term dictionary, segments, merges and the NRT shard index.
#include <unistd.h>

#include <filesystem>
#include <random>
#include <set>
#include <thread>

#include "index/shard_index.h"
#include "testing.h"

using namespace dc;
using namespace dc::index;

namespace {

std::string tmpdir(const std::string& name) {
  auto p = std::filesystem::temp_directory_path() / ("dctest_" + name + "_" + std::to_string(::getpid()));
  std::filesystem::remove_all(p);
  std::filesystem::create_directories(p);
  return p.string();
}

Schema small_schema() {
  Schema s = Schema::default_schema();
  s.vector_dim = 8;
  return s;
}

json make_doc(int i, const std::string& name, const std::string& type = "dish", double lat = 39.95,
              double lon = -75.16) {
  std::vector<float> v(8, 0.0f);
  v[i % 8] = 1.0f;
  v[(i + 3) % 8] = 0.5f;
  return json{{"id", "d:" + std::to_string(i)},
              {"type", type},
              {"restaurant_id", "r:" + std::to_string(i / 10)},
              {"name", name},
              {"restaurant_name", "Resto " + std::to_string(i / 10)},
              {"cuisines", json::array({i % 2 ? "indian" : "italian"})},
              {"categories", json::array({"Restaurants"})},
              {"description", "A tasty " + name + " made fresh"},
              {"tags", json::array({"spicy"})},
              {"reviews", "loved the " + name},
              {"diet", json::array({i % 3 ? "non_veg" : "veg"})},
              {"price", 10.0 + i % 7},
              {"price_tier", 1 + i % 3},
              {"rating", 3.0 + (i % 5) * 0.4},
              {"review_count", 10 * i},
              {"popularity", (i % 100) / 100.0},
              {"lat", lat},
              {"lon", lon},
              {"city", "Philadelphia"},
              {"hours", std::string(84, 'f')},
              {"meal_affinity", json::array({0.1, 0.4, 0.1, 0.4, 0.0})},
              {"delivery_polygon", json::array()},
              {"delivery_radius_km", 4.0},
              {"embedding", v}};
}

}  // namespace

TEST(Postings, RoundTripWithSkipsAndPositions) {
  for (auto codec : {codec::CodecId::kRaw, codec::CodecId::kVByte, codec::CodecId::kPFor}) {
    MemoryStorage st;
    auto post = st.create("p");
    auto pos = st.create("q");
    PostingsWriter w(post.get(), pos.get(), codec);
    PostingsBuffer buf;
    std::mt19937 rng(7);
    uint32_t d = 0;
    for (int i = 0; i < 5000; ++i) {
      d += 1 + rng() % 40;
      uint32_t tf = 1 + rng() % 4;
      buf.docs.push_back(d);
      buf.freqs.push_back(tf);
      uint32_t p = rng() % 3;
      for (uint32_t k = 0; k < tf; ++k) {
        buf.positions.push_back(p);
        p += 1 + rng() % 5;
      }
      buf.max_tf = std::max(buf.max_tf, tf);
    }
    TermInfo ti = w.write(buf, true);
    post->close();
    pos->close();
    // Sequential scan.
    PostingsIterator it(st.open("p"), st.open("q"), ti, codec);
    size_t i = 0, pcur = 0;
    std::vector<uint32_t> got;
    for (DocId x = it.next(); x != kNoMoreDocs; x = it.next(), ++i) {
      ASSERT_EQ(x, buf.docs[i]);
      ASSERT_EQ(it.freq(), buf.freqs[i]);
      if (i % 7 == 0) {
        it.positions(got);
        for (uint32_t k = 0; k < buf.freqs[i]; ++k) ASSERT_EQ(got[k], buf.positions[pcur + k]);
      }
      pcur += buf.freqs[i];
    }
    EXPECT_EQ(i, buf.docs.size());
    // Random advance() against lower_bound, checking the skip list touches few blocks.
    for (int trial = 0; trial < 50; ++trial) {
      PostingsIterator a(st.open("p"), st.open("q"), ti, codec);
      DocId target = 0;
      while (true) {
        target += 1 + rng() % 30000;  // jumps span ~12 blocks
        DocId got_d = a.advance(target);
        auto lb = std::lower_bound(buf.docs.begin(), buf.docs.end(), target);
        if (lb == buf.docs.end()) {
          EXPECT_EQ(got_d, kNoMoreDocs);
          break;
        }
        ASSERT_EQ(got_d, *lb);
        ASSERT_EQ(a.freq(), buf.freqs[lb - buf.docs.begin()]);
      }
      EXPECT_LT(a.blocks_decoded(), uint64_t(15));  // 40 blocks total; the skip list avoids most
    }
  }
}

TEST(TermDict, LookupAndIterate) {
  MemoryStorage st;
  {
    TermDictWriter w(st.create("t"));
    for (int i = 0; i < 1000; ++i) {
      char buf[32];
      std::snprintf(buf, sizeof buf, "term%05d", i * 3);
      TermInfo ti;
      ti.df = i + 1;
      ti.post_off = i * 100ull;
      ti.pos_off = i * 10ull;
      ti.post_len = 100;
      w.add(make_key(i < 500 ? 1 : 2, buf), ti);
    }
    w.finish();
  }
  TermDictReader r(st.open("t"));
  EXPECT_EQ(r.num_terms(), uint64_t(1000));
  auto ti = r.lookup(make_key(1, "term00030"));
  ASSERT_TRUE(ti.has_value());
  EXPECT_EQ(ti->df, 11u);
  EXPECT_EQ(ti->post_off, uint64_t(1000));
  EXPECT_FALSE(r.lookup(make_key(1, "term00031")).has_value());
  EXPECT_FALSE(r.lookup(make_key(2, "term00030")).has_value());
  EXPECT_TRUE(r.lookup(make_key(2, "term02997")).has_value());
  size_t n = 0;
  std::string prev;
  for (auto it = r.iterate(); it.valid(); it.next(), ++n) {
    EXPECT_TRUE(n == 0 || it.key() > prev);
    prev = it.key();
  }
  EXPECT_EQ(n, size_t(1000));
  size_t f2 = 0;
  r.scan_prefix(std::string(1, '\x02'), [&](const std::string&, const TermInfo&) { ++f2; });
  EXPECT_EQ(f2, size_t(500));
}

TEST(Segment, BuildOpenAndRead) {
  Schema schema = small_schema();
  SegmentBuilder b(schema);
  for (int i = 0; i < 300; ++i)
    b.add(parse_document(make_doc(i, i % 4 == 0 ? "Chicken Biryani" : "Margherita Pizza"), schema));
  auto st = std::make_shared<MemoryStorage>();
  SegmentWriteOptions o;
  o.uid = "seg_test";
  o.hnsw_min_docs = 100;
  b.write(*st, o);
  auto seg = SegmentReader::open(st, schema);
  EXPECT_EQ(seg->num_docs(), 300u);
  EXPECT_TRUE(seg->meta().has_hnsw);
  int fname = schema.text_field("name");
  auto ti = seg->term(static_cast<uint8_t>(fname), "biryani");
  ASSERT_TRUE(ti.has_value());
  EXPECT_EQ(ti->df, 75u);
  auto d = seg->find("d:42");
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(seg->ext_id(*d), "d:42");
  EXPECT_FALSE(seg->find("d:999").has_value());
  EXPECT_NEAR(seg->lat(*d), 39.95, 1e-4);
  EXPECT_TRUE(seg->stored(*d).find("Margherita") != std::string_view::npos);
  // Keyword filter bitmap.
  int fdiet = schema.keyword_field("diet");
  auto fe = seg->filter_entry(make_key(static_cast<uint8_t>(fdiet), "veg"));
  ASSERT_TRUE(fe.has_value());
  EXPECT_EQ(seg->filter_bitmap(*fe).cardinality(), uint64_t(100));
  // A delivery-cover key exists for the restaurant's own cell.
  size_t covers = 0;
  seg->filters().scan_prefix(std::string(1, static_cast<char>(kGeoCoverField)),
                             [&](const std::string&, const TermInfo&) { ++covers; });
  EXPECT_GT(covers, size_t(0));
}

TEST(Segment, MergeDropsDeletesAndRemaps) {
  Schema schema = small_schema();
  std::vector<MergeSource> sources;
  for (int s = 0; s < 3; ++s) {
    SegmentBuilder b(schema);
    for (int i = 0; i < 100; ++i)
      b.add(parse_document(make_doc(s * 100 + i, i % 2 ? "paneer tikka" : "pad thai"), schema));
    auto st = std::make_shared<MemoryStorage>();
    SegmentWriteOptions o;
    o.uid = "s" + std::to_string(s);
    b.write(*st, o);
    auto del = std::make_shared<RoaringBitmap>();
    for (uint32_t d = 0; d < 100; d += 10) del->add(d);
    sources.push_back({SegmentReader::open(st, schema), del});
  }
  auto out = std::make_shared<MemoryStorage>();
  SegmentWriteOptions o;
  o.uid = "merged";
  auto res = merge_segments(sources, schema, *out, o);
  EXPECT_EQ(res.meta.num_docs, 270u);
  auto m = SegmentReader::open(out, schema);
  EXPECT_FALSE(m->find("d:0").has_value());
  EXPECT_FALSE(m->find("d:110").has_value());
  auto d = m->find("d:111");
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(res.doc_maps[1][11], static_cast<int32_t>(*d));
  int fname = schema.text_field("name");
  auto ti = m->term(static_cast<uint8_t>(fname), "paneer");
  ASSERT_TRUE(ti.has_value());
  EXPECT_EQ(ti->df, 150u);  // odd docs survive: 3 * 50
  // Positions survive the merge.
  auto it = m->postings(*ti);
  it.next();
  std::vector<uint32_t> pos;
  it.positions(pos);
  ASSERT_EQ(pos.size(), size_t(1));
  EXPECT_EQ(pos[0], 0u);
  int fdiet = schema.keyword_field("diet");
  auto fe = m->filter_entry(make_key(static_cast<uint8_t>(fdiet), "veg"));
  ASSERT_TRUE(fe.has_value());
  uint64_t veg = 0;
  for (auto& sv : sources)
    for (uint32_t x = 0; x < 100; ++x) {
      std::string id(sv.segment->ext_id(x));
      int i = std::stoi(id.substr(2));
      if (i % 3 == 0 && !sv.deletes->contains(x)) ++veg;
    }
  EXPECT_EQ(m->filter_bitmap(*fe).cardinality(), veg);
}

TEST(ShardIndex, NrtRefreshFlushDeleteReopen) {
  std::string dir = tmpdir("nrt");
  Schema schema = small_schema();
  IndexOptions io;
  io.background = false;
  io.hnsw_min_docs = 50;
  {
    ShardIndex idx(dir, schema, io);
    idx.open();
    for (int i = 0; i < 120; ++i) {
      json j = make_doc(i, "veg biryani");
      idx.add(parse_document(j, schema), j.dump());
    }
    EXPECT_EQ(idx.snapshot()->live_docs, uint64_t(0));  // not yet refreshed
    idx.refresh();
    EXPECT_EQ(idx.snapshot()->live_docs, uint64_t(120));
    EXPECT_TRUE(idx.remove("d:5"));
    EXPECT_FALSE(idx.remove("d:5"));
    EXPECT_EQ(idx.snapshot()->live_docs, uint64_t(119));
    idx.flush();
    auto snap = idx.snapshot();
    ASSERT_EQ(snap->segments.size(), size_t(1));
    EXPECT_TRUE(snap->segments[0].seg->on_disk());
    EXPECT_EQ(snap->live_docs, uint64_t(119));
    // Update = delete + add.
    json j = make_doc(7, "butter chicken");
    idx.add(parse_document(j, schema), j.dump());
    idx.refresh();
    EXPECT_EQ(idx.snapshot()->live_docs, uint64_t(119));
    EXPECT_TRUE(idx.remove("d:8"));  // delete on a disk segment, persisted at commit
    idx.flush();
  }
  {
    // WAL-free reopen from the commit point: deletes and updates persisted.
    ShardIndex idx(dir, schema, io);
    idx.open();
    auto snap = idx.snapshot();
    EXPECT_EQ(snap->live_docs, uint64_t(118));
    bool found_update = false;
    for (auto& sv : snap->segments) {
      auto d = sv.seg->find("d:7");
      if (d && !sv.is_deleted(*d)) found_update = sv.seg->stored(*d).find("butter") != std::string_view::npos;
      auto d8 = sv.seg->find("d:8");
      if (d8) EXPECT_TRUE(sv.is_deleted(*d8));
    }
    EXPECT_TRUE(found_update);
    idx.force_merge(1);
    EXPECT_EQ(idx.snapshot()->segments.size(), size_t(1));
    EXPECT_EQ(idx.snapshot()->live_docs, uint64_t(118));
    EXPECT_EQ(idx.snapshot()->max_docs, uint64_t(118));
  }
  {
    // WAL replay: adds that were never flushed survive a crash-like reopen.
    IndexOptions io2 = io;
    auto idx = std::make_unique<ShardIndex>(dir, schema, io2);
    idx->open();
    json j = make_doc(500, "garlic naan");
    idx->add(parse_document(j, schema), j.dump());
    idx->stop_background();
    // Simulate a crash: leak the object so the destructor's final flush never runs.
    (void)idx.release();
    ShardIndex again(dir, schema, io2);
    again.open();
    bool found = false;
    for (auto& sv : again.snapshot()->segments) found |= sv.seg->find("d:500").has_value();
    EXPECT_TRUE(found);
  }
  std::filesystem::remove_all(dir);
}

TEST(MergePolicy, TieredMergesSmallSegmentsFirst) {
  TieredMergePolicy p({10, 4.0, 1 << 20, 1ull << 30, 20.0});
  std::vector<SegmentStat> segs;
  segs.push_back({"big", 500ull << 20, 1000000, 0});
  // 30 small segments exceed the tier budget (~13 segments for 530 MB).
  for (int i = 0; i < 30; ++i) segs.push_back({"small" + std::to_string(i), 1ull << 20, 2000, 0});
  auto m = p.find_merges(segs, {});
  ASSERT_EQ(m.size(), size_t(1));
  for (auto& u : m[0].uids) EXPECT_TRUE(u != "big");
  EXPECT_GE(m[0].uids.size(), size_t(2));
}

TEST(MergePolicy, LogMergesFullLevel) {
  LogMergePolicy p({4, 1 << 20, 1ull << 30});
  std::vector<SegmentStat> segs;
  for (int i = 0; i < 3; ++i) segs.push_back({"a" + std::to_string(i), 1ull << 20, 100, 0});
  EXPECT_TRUE(p.find_merges(segs, {}).empty());
  segs.push_back({"a3", 1ull << 20, 100, 0});
  auto m = p.find_merges(segs, {});
  ASSERT_EQ(m.size(), size_t(1));
  EXPECT_EQ(m[0].uids.size(), size_t(4));
}
