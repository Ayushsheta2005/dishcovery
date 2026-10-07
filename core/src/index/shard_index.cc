#include "index/shard_index.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <sstream>

#include "util/fs.h"
#include "util/log.h"
#include "util/timer.h"

namespace dc::index {

namespace {
constexpr const char* kCommitFile = "COMMIT";
constexpr const char* kWalFile = "wal.log";
}  // namespace

ShardIndex::ShardIndex(std::string dir, Schema schema, IndexOptions opt)
    : dir_(std::move(dir)), schema_(std::move(schema)), opt_(std::move(opt)) {
  policy_ = make_merge_policy(opt_.merge_policy);
  snapshot_ = std::make_shared<Snapshot>();
}

ShardIndex::~ShardIndex() {
  stop_background();
  if (!opt_.read_only) {
    try {
      bool has_ram = false;
      {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& e : segments_) has_ram |= e.ram;
        has_ram |= !buffer_.empty();
      }
      if (has_ram) flush();
    } catch (const std::exception& e) {
      LOG_WARN("final flush failed: " << e.what());
    }
  }
  if (wal_.is_open()) wal_.close();
}

std::string ShardIndex::new_segment_uid() {
  std::ostringstream os;
  os << "seg_" << std::setw(8) << std::setfill('0') << next_uid_.fetch_add(1);
  return os.str();
}

SegmentWriteOptions ShardIndex::write_options(const std::string& uid, const std::string& source) const {
  SegmentWriteOptions o;
  o.uid = uid;
  o.codec = opt_.codec;
  o.build_hnsw = true;
  o.hnsw_min_docs = opt_.hnsw_min_docs;
  o.hnsw = opt_.hnsw;
  o.threads = opt_.build_threads;
  o.source = source;
  return o;
}

// ------------------------------------------------------------------ open / commit

void ShardIndex::open() {
  ensure_dir(dir_);
  std::lock_guard<std::mutex> lk(mu_);
  std::set<std::string> live_files;
  std::string commit_path = join_path(dir_, kCommitFile);
  if (path_exists(commit_path)) {
    json c = json::parse(read_file(commit_path));
    DC_CHECK(c.value("schema_fp", 0ull) == schema_.fingerprint(), "index at " + dir_ + " uses a different schema");
    commit_gen_ = c.value("generation", 0ull);
    next_uid_.store(c.value("next_uid", 1ull));
    for (const auto& s : c.at("segments")) {
      SegEntry e;
      std::string uid = s.at("uid").get<std::string>();
      auto storage = std::make_shared<DirStorage>(join_path(dir_, uid), false);
      e.seg = SegmentReader::open(storage, schema_);
      e.del_file = s.value("del", std::string());
      if (!e.del_file.empty()) {
        std::string bytes = read_file(join_path(dir_, e.del_file));
        ByteReader r(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
        e.deletes = std::make_shared<RoaringBitmap>(RoaringBitmap::deserialize(r));
        live_files.insert(e.del_file);
      }
      live_files.insert(uid);
      segments_.push_back(std::move(e));
    }
  }
  // Garbage-collect files left behind by crashed flushes/merges.
  for (const auto& f : list_dir(dir_)) {
    bool is_seg = f.rfind("seg_", 0) == 0, is_del = f.rfind("del_", 0) == 0;
    if ((is_seg || is_del) && !live_files.count(f)) {
      LOG_WARN("removing unreferenced " << f);
      remove_all(join_path(dir_, f));
    }
  }
  if (!opt_.read_only && opt_.wal) {
    std::vector<std::string> wals;
    for (const auto& f : list_dir(dir_))
      if (f.rfind("wal.", 0) == 0 && f.size() > 8 && f.substr(f.size() - 4) == ".log" && f != kWalFile)
        wals.push_back(f);
    std::sort(wals.begin(), wals.end());
    for (const auto& w : wals) {
      replay_wal(join_path(dir_, w));
      pending_wal_old_.push_back(join_path(dir_, w));
    }
    if (path_exists(join_path(dir_, kWalFile))) replay_wal(join_path(dir_, kWalFile));
    refresh_locked();
    wal_path_ = join_path(dir_, kWalFile);
    wal_.open(wal_path_, std::ios::app | std::ios::binary);
    DC_CHECK(wal_.good(), "cannot open WAL " + wal_path_);
  }
  publish_locked();
  LOG_INFO("opened shard index " << dir_ << ": " << segments_.size() << " segments, "
                                 << snapshot_->live_docs << " live docs");
  if (opt_.background && !opt_.read_only) {
    maint_thread_ = std::thread([this] { maintenance_loop(); });
    merge_thread_ = std::thread([this] { merge_loop(); });
  }
}

void ShardIndex::replay_wal(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::string line;
  size_t n = 0;
  while (std::getline(in, line)) {
    if (line.size() < 2) continue;
    char op = line[0];
    std::string payload = line.substr(2);
    try {
      if (op == 'A') {
        json j = json::parse(payload);
        Document d = parse_document(j, schema_);
        remove_locked(d.id);
        buffer_ids_[d.id] = buffer_.size();
        buffer_.push_back(std::move(d));
        buffer_live_.push_back(true);
        buffer_add_ns_.push_back(now_ns());
      } else if (op == 'D') {
        remove_locked(payload);
      }
      ++n;
    } catch (const std::exception& e) {
      LOG_WARN("skipping corrupt WAL entry in " << path << ": " << e.what());
    }
  }
  LOG_INFO("replayed " << n << " WAL operations from " << path);
}

void ShardIndex::wal_append(char op, const std::string& payload) {
  if (!wal_.is_open()) return;
  wal_ << op << ' ' << payload << '\n';
  wal_.flush();  // page cache; fsync happens at commit (group-commit style)
}

void ShardIndex::wal_rotate_locked(std::string* old_path) {
  if (!wal_.is_open()) return;
  wal_.close();
  std::ostringstream os;
  os << "wal." << std::setw(10) << std::setfill('0') << (commit_gen_ + 1) << "." << now_ns() << ".log";
  *old_path = join_path(dir_, os.str());
  std::rename(wal_path_.c_str(), old_path->c_str());
  wal_.open(wal_path_, std::ios::app | std::ios::binary);
}

void ShardIndex::commit() {
  std::lock_guard<std::mutex> lk(mu_);
  commit_locked();
}

void ShardIndex::commit_locked() {
  ++commit_gen_;
  std::vector<std::string> obsolete_dels;
  json segs = json::array();
  for (auto& e : segments_) {
    if (e.ram) continue;
    if (e.deletes_dirty) {
      std::string name = "del_" + e.seg->uid() + "_" + std::to_string(commit_gen_) + ".rbm";
      ByteWriter w;
      e.deletes->serialize(w);
      write_file_atomic(join_path(dir_, name),
                        std::string_view(reinterpret_cast<const char*>(w.data()), w.size()));
      if (!e.del_file.empty()) obsolete_dels.push_back(e.del_file);
      e.del_file = name;
      e.deletes_dirty = false;
    }
    json s = {{"uid", e.seg->uid()}, {"docs", e.seg->num_docs()}};
    if (!e.del_file.empty()) s["del"] = e.del_file;
    segs.push_back(s);
  }
  json c = {{"generation", commit_gen_},
            {"next_uid", next_uid_.load()},
            {"schema_fp", schema_.fingerprint()},
            {"segments", segs},
            {"committed_ms", wall_ms()}};
  write_file_atomic(join_path(dir_, kCommitFile), c.dump(2));
  for (auto& d : obsolete_dels) remove_all(join_path(dir_, d));
  for (auto& w : pending_wal_old_) std::remove(w.c_str());
  pending_wal_old_.clear();
}

void ShardIndex::adopt_segment(const std::string& uid) {
  auto storage = std::make_shared<DirStorage>(join_path(dir_, uid), false);
  auto seg = SegmentReader::open(storage, schema_);
  std::lock_guard<std::mutex> lk(mu_);
  SegEntry e;
  e.seg = seg;
  segments_.push_back(std::move(e));
  publish_locked();
}

// ------------------------------------------------------------------ writes

void ShardIndex::add(Document doc, const std::string& raw_json) {
  DC_CHECK(!opt_.read_only, "index is read-only");
  std::lock_guard<std::mutex> lk(mu_);
  if (opt_.wal) wal_append('A', raw_json);
  remove_locked(doc.id);  // add == upsert
  buffer_ids_[doc.id] = buffer_.size();
  buffer_.push_back(std::move(doc));
  buffer_live_.push_back(true);
  buffer_add_ns_.push_back(now_ns());
  docs_added_.fetch_add(1);
}

bool ShardIndex::remove(const std::string& id) {
  DC_CHECK(!opt_.read_only, "index is read-only");
  std::lock_guard<std::mutex> lk(mu_);
  if (opt_.wal) wal_append('D', id);
  bool found = remove_locked(id);
  if (found) {
    docs_deleted_.fetch_add(1);
    publish_locked();
  }
  return found;
}

bool ShardIndex::remove_locked(const std::string& id) {
  bool found = false;
  auto bit = buffer_ids_.find(id);
  if (bit != buffer_ids_.end()) {
    buffer_live_[bit->second] = false;
    buffer_ids_.erase(bit);
    found = true;
  }
  for (auto& e : segments_) {
    auto d = e.seg->find(id);
    if (!d || (e.deletes && e.deletes->contains(*d))) continue;
    auto nb = e.deletes ? std::make_shared<RoaringBitmap>(*e.deletes) : std::make_shared<RoaringBitmap>();
    nb->add(*d);
    e.deletes = std::move(nb);
    e.deletes_dirty = !e.ram;
    found = true;
  }
  return found;
}

void ShardIndex::refresh() {
  std::lock_guard<std::mutex> lk(mu_);
  refresh_locked();
}

void ShardIndex::refresh_locked() {
  if (buffer_.empty()) return;
  Stopwatch sw;
  SegmentBuilder b(schema_);
  std::vector<uint64_t> add_times;
  for (size_t i = 0; i < buffer_.size(); ++i) {
    if (!buffer_live_[i]) continue;
    b.add(buffer_[i]);
    add_times.push_back(buffer_add_ns_[i]);
  }
  buffer_.clear();
  buffer_live_.clear();
  buffer_ids_.clear();
  buffer_add_ns_.clear();
  if (b.num_docs() == 0) return;
  auto storage = std::make_shared<MemoryStorage>();
  auto wo = write_options(new_segment_uid(), "refresh");
  wo.build_hnsw = false;  // RAM segments are small: vector search scans them
  b.write(*storage, wo);
  SegEntry e;
  e.seg = SegmentReader::open(storage, schema_);
  e.ram = true;
  segments_.push_back(std::move(e));
  publish_locked();
  uint64_t t = now_ns();
  for (uint64_t a : add_times) visibility_us_.record((t - a) / 1000);
  refresh_us_.record(sw.elapsed_us());
  refreshes_.fetch_add(1);
}

void ShardIndex::flush() {
  if (opt_.read_only) return;
  std::lock_guard<std::mutex> flk(flush_mu_);
  Stopwatch sw;
  std::vector<std::string> uids;
  std::vector<MergeSource> sources;
  std::vector<std::shared_ptr<const RoaringBitmap>> captured;
  std::string uid;
  {
    std::lock_guard<std::mutex> lk(mu_);
    refresh_locked();
    for (auto& e : segments_) {
      if (!e.ram) continue;
      uids.push_back(e.seg->uid());
      sources.push_back({e.seg, e.deletes});
      captured.push_back(e.deletes);
    }
    if (sources.empty()) {
      bool dirty = false;
      for (auto& e : segments_) dirty |= e.deletes_dirty;
      if (dirty) commit_locked();
      return;
    }
    std::string old;
    wal_rotate_locked(&old);
    if (!old.empty()) pending_wal_old_.push_back(old);
    uid = new_segment_uid();
  }
  auto storage = std::make_shared<DirStorage>(join_path(dir_, uid), true);
  MergeResult res = merge_segments(sources, schema_, *storage, write_options(uid, "flush"));
  std::shared_ptr<SegmentReader> reader;
  if (res.meta.num_docs > 0) reader = SegmentReader::open(storage, schema_);
  else storage->destroy();
  {
    std::lock_guard<std::mutex> lk(mu_);
    install_merged_locked(uids, captured, res, reader);
    commit_locked();
    publish_locked();
  }
  bytes_flushed_.fetch_add(res.bytes_written);
  flushes_.fetch_add(1);
  flush_us_.record(sw.elapsed_us());
  merge_cv_.notify_one();
}

void ShardIndex::install_merged_locked(const std::vector<std::string>& uids,
                                       const std::vector<std::shared_ptr<const RoaringBitmap>>& captured,
                                       const MergeResult& res, std::shared_ptr<SegmentReader> merged) {
  std::vector<uint32_t> carried;
  size_t insert_at = SIZE_MAX;
  for (size_t i = 0; i < uids.size(); ++i) {
    for (size_t k = 0; k < segments_.size(); ++k) {
      if (segments_[k].seg->uid() != uids[i]) continue;
      const auto& cur = segments_[k].deletes;
      if (cur && cur != captured[i]) {
        cur->for_each([&](uint32_t d) {
          if (captured[i] && captured[i]->contains(d)) return;
          int32_t nd = res.doc_maps[i][d];
          if (nd >= 0) carried.push_back(static_cast<uint32_t>(nd));
        });
      }
      insert_at = std::min(insert_at, k);
      break;
    }
  }
  std::vector<SegEntry> next;
  for (size_t k = 0; k < segments_.size(); ++k) {
    bool is_source = std::find(uids.begin(), uids.end(), segments_[k].seg->uid()) != uids.end();
    if (k == insert_at && merged) {
      SegEntry e;
      e.seg = merged;
      if (!carried.empty()) {
        std::sort(carried.begin(), carried.end());
        carried.erase(std::unique(carried.begin(), carried.end()), carried.end());
        e.deletes = std::make_shared<RoaringBitmap>(RoaringBitmap::from_sorted(carried.data(), carried.size()));
        e.deletes_dirty = true;
      }
      next.push_back(std::move(e));
    }
    if (is_source) {
      segments_[k].seg->mark_obsolete();
      continue;
    }
    next.push_back(std::move(segments_[k]));
  }
  segments_ = std::move(next);
}

void ShardIndex::publish_locked() {
  auto snap = std::make_shared<Snapshot>();
  snap->generation = generation_.fetch_add(1) + 1;
  for (auto& e : segments_) {
    snap->segments.push_back({e.seg, e.deletes});
    snap->max_docs += e.seg->num_docs();
    snap->live_docs += e.seg->num_docs() - (e.deletes ? e.deletes->cardinality() : 0);
  }
  std::lock_guard<std::mutex> lk(snap_mu_);
  snapshot_ = std::move(snap);
}

std::shared_ptr<const Snapshot> ShardIndex::snapshot() const {
  std::lock_guard<std::mutex> lk(snap_mu_);
  return snapshot_;
}

// ------------------------------------------------------------------ merges

std::vector<SegmentStat> ShardIndex::disk_stats_locked() const {
  std::vector<SegmentStat> out;
  for (auto& e : segments_) {
    if (e.ram) continue;
    out.push_back({e.seg->uid(), e.seg->size_bytes(), e.seg->num_docs(),
                   static_cast<uint32_t>(e.deletes ? e.deletes->cardinality() : 0)});
  }
  return out;
}

void ShardIndex::maybe_merge() {
  if (opt_.read_only) return;
  std::vector<MergeSpec> specs;
  {
    std::lock_guard<std::mutex> lk(mu_);
    specs = policy_->find_merges(disk_stats_locked(), merging_);
    for (auto& s : specs)
      for (auto& u : s.uids) merging_.insert(u);
  }
  for (auto& s : specs) run_merge(s);
}

void ShardIndex::force_merge(int max_segments) {
  if (opt_.read_only) return;
  flush();
  for (int round = 0; round < 64; ++round) {
    std::vector<MergeSpec> specs;
    {
      std::lock_guard<std::mutex> lk(mu_);
      specs = policy_->find_forced_merges(disk_stats_locked(), max_segments);
      for (auto& s : specs)
        for (auto& u : s.uids) merging_.insert(u);
    }
    if (specs.empty()) return;
    for (auto& s : specs) run_merge(s);
  }
}

void ShardIndex::run_merge(const MergeSpec& spec) {
  Stopwatch sw;
  std::vector<MergeSource> sources;
  std::vector<std::shared_ptr<const RoaringBitmap>> captured;
  std::string uid;
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& u : spec.uids)
      for (auto& e : segments_)
        if (e.seg->uid() == u) {
          sources.push_back({e.seg, e.deletes});
          captured.push_back(e.deletes);
        }
    uid = new_segment_uid();
  }
  if (sources.size() != spec.uids.size()) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& u : spec.uids) merging_.erase(u);
    return;
  }
  try {
    auto storage = std::make_shared<DirStorage>(join_path(dir_, uid), true);
    MergeResult res = merge_segments(sources, schema_, *storage, write_options(uid, "merge"));
    std::shared_ptr<SegmentReader> reader;
    if (res.meta.num_docs > 0) reader = SegmentReader::open(storage, schema_);
    else storage->destroy();
    std::lock_guard<std::mutex> lk(mu_);
    install_merged_locked(spec.uids, captured, res, reader);
    commit_locked();
    publish_locked();
    for (auto& u : spec.uids) merging_.erase(u);
    bytes_merge_written_.fetch_add(res.bytes_written);
    bytes_merge_read_.fetch_add(res.bytes_read);
  } catch (...) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& u : spec.uids) merging_.erase(u);
    throw;
  }
  merges_.fetch_add(1);
  merge_us_.record(sw.elapsed_us());
}

// ------------------------------------------------------------------ background

void ShardIndex::maintenance_loop() {
  uint64_t last_flush = now_ns();
  while (!stop_.load()) {
    {
      std::unique_lock<std::mutex> lk(merge_cv_mu_);
      merge_cv_.wait_for(lk, std::chrono::milliseconds(opt_.refresh_ms), [this] { return stop_.load(); });
    }
    if (stop_.load()) break;
    try {
      uint64_t ram_bytes = 0;
      {
        std::lock_guard<std::mutex> lk(mu_);
        refresh_locked();
        for (auto& e : segments_)
          if (e.ram) ram_bytes += e.seg->size_bytes();
      }
      if ((now_ns() - last_flush) / 1000000 >= opt_.flush_ms || ram_bytes > opt_.ram_max_bytes) {
        flush();
        last_flush = now_ns();
      }
    } catch (const std::exception& e) {
      LOG_ERROR("maintenance: " << e.what());
    }
  }
}

void ShardIndex::merge_loop() {
  while (!stop_.load()) {
    {
      std::unique_lock<std::mutex> lk(merge_cv_mu_);
      merge_cv_.wait_for(lk, std::chrono::milliseconds(1000), [this] { return stop_.load(); });
    }
    if (stop_.load()) break;
    try {
      maybe_merge();
    } catch (const std::exception& e) {
      LOG_ERROR("merge: " << e.what());
    }
  }
}

void ShardIndex::stop_background() {
  stop_.store(true);
  merge_cv_.notify_all();
  if (maint_thread_.joinable()) maint_thread_.join();
  if (merge_thread_.joinable()) merge_thread_.join();
}

json ShardIndex::stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  json segs = json::array();
  uint64_t disk_bytes = 0, ram_bytes = 0, max_docs = 0, dels = 0;
  for (auto& e : segments_) {
    uint64_t d = e.deletes ? e.deletes->cardinality() : 0;
    segs.push_back({{"uid", e.seg->uid()},
                    {"docs", e.seg->num_docs()},
                    {"deleted", d},
                    {"bytes", e.seg->size_bytes()},
                    {"ram", e.ram},
                    {"hnsw", e.seg->meta().has_hnsw},
                    {"source", e.seg->meta().source}});
    (e.ram ? ram_bytes : disk_bytes) += e.seg->size_bytes();
    max_docs += e.seg->num_docs();
    dels += d;
  }
  return {{"dir", dir_},
          {"generation", generation_.load()},
          {"commit_generation", commit_gen_},
          {"segments", segs},
          {"num_segments", segments_.size()},
          {"max_docs", max_docs},
          {"live_docs", max_docs - dels},
          {"buffered_docs", buffer_.size()},
          {"disk_bytes", disk_bytes},
          {"ram_bytes", ram_bytes},
          {"docs_added", docs_added_.load()},
          {"docs_deleted", docs_deleted_.load()},
          {"refreshes", refreshes_.load()},
          {"flushes", flushes_.load()},
          {"merges", merges_.load()},
          {"bytes_flushed", bytes_flushed_.load()},
          {"bytes_merge_written", bytes_merge_written_.load()},
          {"bytes_merge_read", bytes_merge_read_.load()},
          {"write_amplification",
           bytes_flushed_.load() ? static_cast<double>(bytes_flushed_.load() + bytes_merge_written_.load()) /
                                       static_cast<double>(bytes_flushed_.load())
                                 : 0.0},
          {"merge_policy", policy_->describe()},
          {"visibility_us", json::parse(visibility_us_.to_json())},
          {"refresh_us", json::parse(refresh_us_.to_json())},
          {"flush_us", json::parse(flush_us_.to_json())},
          {"merge_us", json::parse(merge_us_.to_json())}};
}

}  // namespace dc::index
