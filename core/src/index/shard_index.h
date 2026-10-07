// One shard's index: a list of immutable segments plus near-real-time writes.
//
// Write path:
//   add()/remove()  → write-ahead log + in-memory buffer (deletes become
//                      tombstones immediately)
//   refresh()       → buffer becomes a RAM segment; searchable (every
//                      refresh_ms, default 200 ms → "new doc searchable < 1 s")
//   flush()         → RAM segments merged into one on-disk segment, commit
//                      point written, WAL rotated (every flush_ms or when RAM
//                      segments grow past ram_max_bytes)
//   merge thread    → merge policy picks disk segments to merge in background
//
// Read path: snapshot() returns an immutable point-in-time view (segments +
// their tombstone bitmaps). Readers never take the writer lock; obsolete
// segments are deleted from disk only after the last snapshot drops them.
#pragma once

#include <atomic>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "codec/roaring.h"
#include "index/merge_policy.h"
#include "index/segment.h"
#include "util/histogram.h"

namespace dc::index {

struct IndexOptions {
  codec::CodecId codec = codec::CodecId::kPFor;
  uint32_t refresh_ms = 200;
  uint32_t flush_ms = 5000;
  uint32_t refresh_max_docs = 20000;          // refresh early when the buffer grows
  uint64_t ram_max_bytes = 256ull << 20;      // flush early when RAM segments grow
  bool wal = true;
  bool background = true;                     // run refresh/flush/merge threads
  json merge_policy = json{{"type", "tiered"}};
  vec::HnswParams hnsw;
  uint32_t hnsw_min_docs = 2000;
  unsigned build_threads = 2;                 // HNSW construction during flush/merge
  bool read_only = false;                     // serve a built index without writes
};

struct SegmentView {
  std::shared_ptr<const SegmentReader> seg;
  std::shared_ptr<const RoaringBitmap> deletes;  // null when none
  bool is_deleted(DocId d) const { return deletes && deletes->contains(d); }
};

struct Snapshot {
  uint64_t generation = 0;
  std::vector<SegmentView> segments;
  uint64_t max_docs = 0;   // including deleted
  uint64_t live_docs = 0;
};

class ShardIndex {
 public:
  ShardIndex(std::string dir, Schema schema, IndexOptions opt);
  ~ShardIndex();
  ShardIndex(const ShardIndex&) = delete;
  ShardIndex& operator=(const ShardIndex&) = delete;

  void open();  // load COMMIT, open segments, replay WAL, start threads

  // `raw_json` is what gets logged to the WAL (the original request line).
  void add(Document doc, const std::string& raw_json);
  bool remove(const std::string& id);
  void refresh();
  void flush();
  void maybe_merge();                 // run the merge policy once (synchronously)
  void force_merge(int max_segments);

  // Offline indexer support: adopt a segment directory already written under dir().
  void adopt_segment(const std::string& uid);
  void commit();

  std::shared_ptr<const Snapshot> snapshot() const;
  uint64_t generation() const { return generation_.load(); }
  const Schema& schema() const { return schema_; }
  const std::string& dir() const { return dir_; }
  std::string new_segment_uid();
  json stats() const;

  void stop_background();

 private:
  struct SegEntry {
    std::shared_ptr<SegmentReader> seg;
    std::shared_ptr<const RoaringBitmap> deletes;
    bool ram = false;
    bool deletes_dirty = false;      // deletes changed since last commit
    std::string del_file;            // persisted tombstone file name
  };

  void refresh_locked();
  void publish_locked();
  void commit_locked();
  void wal_append(char op, const std::string& payload);
  void wal_rotate_locked(std::string* old_path);
  void replay_wal(const std::string& path);
  bool remove_locked(const std::string& id);
  std::vector<SegmentStat> disk_stats_locked() const;
  void run_merge(const MergeSpec& spec);
  // Replace `sources` (by uid) with `merged`, carrying deletes that arrived
  // during the merge through the doc maps.
  void install_merged_locked(const std::vector<std::string>& uids,
                             const std::vector<std::shared_ptr<const RoaringBitmap>>& captured,
                             const MergeResult& res, std::shared_ptr<SegmentReader> merged);
  SegmentWriteOptions write_options(const std::string& uid, const std::string& source) const;
  void maintenance_loop();
  void merge_loop();

  std::string dir_;
  Schema schema_;
  IndexOptions opt_;
  std::unique_ptr<MergePolicy> policy_;

  mutable std::mutex mu_;               // guards everything below except snapshot_
  std::vector<SegEntry> segments_;      // creation order
  std::vector<Document> buffer_;        // not yet searchable
  std::vector<bool> buffer_live_;
  std::unordered_map<std::string, size_t> buffer_ids_;
  std::vector<uint64_t> buffer_add_ns_; // for visibility-latency stats
  std::set<std::string> merging_;
  std::atomic<uint64_t> next_uid_{1};
  uint64_t commit_gen_ = 0;
  std::ofstream wal_;
  std::string wal_path_;
  std::vector<std::string> pending_wal_old_;

  mutable std::mutex snap_mu_;
  std::shared_ptr<const Snapshot> snapshot_;
  std::atomic<uint64_t> generation_{0};

  std::mutex flush_mu_;                 // one flush at a time
  std::atomic<bool> stop_{false};
  std::thread maint_thread_, merge_thread_;
  std::condition_variable merge_cv_;
  std::mutex merge_cv_mu_;

  // stats
  std::atomic<uint64_t> docs_added_{0}, docs_deleted_{0}, refreshes_{0}, flushes_{0}, merges_{0};
  std::atomic<uint64_t> bytes_flushed_{0}, bytes_merge_written_{0}, bytes_merge_read_{0};
  Histogram visibility_us_, refresh_us_, flush_us_, merge_us_;
};

}  // namespace dc::index
