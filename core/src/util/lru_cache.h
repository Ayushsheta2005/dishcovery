// Sharded, byte-bounded LRU cache with hit/miss counters.
//
// Used for the postings cache and the filter cache on shards. Values are held
// by shared_ptr so an entry evicted while a query still uses it stays alive.
#pragma once

#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace dc {

template <class V>
class LruCache {
 public:
  struct Stats {
    uint64_t hits = 0, misses = 0, evictions = 0, invalidations = 0, entries = 0, bytes = 0, capacity = 0;
  };

  explicit LruCache(size_t capacity_bytes, size_t shards = 16)
      : capacity_(capacity_bytes), shards_(shards ? shards : 1) {
    for (auto& s : shards_) s.capacity = capacity_bytes / shards_.size();
  }

  std::shared_ptr<const V> get(const std::string& key) {
    if (!capacity_) return nullptr;
    Shard& s = shard(key);
    std::lock_guard<std::mutex> lk(s.mu);
    auto it = s.map.find(key);
    if (it == s.map.end()) {
      misses_.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    s.lru.splice(s.lru.begin(), s.lru, it->second);
    hits_.fetch_add(1, std::memory_order_relaxed);
    return it->second->value;
  }

  void put(const std::string& key, std::shared_ptr<const V> value, size_t bytes) {
    if (!capacity_) return;
    Shard& s = shard(key);
    if (bytes > s.capacity) return;  // never cache something bigger than a shard
    std::lock_guard<std::mutex> lk(s.mu);
    auto it = s.map.find(key);
    if (it != s.map.end()) {
      s.bytes -= it->second->bytes;
      s.lru.erase(it->second);
      s.map.erase(it);
    }
    s.lru.push_front(Entry{key, std::move(value), bytes});
    s.map[key] = s.lru.begin();
    s.bytes += bytes;
    while (s.bytes > s.capacity && !s.lru.empty()) {
      auto& last = s.lru.back();
      s.bytes -= last.bytes;
      s.map.erase(last.key);
      s.lru.pop_back();
      evictions_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  // Get-or-compute. `compute` returns {value, bytes}.
  std::shared_ptr<const V> get_or(const std::string& key,
                                  const std::function<std::pair<std::shared_ptr<const V>, size_t>()>& compute) {
    if (auto v = get(key)) return v;
    auto [val, bytes] = compute();
    put(key, val, bytes);
    return val;
  }

  // Drop every entry whose key starts with `prefix` (e.g. a retired segment uid).
  size_t invalidate_prefix(const std::string& prefix) {
    size_t n = 0;
    for (auto& s : shards_) {
      std::lock_guard<std::mutex> lk(s.mu);
      for (auto it = s.lru.begin(); it != s.lru.end();) {
        if (it->key.compare(0, prefix.size(), prefix) == 0) {
          s.bytes -= it->bytes;
          s.map.erase(it->key);
          it = s.lru.erase(it);
          ++n;
        } else {
          ++it;
        }
      }
    }
    invalidations_.fetch_add(n, std::memory_order_relaxed);
    return n;
  }

  void clear() {
    for (auto& s : shards_) {
      std::lock_guard<std::mutex> lk(s.mu);
      s.map.clear();
      s.lru.clear();
      s.bytes = 0;
    }
  }

  Stats stats() const {
    Stats st;
    st.hits = hits_.load();
    st.misses = misses_.load();
    st.evictions = evictions_.load();
    st.invalidations = invalidations_.load();
    st.capacity = capacity_;
    for (auto& s : shards_) {
      std::lock_guard<std::mutex> lk(s.mu);
      st.entries += s.map.size();
      st.bytes += s.bytes;
    }
    return st;
  }

  size_t capacity() const { return capacity_; }

 private:
  struct Entry {
    std::string key;
    std::shared_ptr<const V> value;
    size_t bytes;
  };
  struct Shard {
    mutable std::mutex mu;
    std::list<Entry> lru;
    std::unordered_map<std::string, typename std::list<Entry>::iterator> map;
    size_t bytes = 0;
    size_t capacity = 0;
  };
  Shard& shard(const std::string& key) { return shards_[std::hash<std::string>{}(key) % shards_.size()]; }

  size_t capacity_;
  std::vector<Shard> shards_;
  std::atomic<uint64_t> hits_{0}, misses_{0}, evictions_{0}, invalidations_{0};
};

}  // namespace dc
