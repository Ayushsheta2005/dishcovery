// Sorted term dictionary with front-coded blocks.
//
// Keys are byte strings: [field id byte][term bytes]. Entries are grouped into
// blocks of 32; within a block each key stores only the suffix that differs
// from its predecessor (front coding), and numeric payloads are varint/delta
// coded. A small block index (first key + offset per block) is held in memory;
// lookup = binary search over block first-keys, then a linear scan of ≤ 32
// entries. The same structure backs both the inverted-index term dictionary
// and the filter-bitmap dictionary.
//
// File layout:
//   [block 0][block 1]...[block n-1]
//   [index: n × (u64 block_offset, u32 key_offset, u32 key_len)]
//   [first-key bytes]
//   [footer: u64 index_offset, u64 keys_offset, u32 num_blocks, u64 num_terms, u32 magic]
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "util/bytes.h"
#include "util/fs.h"

namespace dc::index {

struct TermInfo {
  uint32_t df = 0;         // document frequency (or bitmap cardinality for filters)
  uint64_t post_off = 0;   // offset into the postings (or filters) data file
  uint64_t post_len = 0;   // byte length of the postings
  uint64_t pos_off = 0;    // offset into the positions file
  uint32_t max_tf = 0;     // max term frequency (upper-bound for MaxScore)
  uint32_t min_dl = 0;     // min field length among docs with the term
};

inline std::string make_key(uint8_t field, std::string_view term) {
  std::string k;
  k.reserve(term.size() + 1);
  k.push_back(static_cast<char>(field));
  k.append(term.data(), term.size());
  return k;
}

// Big-endian so byte order == numeric order.
inline std::string make_cell_key(uint8_t field, uint64_t cell) {
  std::string k(9, '\0');
  k[0] = static_cast<char>(field);
  for (int i = 0; i < 8; ++i) k[1 + i] = static_cast<char>((cell >> (56 - 8 * i)) & 0xFF);
  return k;
}

class TermDictWriter {
 public:
  explicit TermDictWriter(std::unique_ptr<OutputStream> out) : out_(std::move(out)) {}
  // Keys must be added in strictly increasing byte order.
  void add(std::string_view key, const TermInfo& info);
  void finish();
  uint64_t num_terms() const { return num_terms_; }

 private:
  void flush_block();
  static constexpr uint32_t kBlockSize = 32;
  std::unique_ptr<OutputStream> out_;
  ByteWriter block_;
  uint32_t in_block_ = 0;
  std::string prev_key_;
  TermInfo prev_info_;
  struct IndexEntry {
    uint64_t offset;
    std::string first_key;
  };
  std::vector<IndexEntry> index_;
  uint64_t num_terms_ = 0;
};

class TermDictReader {
 public:
  TermDictReader() = default;
  explicit TermDictReader(Span data);
  std::optional<TermInfo> lookup(std::string_view key) const;
  uint64_t num_terms() const { return num_terms_; }
  bool empty() const { return num_terms_ == 0; }

  // Sequential iteration in key order (merges, vocabulary export).
  class Iterator {
   public:
    explicit Iterator(const TermDictReader* r);
    bool valid() const { return valid_; }
    void next();
    const std::string& key() const { return key_; }
    const TermInfo& info() const { return info_; }

   private:
    friend class TermDictReader;
    void load_block(uint32_t b);
    const TermDictReader* r_;
    uint32_t block_ = 0;
    uint32_t remaining_in_block_ = 0;
    uint32_t idx_in_block_ = 0;
    ByteReader cur_;
    std::string key_;
    TermInfo info_;
    bool valid_ = false;
  };
  Iterator iterate() const { return Iterator(this); }

  // Iterate keys with a given prefix (e.g. all values of one filter field).
  template <class F>
  void scan_prefix(std::string_view prefix, F&& f) const {
    for (auto it = iterate_from(prefix); it.valid(); it.next()) {
      if (it.key().compare(0, prefix.size(), prefix) != 0) break;
      f(it.key(), it.info());
    }
  }
  Iterator iterate_from(std::string_view key) const;  // first key >= `key`

 private:
  friend class Iterator;
  // Decode one entry given the previous key/info in the block.
  static void decode_entry(ByteReader& r, bool first, std::string& key, TermInfo& info);
  uint32_t find_block(std::string_view key) const;  // last block whose first key <= key

  Span data_;
  uint32_t num_blocks_ = 0;
  uint64_t num_terms_ = 0;
  std::vector<uint64_t> block_offsets_;
  std::vector<std::string_view> first_keys_;
};

}  // namespace dc::index
