// Postings lists: (docID, termFreq, positions[]) with block compression and a
// multi-level skip list.
//
// Per term, in the postings file:
//   df <= 128: one block: [docs][freqs-1]                      (no skip data)
//   df  > 128: [u32 B][u32 L]
//              [level 0: B × (u32 last_doc, u32 block_off, u32 pos_off)]
//              [level l=1..L-1: ceil(B / 8^l) × u32 last_doc of each group]
//              [block 0: docs][freqs-1] ... [block B-1: docs][freqs-1]
// Docs are delta-encoded (first doc of block b relative to block b-1's last
// doc) and every block is compressed with the segment's codec (raw / vbyte /
// pfor). Level-l skip entry g summarises blocks [g·8^l, (g+1)·8^l), so
// advance(target) descends the levels and touches O(log₈ B) entries before
// decoding a single block — sublinear AND-intersection.
//
// Positions (fields with positions only) live in a separate file: for each
// doc, `freq` vbyte-coded position deltas. The level-0 skip entry stores the
// byte offset of each block's positions, so phrase checks never scan
// positions of skipped blocks.
#pragma once

#include <vector>

#include "codec/block_codec.h"
#include "index/term_dict.h"
#include "util/common.h"
#include "util/fs.h"

namespace dc::index {

inline constexpr uint32_t kPostingsBlock = 128;
inline constexpr uint32_t kSkipFanout = 8;

// Term postings accumulated in memory before serialisation.
struct PostingsBuffer {
  std::vector<uint32_t> docs;
  std::vector<uint32_t> freqs;
  std::vector<uint32_t> positions;  // concatenated per doc (absolute positions), empty if not stored
  uint32_t max_tf = 0;
  uint32_t min_dl = UINT32_MAX;
  void clear() {
    docs.clear();
    freqs.clear();
    positions.clear();
    max_tf = 0;
    min_dl = UINT32_MAX;
  }
};

class PostingsWriter {
 public:
  PostingsWriter(OutputStream* postings, OutputStream* positions, codec::CodecId codec)
      : post_(postings), pos_(positions), codec_(codec) {}
  // Serialise one term's postings and return its dictionary entry.
  TermInfo write(const PostingsBuffer& buf, bool with_positions);

 private:
  OutputStream* post_;
  OutputStream* pos_;
  codec::CodecId codec_;
  ByteWriter scratch_;
  ByteWriter blocks_;
  ByteWriter pos_scratch_;
  std::vector<uint32_t> tmp_;
  std::vector<uint32_t> pos_tmp_;
};

class PostingsIterator {
 public:
  PostingsIterator() = default;
  PostingsIterator(Span postings, Span positions, const TermInfo& info, codec::CodecId codec);

  DocId doc() const { return doc_; }
  DocId next();
  DocId advance(DocId target);  // first doc >= target
  uint32_t freq();              // tf of the current doc
  uint32_t df() const { return df_; }
  uint64_t cost() const { return df_; }
  // Positions of the current doc (requires the field to store positions).
  void positions(std::vector<uint32_t>& out);
  bool has_positions() const { return pos_base_ != nullptr; }

  // Instrumentation for the compression / skip-list benchmarks.
  uint64_t blocks_decoded() const { return blocks_decoded_; }

 private:
  void load_block(uint32_t b);
  uint32_t find_block(DocId target) const;  // first block >= cur whose last_doc >= target
  uint32_t level_entry(uint32_t level, uint32_t idx) const;

  const uint8_t* base_ = nullptr;        // start of this term's postings
  const uint8_t* blocks_start_ = nullptr;
  const uint8_t* end_ = nullptr;
  const uint8_t* pos_base_ = nullptr;    // start of this term's positions (nullptr if none)
  const uint8_t* pos_end_ = nullptr;
  codec::CodecId codec_ = codec::CodecId::kPFor;
  uint32_t df_ = 0;
  uint32_t num_blocks_ = 0;
  uint32_t num_levels_ = 0;
  const uint8_t* level0_ = nullptr;      // B × 12 bytes
  std::vector<const uint8_t*> levels_;   // levels 1..L-1
  std::vector<uint32_t> level_sizes_;

  uint32_t block_ = UINT32_MAX;
  uint32_t block_n_ = 0;
  uint32_t idx_ = 0;
  DocId doc_ = kUnpositioned;
  uint32_t docs_[kPostingsBlock];
  uint32_t freqs_[kPostingsBlock];
  const uint8_t* freq_ptr_ = nullptr;
  bool freqs_decoded_ = false;
  // Positions cursor: positions of doc `pos_doc_idx_` in the block start at pos_cur_.
  const uint8_t* pos_block_ = nullptr;
  const uint8_t* pos_cur_ = nullptr;
  uint32_t pos_doc_idx_ = 0;
  uint64_t blocks_decoded_ = 0;
};

}  // namespace dc::index
