#include "index/postings.h"

#include <algorithm>
#include <cstring>

#include "codec/vbyte.h"

namespace dc::index {

TermInfo PostingsWriter::write(const PostingsBuffer& buf, bool with_positions) {
  TermInfo info;
  const uint32_t df = static_cast<uint32_t>(buf.docs.size());
  DC_CHECK(df > 0, "empty postings");
  DC_CHECK(buf.freqs.size() == df, "freqs/docs size mismatch");
  info.df = df;
  info.max_tf = buf.max_tf;
  info.min_dl = buf.min_dl == UINT32_MAX ? 0 : buf.min_dl;
  info.post_off = post_->offset();
  info.pos_off = pos_ ? pos_->offset() : 0;

  const uint32_t num_blocks = (df + kPostingsBlock - 1) / kPostingsBlock;
  std::vector<uint32_t> last_doc(num_blocks), block_off(num_blocks), pos_off(num_blocks);

  blocks_.clear();
  pos_scratch_.clear();
  size_t pos_cursor = 0;  // index into buf.positions
  tmp_.resize(kPostingsBlock);
  for (uint32_t b = 0; b < num_blocks; ++b) {
    uint32_t lo = b * kPostingsBlock, hi = std::min(df, lo + kPostingsBlock);
    uint32_t n = hi - lo;
    block_off[b] = static_cast<uint32_t>(blocks_.size());
    pos_off[b] = static_cast<uint32_t>(pos_scratch_.size());
    last_doc[b] = buf.docs[hi - 1];
    uint32_t base = b ? buf.docs[lo - 1] : 0;
    std::memcpy(tmp_.data(), buf.docs.data() + lo, n * sizeof(uint32_t));
    codec::delta_encode(tmp_.data(), n, base);
    codec::encode_block(codec_, tmp_.data(), n, blocks_);
    for (uint32_t i = 0; i < n; ++i) tmp_[i] = buf.freqs[lo + i] - 1;
    codec::encode_block(codec_, tmp_.data(), n, blocks_);
    if (with_positions) {
      for (uint32_t i = lo; i < hi; ++i) {
        uint32_t f = buf.freqs[i];
        pos_tmp_.resize(f);
        uint32_t prev = 0;
        for (uint32_t k = 0; k < f; ++k) {
          uint32_t p = buf.positions[pos_cursor + k];
          pos_tmp_[k] = p - prev;
          prev = p;
        }
        codec::vbyte_encode(pos_tmp_.data(), f, pos_scratch_);
        pos_cursor += f;
      }
    }
  }

  scratch_.clear();
  if (num_blocks > 1) {
    // Skip-list levels: level 0 is per block; level l has one entry per 8^l blocks.
    std::vector<std::vector<uint32_t>> upper;
    uint64_t span = kSkipFanout;
    while (span < num_blocks) {
      std::vector<uint32_t> lvl;
      for (uint64_t g = 0; g * span < num_blocks; ++g) {
        uint64_t last = std::min<uint64_t>((g + 1) * span, num_blocks) - 1;
        lvl.push_back(last_doc[last]);
      }
      upper.push_back(std::move(lvl));
      span *= kSkipFanout;
    }
    scratch_.put_u32(num_blocks);
    scratch_.put_u32(static_cast<uint32_t>(1 + upper.size()));
    for (uint32_t b = 0; b < num_blocks; ++b) {
      scratch_.put_u32(last_doc[b]);
      scratch_.put_u32(block_off[b]);
      scratch_.put_u32(pos_off[b]);
    }
    for (auto& lvl : upper)
      for (uint32_t v : lvl) scratch_.put_u32(v);
  }
  post_->write(scratch_.data(), scratch_.size());
  post_->write(blocks_.data(), blocks_.size());
  info.post_len = scratch_.size() + blocks_.size();
  if (with_positions && pos_) pos_->write(pos_scratch_.data(), pos_scratch_.size());
  return info;
}

PostingsIterator::PostingsIterator(Span postings, Span positions, const TermInfo& info, codec::CodecId codec)
    : codec_(codec), df_(info.df) {
  DC_CHECK(info.post_off + info.post_len <= postings.size, "postings out of range");
  base_ = postings.data + info.post_off;
  end_ = base_ + info.post_len;
  if (!positions.empty() && info.pos_off < positions.size) {
    pos_base_ = positions.data + info.pos_off;
    pos_end_ = positions.data + positions.size;
  }
  if (df_ > kPostingsBlock) {
    ByteReader r(base_, info.post_len);
    num_blocks_ = r.get_u32();
    num_levels_ = r.get_u32();
    level0_ = r.get_bytes(num_blocks_ * 12ull);
    uint64_t span = kSkipFanout;
    for (uint32_t l = 1; l < num_levels_; ++l) {
      uint32_t sz = static_cast<uint32_t>((num_blocks_ + span - 1) / span);
      levels_.push_back(r.get_bytes(sz * 4ull));
      level_sizes_.push_back(sz);
      span *= kSkipFanout;
    }
    blocks_start_ = r.cursor();
  } else {
    num_blocks_ = 1;
    num_levels_ = 0;
    blocks_start_ = base_;
  }
}

void PostingsIterator::load_block(uint32_t b) {
  block_ = b;
  uint32_t lo = b * kPostingsBlock;
  block_n_ = std::min(df_ - lo, kPostingsBlock);
  const uint8_t* p;
  uint32_t base;
  const uint8_t* pos_block = pos_base_;
  if (level0_) {
    p = blocks_start_ + load_u32(level0_ + b * 12ull + 4);
    base = b ? load_u32(level0_ + (b - 1) * 12ull) : 0;
    if (pos_base_) pos_block = pos_base_ + load_u32(level0_ + b * 12ull + 8);
  } else {
    p = blocks_start_;
    base = 0;
  }
  freq_ptr_ = codec::decode_block(codec_, p, block_n_, docs_);
  codec::prefix_sum(docs_, block_n_, base);
  freqs_decoded_ = false;
  pos_block_ = pos_block;
  pos_cur_ = pos_block;
  pos_doc_idx_ = 0;
  idx_ = 0;
  ++blocks_decoded_;
}

DocId PostingsIterator::next() {
  if (DC_UNLIKELY(doc_ == kNoMoreDocs)) return doc_;
  if (block_ == UINT32_MAX) {
    load_block(0);
    return doc_ = docs_[0];
  }
  if (++idx_ < block_n_) return doc_ = docs_[idx_];
  if (block_ + 1 >= num_blocks_) return doc_ = kNoMoreDocs;
  load_block(block_ + 1);
  return doc_ = docs_[0];
}

uint32_t PostingsIterator::level_entry(uint32_t level, uint32_t idx) const {
  if (level == 0) return load_u32(level0_ + idx * 12ull);
  return load_u32(levels_[level - 1] + idx * 4ull);
}

uint32_t PostingsIterator::find_block(DocId target) const {
  uint32_t b = block_ == UINT32_MAX ? 0 : block_;
  // Descend from the top level: skip whole groups whose last doc < target.
  uint64_t span = 1;
  for (uint32_t l = 1; l < num_levels_; ++l) span *= kSkipFanout;
  for (uint32_t l = num_levels_ - 1; l >= 1; --l) {
    uint32_t g = static_cast<uint32_t>(b / span);
    uint32_t sz = level_sizes_[l - 1];
    while (g < sz && level_entry(l, g) < target) {
      ++g;
      b = static_cast<uint32_t>(g * span);
    }
    if (b >= num_blocks_) return num_blocks_;
    span /= kSkipFanout;
  }
  while (b < num_blocks_ && level_entry(0, b) < target) ++b;
  return b;
}

DocId PostingsIterator::advance(DocId target) {
  if (DC_UNLIKELY(doc_ == kNoMoreDocs)) return doc_;
  if (block_ != UINT32_MAX && doc_ >= target) return doc_;
  if (block_ == UINT32_MAX) {
    if (!level0_) {
      load_block(0);
    } else {
      uint32_t b = find_block(target);
      if (b >= num_blocks_) return doc_ = kNoMoreDocs;
      load_block(b);
    }
  } else if (docs_[block_n_ - 1] < target) {
    if (!level0_) return doc_ = kNoMoreDocs;
    uint32_t b = find_block(target);
    if (b >= num_blocks_) return doc_ = kNoMoreDocs;
    load_block(b);
  }
  // Within-block search: galloping would help for long blocks, but 128
  // entries fit in two cache lines of deltas so a linear scan is fine.
  while (idx_ < block_n_ && docs_[idx_] < target) ++idx_;
  if (idx_ >= block_n_) {  // only possible for the last block
    return doc_ = kNoMoreDocs;
  }
  return doc_ = docs_[idx_];
}

uint32_t PostingsIterator::freq() {
  if (!freqs_decoded_) {
    codec::decode_block(codec_, freq_ptr_, block_n_, freqs_);
    freqs_decoded_ = true;
  }
  return freqs_[idx_] + 1;
}

void PostingsIterator::positions(std::vector<uint32_t>& out) {
  out.clear();
  if (!pos_base_ || doc_ == kNoMoreDocs || doc_ == kUnpositioned) return;
  if (!freqs_decoded_) freq();
  if (idx_ < pos_doc_idx_) {  // never happens for forward iteration, but stay safe
    pos_cur_ = pos_block_;
    pos_doc_idx_ = 0;
  }
  while (pos_doc_idx_ < idx_) {
    pos_cur_ = codec::vbyte_skip(pos_cur_, freqs_[pos_doc_idx_] + 1);
    ++pos_doc_idx_;
  }
  uint32_t f = freqs_[idx_] + 1;
  out.resize(f);
  codec::vbyte_decode(pos_cur_, f, out.data());
  uint32_t acc = 0;
  for (uint32_t i = 0; i < f; ++i) {
    acc += out[i];
    out[i] = acc;
  }
}

}  // namespace dc::index
