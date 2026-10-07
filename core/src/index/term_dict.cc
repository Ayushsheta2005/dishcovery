#include "index/term_dict.h"

#include <algorithm>
#include <cstring>

#include "util/common.h"

namespace dc::index {

namespace {
constexpr uint32_t kMagic = 0x44435444;  // "DTCD"
constexpr size_t kFooterBytes = 8 + 8 + 4 + 8 + 4;

size_t shared_prefix(std::string_view a, std::string_view b) {
  size_t n = std::min(a.size(), b.size());
  size_t i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return i;
}
}  // namespace

void TermDictWriter::add(std::string_view key, const TermInfo& info) {
  DC_CHECK(num_terms_ == 0 || key > std::string_view(prev_key_), "term dict keys must be strictly increasing");
  bool first = in_block_ == 0;
  if (first) {
    index_.push_back({0, std::string(key)});  // offset patched in flush_block
    block_.put_u16(0);                         // entry count, patched in flush_block
  }
  size_t shared = first ? 0 : shared_prefix(prev_key_, key);
  block_.put_varint(shared);
  block_.put_varint(key.size() - shared);
  block_.put_bytes(key.data() + shared, key.size() - shared);
  block_.put_varint(info.df);
  if (first) {
    block_.put_varint(info.post_off);
    block_.put_varint(info.pos_off);
  } else {
    // Postings are written in key order, so offsets increase within a block.
    block_.put_varint(info.post_off - prev_info_.post_off);
    block_.put_varint(info.pos_off - prev_info_.pos_off);
  }
  block_.put_varint(info.post_len);
  block_.put_varint(info.max_tf);
  block_.put_varint(info.min_dl);
  prev_key_.assign(key.data(), key.size());
  prev_info_ = info;
  ++num_terms_;
  if (++in_block_ == kBlockSize) flush_block();
}

void TermDictWriter::flush_block() {
  if (in_block_ == 0) return;
  index_.back().offset = out_->offset();
  uint16_t n = static_cast<uint16_t>(in_block_);
  std::memcpy(block_.mutable_data(), &n, 2);
  out_->write(block_.data(), block_.size());
  block_.clear();
  in_block_ = 0;
}

void TermDictWriter::finish() {
  flush_block();
  uint64_t index_offset = out_->offset();
  ByteWriter idx;
  uint32_t key_off = 0;
  for (auto& e : index_) {
    idx.put_u64(e.offset);
    idx.put_u32(key_off);
    idx.put_u32(static_cast<uint32_t>(e.first_key.size()));
    key_off += static_cast<uint32_t>(e.first_key.size());
  }
  out_->write(idx.data(), idx.size());
  uint64_t keys_offset = out_->offset();
  for (auto& e : index_) out_->write(e.first_key.data(), e.first_key.size());
  ByteWriter footer;
  footer.put_u64(index_offset);
  footer.put_u64(keys_offset);
  footer.put_u32(static_cast<uint32_t>(index_.size()));
  footer.put_u64(num_terms_);
  footer.put_u32(kMagic);
  out_->write(footer.data(), footer.size());
  out_->close();
}

TermDictReader::TermDictReader(Span data) : data_(data) {
  if (data.size == 0) return;
  DC_CHECK(data.size >= kFooterBytes, "term dict too small");
  ByteReader f(data.data + data.size - kFooterBytes, kFooterBytes);
  uint64_t index_offset = f.get_u64();
  uint64_t keys_offset = f.get_u64();
  num_blocks_ = f.get_u32();
  num_terms_ = f.get_u64();
  DC_CHECK(f.get_u32() == kMagic, "bad term dict magic");
  block_offsets_.resize(num_blocks_);
  first_keys_.resize(num_blocks_);
  ByteReader idx(data.data + index_offset, num_blocks_ * 16ull);
  const char* keys = reinterpret_cast<const char*>(data.data + keys_offset);
  for (uint32_t b = 0; b < num_blocks_; ++b) {
    block_offsets_[b] = idx.get_u64();
    uint32_t ko = idx.get_u32(), kl = idx.get_u32();
    first_keys_[b] = std::string_view(keys + ko, kl);
  }
}

void TermDictReader::decode_entry(ByteReader& r, bool first, std::string& key, TermInfo& info) {
  size_t shared = r.get_varint();
  size_t suffix = r.get_varint();
  key.resize(shared);
  const uint8_t* s = r.get_bytes(suffix);
  key.append(reinterpret_cast<const char*>(s), suffix);
  info.df = static_cast<uint32_t>(r.get_varint());
  if (first) {
    info.post_off = r.get_varint();
    info.pos_off = r.get_varint();
  } else {
    info.post_off += r.get_varint();
    info.pos_off += r.get_varint();
  }
  info.post_len = r.get_varint();
  info.max_tf = static_cast<uint32_t>(r.get_varint());
  info.min_dl = static_cast<uint32_t>(r.get_varint());
}

uint32_t TermDictReader::find_block(std::string_view key) const {
  // upper_bound over first keys, then step back one block.
  auto it = std::upper_bound(first_keys_.begin(), first_keys_.end(), key);
  if (it == first_keys_.begin()) return UINT32_MAX;
  return static_cast<uint32_t>((it - first_keys_.begin()) - 1);
}

std::optional<TermInfo> TermDictReader::lookup(std::string_view key) const {
  if (num_blocks_ == 0) return std::nullopt;
  uint32_t b = find_block(key);
  if (b == UINT32_MAX) return std::nullopt;
  ByteReader r(data_.data + block_offsets_[b], data_.size - block_offsets_[b]);
  uint16_t n = r.get_u16();
  std::string cur;
  TermInfo info;
  for (uint16_t i = 0; i < n; ++i) {
    decode_entry(r, i == 0, cur, info);
    int c = std::string_view(cur).compare(key);
    if (c == 0) return info;
    if (c > 0) break;
  }
  return std::nullopt;
}

TermDictReader::Iterator::Iterator(const TermDictReader* r) : r_(r) {
  if (r_->num_blocks_ > 0) {
    load_block(0);
    next();
  }
}

void TermDictReader::Iterator::load_block(uint32_t b) {
  block_ = b;
  cur_ = ByteReader(r_->data_.data + r_->block_offsets_[b], r_->data_.size - r_->block_offsets_[b]);
  remaining_in_block_ = cur_.get_u16();
  idx_in_block_ = 0;
}

void TermDictReader::Iterator::next() {
  if (remaining_in_block_ == 0) {
    if (block_ + 1 >= r_->num_blocks_) {
      valid_ = false;
      return;
    }
    load_block(block_ + 1);
  }
  decode_entry(cur_, idx_in_block_ == 0, key_, info_);
  ++idx_in_block_;
  --remaining_in_block_;
  valid_ = true;
}

TermDictReader::Iterator TermDictReader::iterate_from(std::string_view key) const {
  Iterator it(this);
  if (num_blocks_ == 0) return it;
  uint32_t b = find_block(key);
  if (b != UINT32_MAX && b != 0) {
    it.load_block(b);
    it.next();
  }
  while (it.valid() && std::string_view(it.key()) < key) it.next();
  return it;
}

}  // namespace dc::index
