// Little-endian byte serialization helpers shared by every on-disk structure.
//
// ByteWriter appends to a growable buffer; ByteReader is a bounds-checked
// cursor over a borrowed span (typically an mmap'd segment file).
#pragma once

#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "util/common.h"

namespace dc {

class ByteWriter {
 public:
  ByteWriter() = default;
  explicit ByteWriter(size_t reserve) { buf_.reserve(reserve); }

  void put_u8(uint8_t v) { buf_.push_back(v); }
  void put_u16(uint16_t v) { put_raw(&v, sizeof v); }
  void put_u32(uint32_t v) { put_raw(&v, sizeof v); }
  void put_u64(uint64_t v) { put_raw(&v, sizeof v); }
  void put_i32(int32_t v) { put_raw(&v, sizeof v); }
  void put_f32(float v) { put_raw(&v, sizeof v); }
  void put_f64(double v) { put_raw(&v, sizeof v); }

  // LEB128-style variable-length unsigned integer (7 bits per byte, high bit
  // = continuation).
  void put_varint(uint64_t v) {
    while (v >= 0x80) {
      buf_.push_back(static_cast<uint8_t>(v) | 0x80);
      v >>= 7;
    }
    buf_.push_back(static_cast<uint8_t>(v));
  }

  void put_bytes(const void* p, size_t n) { put_raw(p, n); }
  // Length-prefixed (varint) string.
  void put_string(std::string_view s) {
    put_varint(s.size());
    put_raw(s.data(), s.size());
  }

  // Pad with zeros until size() is a multiple of `align`.
  void align_to(size_t align) {
    while (buf_.size() % align) buf_.push_back(0);
  }

  // Overwrite a previously written u32/u64 (for back-patching offsets).
  void patch_u32(size_t at, uint32_t v) {
    DC_CHECK(at + 4 <= buf_.size(), "patch out of range");
    std::memcpy(buf_.data() + at, &v, 4);
  }
  void patch_u64(size_t at, uint64_t v) {
    DC_CHECK(at + 8 <= buf_.size(), "patch out of range");
    std::memcpy(buf_.data() + at, &v, 8);
  }

  size_t size() const { return buf_.size(); }
  const uint8_t* data() const { return buf_.data(); }
  uint8_t* mutable_data() { return buf_.data(); }
  std::vector<uint8_t>& buffer() { return buf_; }
  const std::vector<uint8_t>& buffer() const { return buf_; }
  std::vector<uint8_t> release() { return std::move(buf_); }
  void clear() { buf_.clear(); }
  void reserve(size_t n) { buf_.reserve(n); }

  // Grow by n bytes and return a pointer to the new region (for codecs that
  // write in place).
  uint8_t* extend(size_t n) {
    size_t old = buf_.size();
    buf_.resize(old + n);
    return buf_.data() + old;
  }
  void truncate(size_t n) { buf_.resize(n); }

 private:
  void put_raw(const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    buf_.insert(buf_.end(), b, b + n);
  }
  std::vector<uint8_t> buf_;
};

class ByteReader {
 public:
  ByteReader() = default;
  ByteReader(const uint8_t* data, size_t size) : begin_(data), p_(data), end_(data + size) {}
  explicit ByteReader(const std::vector<uint8_t>& v) : ByteReader(v.data(), v.size()) {}

  uint8_t get_u8() {
    need(1);
    return *p_++;
  }
  uint16_t get_u16() { return get_pod<uint16_t>(); }
  uint32_t get_u32() { return get_pod<uint32_t>(); }
  uint64_t get_u64() { return get_pod<uint64_t>(); }
  int32_t get_i32() { return get_pod<int32_t>(); }
  float get_f32() { return get_pod<float>(); }
  double get_f64() { return get_pod<double>(); }

  uint64_t get_varint() {
    uint64_t v = 0;
    int shift = 0;
    while (true) {
      need(1);
      uint8_t b = *p_++;
      v |= static_cast<uint64_t>(b & 0x7F) << shift;
      if (!(b & 0x80)) break;
      shift += 7;
      DC_CHECK(shift < 64, "varint too long");
    }
    return v;
  }

  std::string_view get_string() {
    size_t n = static_cast<size_t>(get_varint());
    need(n);
    std::string_view s(reinterpret_cast<const char*>(p_), n);
    p_ += n;
    return s;
  }

  // Borrow n bytes without copying.
  const uint8_t* get_bytes(size_t n) {
    need(n);
    const uint8_t* r = p_;
    p_ += n;
    return r;
  }

  void skip(size_t n) {
    need(n);
    p_ += n;
  }
  void align_to(size_t align) {
    size_t off = offset();
    size_t pad = (align - off % align) % align;
    skip(pad);
  }
  void seek(size_t off) {
    DC_CHECK(begin_ + off <= end_, "seek out of range");
    p_ = begin_ + off;
  }

  size_t offset() const { return static_cast<size_t>(p_ - begin_); }
  size_t remaining() const { return static_cast<size_t>(end_ - p_); }
  bool eof() const { return p_ >= end_; }
  const uint8_t* cursor() const { return p_; }
  const uint8_t* base() const { return begin_; }
  size_t size() const { return static_cast<size_t>(end_ - begin_); }

 private:
  template <typename T>
  T get_pod() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, p_, sizeof(T));
    p_ += sizeof(T);
    return v;
  }
  void need(size_t n) const {
    if (DC_UNLIKELY(static_cast<size_t>(end_ - p_) < n)) fail("ByteReader: read past end of buffer");
  }

  const uint8_t* begin_ = nullptr;
  const uint8_t* p_ = nullptr;
  const uint8_t* end_ = nullptr;
};

// Unchecked little-endian loads for hot loops over validated buffers.
inline uint32_t load_u32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
inline uint64_t load_u64(const uint8_t* p) {
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}
inline float load_f32(const uint8_t* p) {
  float v;
  std::memcpy(&v, p, 4);
  return v;
}

}  // namespace dc
