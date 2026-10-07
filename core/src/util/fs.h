// File-system helpers: mmap'd read-only files, buffered output streams that
// target either disk or memory, and atomic file replacement.
#pragma once

#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "util/common.h"

namespace dc {

// A borrowed byte range (from an mmap or an in-memory buffer).
struct Span {
  const uint8_t* data = nullptr;
  size_t size = 0;
  bool empty() const { return size == 0; }
};

// Read-only memory map of a whole file. Empty files map to an empty span.
class MMapFile {
 public:
  explicit MMapFile(const std::string& path, bool populate = false);
  ~MMapFile();
  MMapFile(const MMapFile&) = delete;
  MMapFile& operator=(const MMapFile&) = delete;
  Span span() const { return {static_cast<const uint8_t*>(addr_), size_}; }
  size_t size() const { return size_; }
  const std::string& path() const { return path_; }
  void advise_random() const;
  void advise_willneed() const;

 private:
  std::string path_;
  void* addr_ = nullptr;
  size_t size_ = 0;
};

// Append-only output stream. Implementations write to a file or to memory.
class OutputStream {
 public:
  virtual ~OutputStream() = default;
  virtual void write(const void* p, size_t n) = 0;
  virtual uint64_t offset() const = 0;
  virtual void close() = 0;  // flush + fsync for files
  void write_vec(const std::vector<uint8_t>& v) { write(v.data(), v.size()); }
  // Pad with zeros to a multiple of `align` bytes.
  void align_to(size_t align) {
    static const uint8_t zeros[64] = {};
    size_t pad = (align - offset() % align) % align;
    while (pad) {
      size_t n = pad < sizeof zeros ? pad : sizeof zeros;
      write(zeros, n);
      pad -= n;
    }
  }
};

class FileOutputStream : public OutputStream {
 public:
  explicit FileOutputStream(const std::string& path, size_t buffer_bytes = 1 << 20);
  ~FileOutputStream() override;
  void write(const void* p, size_t n) override;
  uint64_t offset() const override { return offset_; }
  void close() override;

 private:
  void flush_buffer();
  std::string path_;
  int fd_ = -1;
  std::vector<uint8_t> buf_;
  size_t used_ = 0;
  uint64_t offset_ = 0;
};

class MemoryOutputStream : public OutputStream {
 public:
  explicit MemoryOutputStream(std::shared_ptr<std::vector<uint8_t>> sink) : sink_(std::move(sink)) {}
  void write(const void* p, size_t n) override {
    const auto* b = static_cast<const uint8_t*>(p);
    sink_->insert(sink_->end(), b, b + n);
  }
  uint64_t offset() const override { return sink_->size(); }
  void close() override {}

 private:
  std::shared_ptr<std::vector<uint8_t>> sink_;
};

// A named set of files for one segment: either a directory on disk or an
// in-memory map (used for near-real-time RAM segments).
class SegmentStorage {
 public:
  virtual ~SegmentStorage() = default;
  virtual std::unique_ptr<OutputStream> create(const std::string& name) = 0;
  virtual Span open(const std::string& name) = 0;  // empty span if missing
  virtual bool exists(const std::string& name) const = 0;
  virtual uint64_t total_bytes() const = 0;
  virtual bool on_disk() const = 0;
  virtual std::string location() const = 0;
  // Remove every file (called when an obsolete segment is finally released).
  virtual void destroy() = 0;
};

class DirStorage : public SegmentStorage {
 public:
  explicit DirStorage(std::string dir, bool create_dir);
  std::unique_ptr<OutputStream> create(const std::string& name) override;
  Span open(const std::string& name) override;
  bool exists(const std::string& name) const override;
  uint64_t total_bytes() const override;
  bool on_disk() const override { return true; }
  std::string location() const override { return dir_; }
  void destroy() override;
  const std::string& dir() const { return dir_; }

 private:
  std::string dir_;
  std::mutex mu_;
  std::map<std::string, std::unique_ptr<MMapFile>> maps_;
};

class MemoryStorage : public SegmentStorage {
 public:
  std::unique_ptr<OutputStream> create(const std::string& name) override;
  Span open(const std::string& name) override;
  bool exists(const std::string& name) const override;
  uint64_t total_bytes() const override;
  bool on_disk() const override { return false; }
  std::string location() const override { return "<memory>"; }
  void destroy() override;

 private:
  mutable std::mutex mu_;
  std::map<std::string, std::shared_ptr<std::vector<uint8_t>>> files_;
};

// Filesystem utilities.
std::string read_file(const std::string& path);
void write_file_atomic(const std::string& path, std::string_view contents);  // tmp + fsync + rename
void ensure_dir(const std::string& path);
bool path_exists(const std::string& path);
void remove_all(const std::string& path);
std::vector<std::string> list_dir(const std::string& path);
uint64_t dir_size_bytes(const std::string& path);
std::string join_path(const std::string& a, const std::string& b);

}  // namespace dc
