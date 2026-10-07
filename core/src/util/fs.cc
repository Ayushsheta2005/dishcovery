#include "util/fs.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace dc {

static std::string errno_str() { return std::strerror(errno); }

MMapFile::MMapFile(const std::string& path, bool populate) : path_(path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) fail("open " + path + ": " + errno_str());
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    fail("fstat " + path + ": " + errno_str());
  }
  size_ = static_cast<size_t>(st.st_size);
  if (size_ > 0) {
    int flags = MAP_SHARED;
#ifdef MAP_POPULATE
    if (populate) flags |= MAP_POPULATE;
#else
    (void)populate;
#endif
    addr_ = ::mmap(nullptr, size_, PROT_READ, flags, fd, 0);
    if (addr_ == MAP_FAILED) {
      ::close(fd);
      addr_ = nullptr;
      fail("mmap " + path + ": " + errno_str());
    }
  }
  ::close(fd);  // the mapping keeps the file alive
}

MMapFile::~MMapFile() {
  if (addr_) ::munmap(addr_, size_);
}

void MMapFile::advise_random() const {
  if (addr_) ::madvise(addr_, size_, MADV_RANDOM);
}
void MMapFile::advise_willneed() const {
  if (addr_) ::madvise(addr_, size_, MADV_WILLNEED);
}

FileOutputStream::FileOutputStream(const std::string& path, size_t buffer_bytes) : path_(path), buf_(buffer_bytes) {
  fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd_ < 0) fail("create " + path + ": " + errno_str());
}

FileOutputStream::~FileOutputStream() {
  if (fd_ >= 0) {
    try {
      close();
    } catch (...) {
    }
  }
}

void FileOutputStream::flush_buffer() {
  size_t off = 0;
  while (off < used_) {
    ssize_t w = ::write(fd_, buf_.data() + off, used_ - off);
    if (w < 0) {
      if (errno == EINTR) continue;
      fail("write " + path_ + ": " + errno_str());
    }
    off += static_cast<size_t>(w);
  }
  used_ = 0;
}

void FileOutputStream::write(const void* p, size_t n) {
  const auto* b = static_cast<const uint8_t*>(p);
  offset_ += n;
  if (n >= buf_.size()) {
    flush_buffer();
    size_t off = 0;
    while (off < n) {
      ssize_t w = ::write(fd_, b + off, n - off);
      if (w < 0) {
        if (errno == EINTR) continue;
        fail("write " + path_ + ": " + errno_str());
      }
      off += static_cast<size_t>(w);
    }
    return;
  }
  if (used_ + n > buf_.size()) flush_buffer();
  std::memcpy(buf_.data() + used_, b, n);
  used_ += n;
}

void FileOutputStream::close() {
  if (fd_ < 0) return;
  flush_buffer();
  ::fsync(fd_);
  ::close(fd_);
  fd_ = -1;
}

DirStorage::DirStorage(std::string dir, bool create_dir) : dir_(std::move(dir)) {
  if (create_dir) ensure_dir(dir_);
}

std::unique_ptr<OutputStream> DirStorage::create(const std::string& name) {
  return std::make_unique<FileOutputStream>(join_path(dir_, name));
}

Span DirStorage::open(const std::string& name) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = maps_.find(name);
  if (it == maps_.end()) {
    std::string p = join_path(dir_, name);
    if (!path_exists(p)) return {};
    it = maps_.emplace(name, std::make_unique<MMapFile>(p)).first;
  }
  return it->second->span();
}

bool DirStorage::exists(const std::string& name) const { return path_exists(join_path(dir_, name)); }

uint64_t DirStorage::total_bytes() const { return dir_size_bytes(dir_); }

void DirStorage::destroy() {
  std::lock_guard<std::mutex> lk(mu_);
  maps_.clear();
  remove_all(dir_);
}

std::unique_ptr<OutputStream> MemoryStorage::create(const std::string& name) {
  auto sink = std::make_shared<std::vector<uint8_t>>();
  {
    std::lock_guard<std::mutex> lk(mu_);
    files_[name] = sink;
  }
  return std::make_unique<MemoryOutputStream>(sink);
}

Span MemoryStorage::open(const std::string& name) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = files_.find(name);
  if (it == files_.end()) return {};
  return {it->second->data(), it->second->size()};
}

bool MemoryStorage::exists(const std::string& name) const {
  std::lock_guard<std::mutex> lk(mu_);
  return files_.count(name) > 0;
}

uint64_t MemoryStorage::total_bytes() const {
  std::lock_guard<std::mutex> lk(mu_);
  uint64_t n = 0;
  for (auto& [k, v] : files_) n += v->size();
  return n;
}

void MemoryStorage::destroy() {
  std::lock_guard<std::mutex> lk(mu_);
  files_.clear();
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("cannot read " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void write_file_atomic(const std::string& path, std::string_view contents) {
  std::string tmp = path + ".tmp";
  {
    FileOutputStream out(tmp, 1 << 16);
    out.write(contents.data(), contents.size());
    out.close();
  }
  if (::rename(tmp.c_str(), path.c_str()) != 0) fail("rename " + tmp + ": " + errno_str());
  // fsync the directory so the rename is durable.
  std::string dir = fs::path(path).parent_path().string();
  if (dir.empty()) dir = ".";
  int dfd = ::open(dir.c_str(), O_RDONLY);
  if (dfd >= 0) {
    ::fsync(dfd);
    ::close(dfd);
  }
}

void ensure_dir(const std::string& path) {
  std::error_code ec;
  fs::create_directories(path, ec);
  if (ec) fail("mkdir " + path + ": " + ec.message());
}

bool path_exists(const std::string& path) { return fs::exists(path); }

void remove_all(const std::string& path) {
  std::error_code ec;
  fs::remove_all(path, ec);
}

std::vector<std::string> list_dir(const std::string& path) {
  std::vector<std::string> out;
  std::error_code ec;
  for (auto& e : fs::directory_iterator(path, ec)) out.push_back(e.path().filename().string());
  std::sort(out.begin(), out.end());
  return out;
}

uint64_t dir_size_bytes(const std::string& path) {
  uint64_t n = 0;
  std::error_code ec;
  for (auto& e : fs::recursive_directory_iterator(path, ec))
    if (e.is_regular_file(ec)) n += e.file_size(ec);
  return n;
}

std::string join_path(const std::string& a, const std::string& b) { return (fs::path(a) / b).string(); }

}  // namespace dc
