#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace dc {

class ThreadPool {
 public:
  explicit ThreadPool(unsigned n) {
    if (n == 0) n = 1;
    for (unsigned i = 0; i < n; ++i)
      workers_.emplace_back([this] { loop(); });
  }
  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
  }
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  template <class F>
  auto submit(F&& f) -> std::future<decltype(f())> {
    using R = decltype(f());
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    std::future<R> fut = task->get_future();
    {
      std::lock_guard<std::mutex> lk(mu_);
      q_.emplace([task] { (*task)(); });
    }
    cv_.notify_one();
    return fut;
  }

  size_t size() const { return workers_.size(); }

 private:
  void loop() {
    for (;;) {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return stop_ || !q_.empty(); });
        if (stop_ && q_.empty()) return;
        job = std::move(q_.front());
        q_.pop();
      }
      job();
    }
  }

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> q_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
};

// Run fn(i) for i in [0, n) over `threads` threads (static chunking by atomic counter).
inline void parallel_for(size_t n, unsigned threads, const std::function<void(size_t)>& fn) {
  if (threads <= 1 || n <= 1) {
    for (size_t i = 0; i < n; ++i) fn(i);
    return;
  }
  std::atomic<size_t> next{0};
  std::vector<std::thread> ts;
  std::exception_ptr err;
  std::mutex err_mu;
  for (unsigned t = 0; t < threads; ++t) {
    ts.emplace_back([&] {
      try {
        for (size_t i; (i = next.fetch_add(1)) < n;) fn(i);
      } catch (...) {
        std::lock_guard<std::mutex> lk(err_mu);
        if (!err) err = std::current_exception();
      }
    });
  }
  for (auto& t : ts) t.join();
  if (err) std::rethrow_exception(err);
}

}  // namespace dc
