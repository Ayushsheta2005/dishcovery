#include "server/shard_server.h"

#include <httplib.h>

#include <chrono>
#include <sstream>
#include <thread>

#include "util/log.h"
#include "util/timer.h"

namespace dc::server {

namespace {

void reply_json(httplib::Response& res, std::string body, int status = 200) {
  res.status = status;
  res.set_content(std::move(body), "application/json");
}

void reply_error(httplib::Response& res, int status, const std::string& msg) {
  JsonWriter w;
  w.begin_object().kv("error", msg).end_object();
  reply_json(res, w.take(), status);
}

void write_path(JsonWriter& w, const char* name, const search::PathResult& p) {
  w.key(name).begin_object();
  w.key("hits").begin_array();
  for (const auto& h : p.hits) {
    w.begin_array().value(h.seg).value(h.doc).value(h.score).end_array();
  }
  w.end_array();
  w.kv("scored", p.scored).kv("matched", p.matched).kv("time_us", p.time_us);
  w.end_object();
}

}  // namespace

ShardServer::ShardServer(index::ShardIndex& idx, search::Searcher& searcher, ShardServerOptions opt)
    : idx_(idx), searcher_(searcher), opt_(std::move(opt)), http_(std::make_unique<httplib::Server>()) {}

ShardServer::~ShardServer() = default;

uint64_t ShardServer::put_context(std::shared_ptr<const index::Snapshot> snap) {
  uint64_t id = next_ctx_.fetch_add(1);
  uint64_t now = now_ns();
  std::lock_guard<std::mutex> lk(ctx_mu_);
  if (contexts_.size() >= opt_.max_contexts) {
    for (auto it = contexts_.begin(); it != contexts_.end();) {
      if (it->second.expires_ns < now) it = contexts_.erase(it);
      else ++it;
    }
    // Still full: drop arbitrary entries (oldest ids are most likely stale).
    while (contexts_.size() >= opt_.max_contexts) contexts_.erase(contexts_.begin());
  }
  contexts_[id] = {std::move(snap), now + uint64_t(opt_.ctx_ttl_ms) * 1000000ull};
  return id;
}

std::shared_ptr<const index::Snapshot> ShardServer::get_context(uint64_t id) {
  std::lock_guard<std::mutex> lk(ctx_mu_);
  auto it = contexts_.find(id);
  if (it == contexts_.end() || it->second.expires_ns < now_ns()) return nullptr;
  return it->second.snap;
}

void ShardServer::retire_dropped_segments(const index::Snapshot& snap) {
  std::lock_guard<std::mutex> lk(retire_mu_);
  if (snap.generation == seen_generation_) return;
  std::set<std::string> now;
  for (auto& sv : snap.segments) now.insert(sv.seg->uid());
  for (auto& u : seen_uids_)
    if (!now.count(u)) searcher_.retire_segment(u);  // explicit cache invalidation
  seen_uids_ = std::move(now);
  seen_generation_ = snap.generation;
}

bool ShardServer::maybe_inject_fault() {
  double delay = 0;
  bool error = false;
  {
    std::lock_guard<std::mutex> lk(fault_mu_);
    if (fault_delay_prob_ <= 0 && fault_error_prob_ <= 0) return true;
    std::uniform_real_distribution<double> u(0, 1);
    if (u(rng_) < fault_delay_prob_) delay = fault_delay_ms_;
    error = u(rng_) < fault_error_prob_;
  }
  if (delay > 0) std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int64_t>(delay * 1000)));
  return !error;
}

void ShardServer::run() {
  auto& svr = *http_;
  int threads = opt_.threads;
  svr.new_task_queue = [threads] { return new httplib::ThreadPool(threads); };
  svr.set_keep_alive_max_count(100000);
  // httplib's default also sets SO_REUSEPORT, which lets a second (stale) process
  // bind the same port and silently receive half the traffic. Keep only SO_REUSEADDR
  // so a busy port is a startup error.
  svr.set_socket_options([](socket_t sock) {
    int yes = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof yes);
  });
  svr.set_payload_max_length(512ull << 20);

  svr.set_exception_handler([this](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
    errors_.fetch_add(1);
    std::string msg = "internal error";
    try {
      if (ep) std::rethrow_exception(ep);
    } catch (const std::exception& e) {
      msg = e.what();
    } catch (...) {
    }
    reply_error(res, 500, msg);
  });

  svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
    auto snap = idx_.snapshot();
    JsonWriter w;
    w.begin_object()
        .kv("ok", true)
        .kv("shard", opt_.shard_id)
        .kv("replica", opt_.replica)
        .kv("generation", snap->generation)
        .kv("live_docs", snap->live_docs)
        .kv("segments", static_cast<uint64_t>(snap->segments.size()))
        .end_object();
    reply_json(res, w.take());
  });

  svr.Post("/search", [this](const httplib::Request& req, httplib::Response& res) {
    Stopwatch sw;
    if (!maybe_inject_fault()) return reply_error(res, 503, "injected fault");
    json j = json::parse(req.body);
    auto sreq = search::SearchRequest::from_json(j, idx_.schema());
    auto snap = idx_.snapshot();
    retire_dropped_segments(*snap);
    uint64_t parse_us = sw.elapsed_us();
    auto r = searcher_.search(*snap, sreq);
    uint64_t ctx = put_context(snap);
    JsonWriter w(64 * 1024);
    w.begin_object()
        .kv("shard", opt_.shard_id)
        .kv("replica", opt_.replica)
        .kv("generation", r.generation)
        .kv("ctx", ctx)
        .kv("segments", r.segments)
        .kv("filter_cardinality", r.filter_cardinality)
        .kv("filter_cache_hits", r.filter_cache_hits)
        .kv("filter_mode", search::filter_mode_name(sreq.filter_mode));
    w.key("paths").begin_object();
    if (sreq.bm25) write_path(w, "bm25", r.bm25);
    if (sreq.exact) write_path(w, "exact", r.exact);
    if (sreq.vector) write_path(w, "vector", r.vector);
    if (sreq.geo) write_path(w, "geo", r.geo);
    w.end_object();
    uint64_t total = sw.elapsed_us();
    w.key("timings_us")
        .begin_object()
        .kv("parse", parse_us)
        .kv("filter", r.filter_us)
        .kv("bm25", r.bm25.time_us)
        .kv("exact", r.exact.time_us)
        .kv("vector", r.vector.time_us)
        .kv("geo", r.geo.time_us)
        .kv("search", r.total_us)
        .kv("server", total)
        .end_object();
    w.end_object();
    search_us_.record(total);
    searches_.fetch_add(1);
    reply_json(res, w.take());
  });

  svr.Post("/features", [this](const httplib::Request& req, httplib::Response& res) {
    Stopwatch sw;
    if (!maybe_inject_fault()) return reply_error(res, 503, "injected fault");
    json j = json::parse(req.body);
    auto snap = get_context(j.at("ctx").get<uint64_t>());
    if (!snap) {
      ctx_misses_.fetch_add(1);
      return reply_error(res, 410, "search context expired");
    }
    auto freq = search::FeatureRequest::from_json(j, idx_.schema());
    auto fr = searcher_.features(*snap, freq);
    JsonWriter w(256 * 1024);
    w.begin_object().kv("shard", opt_.shard_id);
    w.key("names").begin_array();
    for (auto& n : fr.names) w.value(n);
    w.end_array();
    w.key("rows").begin_array();
    for (auto& row : fr.rows) {
      w.begin_array();
      for (float v : row) w.value(v);
      w.end_array();
    }
    w.end_array();
    w.key("ids").begin_array();
    for (auto id : fr.ids) w.value(id);
    w.end_array();
    if (freq.include_stored) {
      w.key("docs").begin_array();
      for (auto s : fr.stored) {
        if (s.empty()) w.null();
        else w.raw_json(s);
      }
      w.end_array();
    }
    uint64_t total = sw.elapsed_us();
    w.key("timings_us").begin_object().kv("features", fr.time_us).kv("server", total).end_object();
    w.end_object();
    features_us_.record(total);
    feature_calls_.fetch_add(1);
    reply_json(res, w.take());
  });

  svr.Post("/index", [this](const httplib::Request& req, httplib::Response& res) {
    Stopwatch sw;
    size_t added = 0;
    auto add_one = [&](const json& d) {
      idx_.add(index::parse_document(d, idx_.schema()), d.dump());
      ++added;
    };
    const std::string& body = req.body;
    size_t first = body.find_first_not_of(" \t\r\n");
    if (first != std::string::npos && body[first] == '[') {
      for (const auto& d : json::parse(body)) add_one(d);
    } else {
      std::istringstream in(body);
      std::string line;
      while (std::getline(in, line))
        if (line.find_first_not_of(" \t\r") != std::string::npos) add_one(json::parse(line));
    }
    if (req.has_param("refresh") && req.get_param_value("refresh") == "true") idx_.refresh();
    index_us_.record(sw.elapsed_us());
    JsonWriter w;
    w.begin_object().kv("added", static_cast<uint64_t>(added)).kv("generation", idx_.generation()).end_object();
    reply_json(res, w.take());
  });

  svr.Post("/delete", [this](const httplib::Request& req, httplib::Response& res) {
    json j = json::parse(req.body);
    uint64_t n = 0;
    for (const auto& id : j.at("ids")) n += idx_.remove(id.get<std::string>());
    JsonWriter w;
    w.begin_object().kv("deleted", n).kv("generation", idx_.generation()).end_object();
    reply_json(res, w.take());
  });

  svr.Get("/doc", [this](const httplib::Request& req, httplib::Response& res) {
    std::string id = req.get_param_value("id");
    auto snap = idx_.snapshot();
    for (size_t s = 0; s < snap->segments.size(); ++s) {
      const auto& sv = snap->segments[s];
      auto d = sv.seg->find(id);
      if (!d || sv.is_deleted(*d)) continue;
      JsonWriter w;
      w.begin_object()
          .kv("found", true)
          .kv("segment", sv.seg->uid())
          .kv("ram", !sv.seg->on_disk())
          .kv("generation", snap->generation);
      w.key("doc").raw_json(sv.seg->stored(*d));
      w.end_object();
      return reply_json(res, w.take());
    }
    reply_json(res, "{\"found\":false}", 404);
  });

  svr.Get("/stats", [this](const httplib::Request&, httplib::Response& res) {
    json j = idx_.stats();
    j["shard"] = opt_.shard_id;
    j["replica"] = opt_.replica;
    j["caches"] = searcher_.cache_stats();
    j["endpoints"] = {{"search_us", json::parse(search_us_.to_json())},
                      {"features_us", json::parse(features_us_.to_json())},
                      {"index_us", json::parse(index_us_.to_json())}};
    j["counters"] = {{"searches", searches_.load()},
                     {"feature_calls", feature_calls_.load()},
                     {"ctx_misses", ctx_misses_.load()},
                     {"errors", errors_.load()}};
    {
      std::lock_guard<std::mutex> lk(ctx_mu_);
      j["contexts"] = contexts_.size();
    }
    reply_json(res, j.dump());
  });

  svr.Post("/admin/refresh", [this](const httplib::Request&, httplib::Response& res) {
    idx_.refresh();
    reply_json(res, json{{"generation", idx_.generation()}}.dump());
  });
  svr.Post("/admin/flush", [this](const httplib::Request&, httplib::Response& res) {
    idx_.flush();
    reply_json(res, json{{"generation", idx_.generation()}}.dump());
  });
  svr.Post("/admin/maybe_merge", [this](const httplib::Request&, httplib::Response& res) {
    idx_.maybe_merge();
    reply_json(res, json{{"generation", idx_.generation()}}.dump());
  });
  svr.Post("/admin/merge", [this](const httplib::Request& req, httplib::Response& res) {
    int max_segments = req.has_param("max_segments") ? std::stoi(req.get_param_value("max_segments")) : 1;
    idx_.force_merge(max_segments);
    reply_json(res, json{{"generation", idx_.generation()}}.dump());
  });
  svr.Post("/admin/cache/clear", [this](const httplib::Request&, httplib::Response& res) {
    searcher_.filters().cache().clear();
    reply_json(res, "{\"ok\":true}");
  });
  svr.Post("/admin/fault", [this](const httplib::Request& req, httplib::Response& res) {
    json j = req.body.empty() ? json::object() : json::parse(req.body);
    std::lock_guard<std::mutex> lk(fault_mu_);
    fault_delay_ms_ = j.value("delay_ms", 0.0);
    fault_delay_prob_ = j.value("delay_prob", 0.0);
    fault_error_prob_ = j.value("error_prob", 0.0);
    reply_json(res, json{{"delay_ms", fault_delay_ms_},
                         {"delay_prob", fault_delay_prob_},
                         {"error_prob", fault_error_prob_}}
                        .dump());
  });

  LOG_INFO("shard " << opt_.shard_id << " replica " << opt_.replica << " listening on " << opt_.host << ":"
                    << opt_.port << " (" << opt_.threads << " threads)");
  if (!svr.listen(opt_.host, opt_.port)) fail("cannot listen on port " + std::to_string(opt_.port));
}

void ShardServer::stop() { http_->stop(); }

}  // namespace dc::server
