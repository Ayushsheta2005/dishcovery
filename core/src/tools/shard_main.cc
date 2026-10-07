// dishcovery-shard: serve one shard replica over HTTP.
//
//   dishcovery-shard --index data/index/shard_0 --schema configs/schema.json \
//       --port 7001 --shard-id 0 --replica r0 [--read-only] [--threads 8]
#include <csignal>
#include <iostream>

#include "index/shard_index.h"
#include "search/searcher.h"
#include "server/shard_server.h"
#include "util/args.h"
#include "util/log.h"

using namespace dc;

static server::ShardServer* g_server = nullptr;

int main(int argc, char** argv) {
  Args a(argc, argv);
  if (a.has("help") || !a.has("index")) {
    std::cerr << "usage: dishcovery-shard --index DIR [--schema FILE] [--port 7001] [--shard-id 0]\n"
                 "         [--replica r0] [--threads 64] [--read-only] [--refresh-ms 200] [--flush-ms 5000]\n"
                 "         [--merge-policy '{\"type\":\"tiered\"}'] [--codec pfor] [--no-wal]\n"
                 "         [--filter-cache-mb 256] [--postings-cache-mb 256] [--build-threads 2]\n";
    return 2;
  }
  if (a.flag("verbose")) log_level().store(static_cast<int>(LogLevel::kDebug));
  try {
    index::Schema schema =
        a.has("schema") ? index::Schema::load(a.str("schema")) : index::Schema::default_schema();
    index::IndexOptions io;
    io.read_only = a.flag("read-only");
    io.refresh_ms = static_cast<uint32_t>(a.integer("refresh-ms", 200));
    io.flush_ms = static_cast<uint32_t>(a.integer("flush-ms", 5000));
    io.codec = codec::codec_from_name(a.str("codec", "pfor"));
    io.wal = !a.flag("no-wal");
    io.build_threads = static_cast<unsigned>(a.integer("build-threads", 2));
    if (a.has("merge-policy")) io.merge_policy = json::parse(a.str("merge-policy"));
    index::ShardIndex idx(a.required("index"), schema, io);
    idx.open();

    search::SearcherOptions so;
    so.filter_cache_bytes = static_cast<size_t>(a.integer("filter-cache-mb", 256)) << 20;
    so.postings_cache_bytes = static_cast<size_t>(a.integer("postings-cache-mb", 256)) << 20;
    so.postings_cache_min_df = static_cast<uint32_t>(a.integer("postings-cache-min-df", 4096));
    search::Searcher searcher(idx.schema(), so);

    server::ShardServerOptions opt;
    opt.host = a.str("host", "0.0.0.0");
    opt.port = static_cast<int>(a.integer("port", 7001));
    opt.threads = static_cast<int>(a.integer("threads", 64));  // one per keep-alive connection
    opt.shard_id = static_cast<uint32_t>(a.integer("shard-id", 0));
    opt.replica = a.str("replica", "r0");
    opt.ctx_ttl_ms = static_cast<uint32_t>(a.integer("ctx-ttl-ms", 60000));
    server::ShardServer srv(idx, searcher, opt);
    g_server = &srv;
    std::signal(SIGINT, [](int) {
      if (g_server) g_server->stop();
    });
    std::signal(SIGTERM, [](int) {
      if (g_server) g_server->stop();
    });
    srv.run();
    LOG_INFO("shutting down");
  } catch (const std::exception& e) {
    LOG_ERROR(e.what());
    return 1;
  }
  return 0;
}
