cmake_minimum_required(VERSION 3.20)
project(dishcovery LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(DC_NATIVE "Compile with -march=native (disable for portable binaries)" ON)
option(DC_SANITIZE "Build with AddressSanitizer + UBSan" OFF)
option(DC_BUILD_TESTS "Build unit tests" ON)
option(DC_BUILD_BENCH "Build benchmarks" ON)

# Some third-party CMakeLists still declare cmake_minimum_required < 3.5, which
# CMake 4 rejects without this.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

find_package(Threads REQUIRED)
include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# --- H3 (hexagonal geo index) ---
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_FILTERS OFF CACHE BOOL "" FORCE)
set(BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
set(BUILD_GENERATORS OFF CACHE BOOL "" FORCE)
set(ENABLE_DOCS OFF CACHE BOOL "" FORCE)
set(ENABLE_COVERAGE OFF CACHE BOOL "" FORCE)
set(ENABLE_LINTING OFF CACHE BOOL "" FORCE)
set(ENABLE_FORMAT OFF CACHE BOOL "" FORCE)
FetchContent_Declare(h3
  URL https://github.com/uber/h3/archive/refs/tags/v4.1.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

# --- utf8proc (Unicode normalisation / case folding) ---
set(UTF8PROC_INSTALL OFF CACHE BOOL "" FORCE)
set(UTF8PROC_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(utf8proc
  URL https://github.com/JuliaStrings/utf8proc/archive/refs/tags/v2.9.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

# --- nlohmann/json ---
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

# --- cpp-httplib (header-only HTTP server for shard / suggest services) ---
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_REQUIRE_OPENSSL OFF CACHE BOOL "" FORCE)
set(HTTPLIB_REQUIRE_ZLIB OFF CACHE BOOL "" FORCE)
set(HTTPLIB_REQUIRE_BROTLI OFF CACHE BOOL "" FORCE)
FetchContent_Declare(httplib
  URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.18.1.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

FetchContent_MakeAvailable(h3 utf8proc json httplib)

add_subdirectory(core)
