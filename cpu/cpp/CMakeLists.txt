cmake_minimum_required(VERSION 3.16)
project(RustVsCpp_CPU_Benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Optimization Flags ──────────────────────────────────────────────────────
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# ── Google Benchmark ─────────────────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# ── OpenMP ───────────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)

# ── Dataset path ─────────────────────────────────────────────────────────────
set(DATASET_DIR "${CMAKE_SOURCE_DIR}/../../datasets" CACHE PATH "Path to datasets")
add_compile_definitions(DATASET_DIR="${DATASET_DIR}")

# ── Include dirs ─────────────────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ── CPU Benchmark Executable ─────────────────────────────────────────────────
add_executable(cpu_benchmarks
    src/quicksort.cpp
    src/mergesort.cpp
    src/matmul.cpp
    src/parallel_reduction.cpp
    src/binary_search.cpp
    src/nbody.cpp
    src/hashmap.cpp
    src/fft.cpp
    src/main.cpp
)

target_link_libraries(cpu_benchmarks
    benchmark::benchmark
    benchmark::benchmark_main
    OpenMP::OpenMP_CXX
)

# ── Standalone executables for each benchmark (optional) ─────────────────────
add_executable(bench_quicksort src/quicksort.cpp src/main.cpp)
target_link_libraries(bench_quicksort benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_mergesort src/mergesort.cpp src/main.cpp)
target_link_libraries(bench_mergesort benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_matmul src/matmul.cpp src/main.cpp)
target_link_libraries(bench_matmul benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_parallel_reduction src/parallel_reduction.cpp src/main.cpp)
target_link_libraries(bench_parallel_reduction benchmark::benchmark benchmark::benchmark_main OpenMP::OpenMP_CXX)

add_executable(bench_binary_search src/binary_search.cpp src/main.cpp)
target_link_libraries(bench_binary_search benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_nbody src/nbody.cpp src/main.cpp)
target_link_libraries(bench_nbody benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_hashmap src/hashmap.cpp src/main.cpp)
target_link_libraries(bench_hashmap benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_fft src/fft.cpp src/main.cpp)
target_link_libraries(bench_fft benchmark::benchmark benchmark::benchmark_main)
