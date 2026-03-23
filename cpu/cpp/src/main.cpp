/**
 * Main entry point for CPU benchmark suite.
 * Google Benchmark handles main() via BENCHMARK_MAIN().
 * All benchmarks are registered via static initializers in their respective files.
 */

#include <benchmark/benchmark.h>

// Benchmarks auto-register via BENCHMARK() macros in:
//   - quicksort.cpp
//   - mergesort.cpp
//   - matmul.cpp
//   - parallel_reduction.cpp

BENCHMARK_MAIN();
