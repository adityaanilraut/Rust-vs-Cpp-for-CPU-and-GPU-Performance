/**
 * Matrix Multiplication Benchmark — C++
 * ======================================
 * Naive triple-loop dense matrix multiplication (f64).
 * No BLAS — ensures apples-to-apples language comparison.
 * Sizes: 512×512, 1024×1024, 2048×2048, 4096×4096.
 */

#include <benchmark/benchmark.h>
#include <cstdint>
#include <vector>

#include "benchmark_utils.h"

// ── Naive Matrix Multiplication ─────────────────────────────────────────────

namespace {

/**
 * C = A × B (naive O(n³) triple-loop)
 * Row-major layout. No vectorization hints — let the compiler optimize.
 */
void matmul_naive(const double *__restrict__ A, const double *__restrict__ B,
                  double *__restrict__ C, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k < n; ++k) {
      double a_ik = A[i * n + k];
      for (size_t j = 0; j < n; ++j) {
        C[i * n + j] += a_ik * B[k * n + j];
      }
    }
  }
}

/**
 * Compute GFLOPS for matrix multiplication.
 * Standard matmul: 2*n³ floating-point operations.
 */
inline double compute_gflops(size_t n, double time_seconds) {
  double flops = 2.0 * static_cast<double>(n) * n * n;
  return flops / (time_seconds * 1e9);
}

} // anonymous namespace

// ── Benchmark ───────────────────────────────────────────────────────────────

static void BM_MatMul(benchmark::State &state) {
  const size_t n = state.range(0);

  // Generate random matrices
  auto A = bench_utils::generate_random_array<double>(n * n, 42);
  auto B = bench_utils::generate_random_array<double>(n * n, 43);
  std::vector<double> C(n * n, 0.0);

  for (auto _ : state) {
    // Zero output matrix each iteration
    std::fill(C.begin(), C.end(), 0.0);
    matmul_naive(A.data(), B.data(), C.data(), n);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  // Report GFLOPS (timing derived by the benchmark library from iterations)
  state.counters["GFLOPS"] = benchmark::Counter(
      2.0 * n * n * n, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
  state.SetLabel("naive_triple_loop");
}

// ── Register ────────────────────────────────────────────────────────────────

BENCHMARK(BM_MatMul)
    ->Unit(benchmark::kMillisecond)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    // 4096 is very slow with naive — run separately if needed
    // ->Arg(4096)
    ->MinTime(bench_utils::cpu_min_time_seconds())
    ->Repetitions(bench_utils::cpu_repetitions(5))
    ->ReportAggregatesOnly(true)
    ->DisplayAggregatesOnly(true);

// Separate registration for small sizes with full 30 reps
BENCHMARK(BM_MatMul)
    ->Name("BM_MatMul_Full30")
    ->Unit(benchmark::kMillisecond)
    ->Arg(512)
    ->MinTime(bench_utils::cpu_min_time_seconds())
    ->Repetitions(bench_utils::cpu_repetitions(30))
    ->ReportAggregatesOnly(true)
    ->DisplayAggregatesOnly(true);
