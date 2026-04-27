/**
 * Parallel Vector Reduction Benchmark — C++
 * ==========================================
 * Parallel summation of f64 vectors using OpenMP.
 * Input sizes: 1M, 10M, 100M elements.
 * Thread counts matched to system/Rayon defaults for fair comparison.
 */

#include <benchmark/benchmark.h>
#include <cstdint>
#include <numeric>
#include <omp.h>
#include <vector>

#include "benchmark_utils.h"

// ── Sequential Reduction (Baseline) ─────────────────────────────────────────

namespace {

double sequential_reduce(const double* data, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

// ── OpenMP Parallel Reduction ───────────────────────────────────────────────

double parallel_reduce_omp(const double* data, size_t n) {
    double sum = 0.0;
    const long long len = static_cast<long long>(n);
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long long i = 0; i < len; ++i) {
        sum += data[i];
    }
    return sum;
}

} // anonymous namespace

// ── Benchmark: Sequential ───────────────────────────────────────────────────

static void BM_SequentialReduction(benchmark::State& state) {
    const size_t n = state.range(0);
    auto data = bench_utils::generate_random_array<double>(n);
    
    for (auto _ : state) {
        double result = sequential_reduce(data.data(), n);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * n * sizeof(double));
    state.SetLabel("sequential");
}

// ── Benchmark: OpenMP Parallel ──────────────────────────────────────────────

static void BM_ParallelReduction(benchmark::State& state) {
    const size_t n = state.range(0);
    auto data = bench_utils::generate_random_array<double>(n);
    
    // Report thread count
    int num_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    
    for (auto _ : state) {
        double result = parallel_reduce_omp(data.data(), n);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * n * sizeof(double));
    state.SetLabel("openmp_" + std::to_string(num_threads) + "threads");
}

// ── Register ────────────────────────────────────────────────────────────────

BENCHMARK(BM_SequentialReduction)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1'000'000)
    ->Arg(10'000'000)
    ->Arg(100'000'000)
    ->MinTime(bench_utils::cpu_min_time_seconds())
    ->Repetitions(bench_utils::cpu_repetitions(30))
    ->ReportAggregatesOnly(true)
    ->DisplayAggregatesOnly(true);

BENCHMARK(BM_ParallelReduction)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1'000'000)
    ->Arg(10'000'000)
    ->Arg(100'000'000)
    ->MinTime(bench_utils::cpu_min_time_seconds())
    ->Repetitions(bench_utils::cpu_repetitions(30))
    ->ReportAggregatesOnly(true)
    ->DisplayAggregatesOnly(true);
