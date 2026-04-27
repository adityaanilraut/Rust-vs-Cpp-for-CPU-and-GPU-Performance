/**
 * Binary Search Benchmark — C++
 * ===============================
 * Benchmarks std::binary_search and std::lower_bound on sorted arrays.
 * Tests memory access patterns and branch prediction.
 * Input sizes: 1M, 10M, 100M sorted elements with 100K random lookups each.
 */

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "benchmark_utils.h"

// ── Benchmark: std::binary_search ───────────────────────────────────────────

static void BM_BinarySearch(benchmark::State& state) {
    const size_t n = state.range(0);
    const size_t num_queries = 100'000;

    // Create sorted array
    std::vector<int32_t> data(n);
    std::iota(data.begin(), data.end(), 0);

    // Generate random search targets (mix of hits and misses)
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(n * 1.2));
    std::vector<int32_t> queries(num_queries);
    for (auto& q : queries) q = dist(rng);

    for (auto _ : state) {
        int64_t found = 0;
        for (const auto& q : queries) {
            found += std::binary_search(data.begin(), data.end(), q);
        }
        benchmark::DoNotOptimize(found);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_queries);
    state.SetLabel("std::binary_search");
}

// ── Benchmark: std::lower_bound ─────────────────────────────────────────────

static void BM_LowerBound(benchmark::State& state) {
    const size_t n = state.range(0);
    const size_t num_queries = 100'000;

    std::vector<int32_t> data(n);
    std::iota(data.begin(), data.end(), 0);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(n * 1.2));
    std::vector<int32_t> queries(num_queries);
    for (auto& q : queries) q = dist(rng);

    for (auto _ : state) {
        int64_t sum_indices = 0;
        for (const auto& q : queries) {
            auto it = std::lower_bound(data.begin(), data.end(), q);
            sum_indices += std::distance(data.begin(), it);
        }
        benchmark::DoNotOptimize(sum_indices);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_queries);
    state.SetLabel("std::lower_bound");
}

// ── Register ────────────────────────────────────────────────────────────────

BENCHMARK(BM_BinarySearch)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1'000'000)
    ->Arg(10'000'000)
    ->Arg(100'000'000)
    ->MinTime(bench_utils::cpu_min_time_seconds())
    ->Repetitions(bench_utils::cpu_repetitions(30))
    ->ReportAggregatesOnly(true)
    ->DisplayAggregatesOnly(true);

BENCHMARK(BM_LowerBound)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1'000'000)
    ->Arg(10'000'000)
    ->Arg(100'000'000)
    ->MinTime(bench_utils::cpu_min_time_seconds())
    ->Repetitions(bench_utils::cpu_repetitions(30))
    ->ReportAggregatesOnly(true)
    ->DisplayAggregatesOnly(true);
