/**
 * QuickSort Benchmark — C++
 * ==========================
 * Benchmarks std::sort (introsort) and manual quicksort on integer arrays.
 * Input sizes: 1M, 10M, 100M elements.
 * Compiled with -O3 for fair comparison against Rust's slice::sort_unstable.
 */

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "benchmark_utils.h"

// ── Manual QuickSort Implementation ─────────────────────────────────────────

namespace {

template <typename Iter>
Iter partition(Iter first, Iter last) {
    auto pivot = std::prev(last);
    auto i = first;
    for (auto j = first; j != pivot; ++j) {
        if (*j <= *pivot) {
            std::iter_swap(i, j);
            ++i;
        }
    }
    std::iter_swap(i, pivot);
    return i;
}

template <typename Iter>
void quicksort_impl(Iter first, Iter last) {
    if (std::distance(first, last) <= 1) return;
    
    // Use median-of-three for pivot selection
    auto mid = first + std::distance(first, last) / 2;
    auto pivot_candidate = std::prev(last);
    
    if (*mid < *first) std::iter_swap(mid, first);
    if (*pivot_candidate < *first) std::iter_swap(pivot_candidate, first);
    if (*mid < *pivot_candidate) std::iter_swap(mid, pivot_candidate);
    
    auto pivot = partition(first, last);
    quicksort_impl(first, pivot);
    quicksort_impl(std::next(pivot), last);
}

} // anonymous namespace

// ── Benchmark: std::sort ────────────────────────────────────────────────────

static void BM_StdSort(benchmark::State& state) {
    const size_t n = state.range(0);
    auto original = bench_utils::generate_random_array<int32_t>(n);
    
    for (auto _ : state) {
        // Copy data each iteration (sorting is destructive)
        auto data = original;
        std::sort(data.begin(), data.end());
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
    state.SetLabel("std::sort");
}

// ── Benchmark: Manual QuickSort ─────────────────────────────────────────────

static void BM_ManualQuickSort(benchmark::State& state) {
    const size_t n = state.range(0);
    auto original = bench_utils::generate_random_array<int32_t>(n);
    
    for (auto _ : state) {
        auto data = original;
        quicksort_impl(data.begin(), data.end());
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
    state.SetLabel("manual_quicksort");
}

// ── Register Benchmarks ─────────────────────────────────────────────────────

BENCHMARK(BM_StdSort)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1'000'000)
    ->Arg(10'000'000)
    ->Arg(100'000'000)
    ->MinTime(bench_utils::cpu_min_time_seconds())
    ->Repetitions(bench_utils::cpu_repetitions(30))
    ->ReportAggregatesOnly(true)
    ->DisplayAggregatesOnly(true);

BENCHMARK(BM_ManualQuickSort)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1'000'000)
    ->Arg(10'000'000)
    ->Arg(100'000'000)
    ->MinTime(bench_utils::cpu_min_time_seconds())
    ->Repetitions(bench_utils::cpu_repetitions(30))
    ->ReportAggregatesOnly(true)
    ->DisplayAggregatesOnly(true);
