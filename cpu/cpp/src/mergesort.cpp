/**
 * MergeSort Benchmark — C++
 * ==========================
 * Custom recursive merge sort implementation.
 * Input sizes: 1M, 10M, 100M int32 elements.
 * Uses manual implementation to ensure kernel parity with Rust.
 */

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <vector>

#include "benchmark_utils.h"

// ── Recursive MergeSort Implementation ──────────────────────────────────────

namespace {

template <typename T>
void merge(std::vector<T>& arr, std::vector<T>& tmp,
           size_t left, size_t mid, size_t right) {
    size_t i = left, j = mid, k = left;
    
    while (i < mid && j < right) {
        if (arr[i] <= arr[j]) {
            tmp[k++] = arr[i++];
        } else {
            tmp[k++] = arr[j++];
        }
    }
    
    while (i < mid) tmp[k++] = arr[i++];
    while (j < right) tmp[k++] = arr[j++];
    
    std::copy(tmp.begin() + left, tmp.begin() + right,
              arr.begin() + left);
}

template <typename T>
void mergesort_impl(std::vector<T>& arr, std::vector<T>& tmp,
                    size_t left, size_t right) {
    if (right - left <= 1) return;
    
    // Switch to insertion sort for small ranges (optimization)
    if (right - left <= 32) {
        for (size_t i = left + 1; i < right; ++i) {
            T key = arr[i];
            size_t j = i;
            while (j > left && arr[j - 1] > key) {
                arr[j] = arr[j - 1];
                --j;
            }
            arr[j] = key;
        }
        return;
    }
    
    size_t mid = left + (right - left) / 2;
    mergesort_impl(arr, tmp, left, mid);
    mergesort_impl(arr, tmp, mid, right);
    merge(arr, tmp, left, mid, right);
}

} // anonymous namespace

// ── Benchmark ───────────────────────────────────────────────────────────────

static void BM_MergeSort(benchmark::State& state) {
    const size_t n = state.range(0);
    auto original = bench_utils::generate_random_array<int32_t>(n);
    std::vector<int32_t> tmp(n);
    
    for (auto _ : state) {
        auto data = original;
        mergesort_impl(data, tmp, 0, n);
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
    
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
    state.SetLabel("recursive_mergesort");
}

// ── Register ────────────────────────────────────────────────────────────────

BENCHMARK(BM_MergeSort)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1'000'000)
    ->Arg(10'000'000)
    ->Arg(100'000'000)
    ->MinTime(bench_utils::cpu_min_time_seconds())
    ->Repetitions(bench_utils::cpu_repetitions(30))
    ->ReportAggregatesOnly(true)
    ->DisplayAggregatesOnly(true);
