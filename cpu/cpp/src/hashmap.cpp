/**
 * HashMap Benchmark — C++
 * ========================
 * Benchmarks std::unordered_map for insert and lookup operations.
 * Tests real-world data structure performance: hashing, allocation, cache.
 * Input sizes: 100K, 1M, 10M key-value pairs.
 */

#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "benchmark_utils.h"

// ── Benchmark: Integer key insert ───────────────────────────────────────────

static void BM_HashMapInsert_Int(benchmark::State& state) {
    const size_t n = state.range(0);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, static_cast<int64_t>(n * 10));
    std::vector<int64_t> keys(n);
    for (auto& k : keys) k = dist(rng);

    for (auto _ : state) {
        std::unordered_map<int64_t, int64_t> map;
        map.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            map[keys[i]] = static_cast<int64_t>(i);
        }
        benchmark::DoNotOptimize(map);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
    state.SetLabel("int_insert");
}

// ── Benchmark: Integer key lookup ───────────────────────────────────────────

static void BM_HashMapLookup_Int(benchmark::State& state) {
    const size_t n = state.range(0);
    const size_t num_queries = 100'000;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, static_cast<int64_t>(n * 10));

    // Build map
    std::unordered_map<int64_t, int64_t> map;
    map.reserve(n);
    std::vector<int64_t> keys(n);
    for (size_t i = 0; i < n; ++i) {
        keys[i] = dist(rng);
        map[keys[i]] = static_cast<int64_t>(i);
    }

    // Generate queries (mix of hits and misses)
    std::uniform_int_distribution<size_t> idx_dist(0, n - 1);
    std::vector<int64_t> queries(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        if (i % 2 == 0) {
            queries[i] = keys[idx_dist(rng)];  // Hit
        } else {
            queries[i] = dist(rng);  // Probable miss
        }
    }

    for (auto _ : state) {
        int64_t sum = 0;
        for (const auto& q : queries) {
            auto it = map.find(q);
            if (it != map.end()) sum += it->second;
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_queries);
    state.SetLabel("int_lookup");
}

// ── Benchmark: String key insert ────────────────────────────────────────────

static void BM_HashMapInsert_String(benchmark::State& state) {
    const size_t n = state.range(0);

    // Generate random string keys
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> char_dist('a', 'z');
    std::uniform_int_distribution<int> len_dist(8, 32);

    std::vector<std::string> keys(n);
    for (auto& k : keys) {
        int len = len_dist(rng);
        k.resize(len);
        for (auto& c : k) c = static_cast<char>(char_dist(rng));
    }

    for (auto _ : state) {
        std::unordered_map<std::string, int64_t> map;
        map.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            map[keys[i]] = static_cast<int64_t>(i);
        }
        benchmark::DoNotOptimize(map);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
    state.SetLabel("string_insert");
}

// ── Register ────────────────────────────────────────────────────────────────

BENCHMARK(BM_HashMapInsert_Int)
    ->Unit(benchmark::kMillisecond)
    ->Arg(100'000)
    ->Arg(1'000'000)
    ->Arg(10'000'000)
    ->MinTime(bench_utils::cpu_min_time_seconds())
    ->Repetitions(bench_utils::cpu_repetitions(30))
    ->ReportAggregatesOnly(true)
    ->DisplayAggregatesOnly(true);

BENCHMARK(BM_HashMapLookup_Int)
    ->Unit(benchmark::kMillisecond)
    ->Arg(100'000)
    ->Arg(1'000'000)
    ->Arg(10'000'000)
    ->MinTime(bench_utils::cpu_min_time_seconds())
    ->Repetitions(bench_utils::cpu_repetitions(30))
    ->ReportAggregatesOnly(true)
    ->DisplayAggregatesOnly(true);

BENCHMARK(BM_HashMapInsert_String)
    ->Unit(benchmark::kMillisecond)
    ->Arg(100'000)
    ->Arg(1'000'000)
    ->MinTime(bench_utils::cpu_min_time_seconds())
    ->Repetitions(bench_utils::cpu_repetitions(30))
    ->ReportAggregatesOnly(true)
    ->DisplayAggregatesOnly(true);
