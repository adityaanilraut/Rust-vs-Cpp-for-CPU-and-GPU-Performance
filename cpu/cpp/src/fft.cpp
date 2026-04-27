/**
 * FFT Benchmark — C++
 * =====================
 * Cooley-Tukey radix-2 Fast Fourier Transform.
 * Classic HPC workload testing complex arithmetic and memory access patterns.
 * Input sizes: 2^16 (64K), 2^20 (1M), 2^24 (16M) complex elements.
 */

#include <benchmark/benchmark.h>
#include <cmath>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif
#include <complex>
#include <cstdint>
#include <random>
#include <vector>

#include "benchmark_utils.h"

using Complex = std::complex<double>;

// ── Iterative Cooley-Tukey FFT ──────────────────────────────────────────────

namespace {

/**
 * Bit-reversal permutation.
 */
void bit_reverse(std::vector<Complex>& data) {
    size_t n = data.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }
}

/**
 * In-place iterative radix-2 FFT (Cooley-Tukey).
 * Direction: forward (DFT) when inverse=false, inverse (IDFT) when inverse=true.
 */
void fft(std::vector<Complex>& data, bool inverse = false) {
    size_t n = data.size();
    bit_reverse(data);

    for (size_t len = 2; len <= n; len <<= 1) {
        double angle = 2.0 * M_PI / static_cast<double>(len) * (inverse ? -1.0 : 1.0);
        Complex wlen(std::cos(angle), std::sin(angle));

        for (size_t i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; ++j) {
                Complex u = data[i + j];
                Complex v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        for (auto& x : data) x /= static_cast<double>(n);
    }
}

std::vector<Complex> generate_signal(size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<Complex> data(n);
    for (auto& x : data) {
        x = Complex(dist(rng), dist(rng));
    }
    return data;
}

} // anonymous namespace

// ── Benchmark: Forward FFT ──────────────────────────────────────────────────

static void BM_FFT_Forward(benchmark::State& state) {
    const size_t n = state.range(0);
    auto original = generate_signal(n);

    for (auto _ : state) {
        auto data = original;
        fft(data, false);
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }

    // FFT: ~5 * n * log2(n) FLOPs
    double flops = 5.0 * n * std::log2(static_cast<double>(n));
    state.counters["GFLOPS"] = benchmark::Counter(
        flops,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
    state.SetLabel("forward_fft");
}

// ── Benchmark: Forward + Inverse round-trip ─────────────────────────────────

static void BM_FFT_Roundtrip(benchmark::State& state) {
    const size_t n = state.range(0);
    auto original = generate_signal(n);

    for (auto _ : state) {
        auto data = original;
        fft(data, false);   // Forward
        fft(data, true);    // Inverse
        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }

    double flops = 2.0 * 5.0 * n * std::log2(static_cast<double>(n));
    state.counters["GFLOPS"] = benchmark::Counter(
        flops,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
    state.SetLabel("fft_roundtrip");
}

// ── Register ────────────────────────────────────────────────────────────────

BENCHMARK(BM_FFT_Forward)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1 << 16)    // 64K
    ->Arg(1 << 20)    // 1M
    ->Arg(1 << 24)    // 16M
    ->MinTime(bench_utils::cpu_min_time_seconds())
    ->Repetitions(bench_utils::cpu_repetitions(30))
    ->ReportAggregatesOnly(true)
    ->DisplayAggregatesOnly(true);

BENCHMARK(BM_FFT_Roundtrip)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1 << 16)
    ->Arg(1 << 20)
    ->Arg(1 << 24)
    ->MinTime(bench_utils::cpu_min_time_seconds())
    ->Repetitions(bench_utils::cpu_repetitions(30))
    ->ReportAggregatesOnly(true)
    ->DisplayAggregatesOnly(true);
