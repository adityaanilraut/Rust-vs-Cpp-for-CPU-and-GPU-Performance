/**
 * N-Body Simulation Benchmark — C++
 * ====================================
 * Gravitational N-body simulation using direct pairwise force calculation.
 * Classic compute-intensive HPC workload (O(n²) complexity).
 * Tests FP throughput, SIMD auto-vectorization, and cache efficiency.
 * Input sizes: 1024, 4096, 16384 bodies.
 */

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "benchmark_utils.h"

// ── Data Structures ─────────────────────────────────────────────────────────

struct Body {
    double x, y, z;     // Position
    double vx, vy, vz;  // Velocity
    double mass;
};

// ── N-Body Force Calculation ────────────────────────────────────────────────

namespace {

constexpr double SOFTENING = 1e-9;  // Prevent division by zero
constexpr double DT = 0.01;        // Time step

void compute_forces(std::vector<Body>& bodies) {
    const size_t n = bodies.size();

    for (size_t i = 0; i < n; ++i) {
        double fx = 0.0, fy = 0.0, fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;

            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dz = bodies[j].z - bodies[i].z;

            double dist_sq = dx * dx + dy * dy + dz * dz + SOFTENING;
            double inv_dist = 1.0 / std::sqrt(dist_sq);
            double inv_dist3 = inv_dist * inv_dist * inv_dist;

            double force = bodies[j].mass * inv_dist3;
            fx += dx * force;
            fy += dy * force;
            fz += dz * force;
        }

        bodies[i].vx += DT * fx;
        bodies[i].vy += DT * fy;
        bodies[i].vz += DT * fz;
    }
}

void integrate_positions(std::vector<Body>& bodies) {
    for (auto& b : bodies) {
        b.x += DT * b.vx;
        b.y += DT * b.vy;
        b.z += DT * b.vz;
    }
}

std::vector<Body> generate_bodies(size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> pos_dist(-100.0, 100.0);
    std::uniform_real_distribution<double> vel_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> mass_dist(0.1, 10.0);

    std::vector<Body> bodies(n);
    for (auto& b : bodies) {
        b.x = pos_dist(rng); b.y = pos_dist(rng); b.z = pos_dist(rng);
        b.vx = vel_dist(rng); b.vy = vel_dist(rng); b.vz = vel_dist(rng);
        b.mass = mass_dist(rng);
    }
    return bodies;
}

} // anonymous namespace

// ── Benchmark: Single timestep ──────────────────────────────────────────────

static void BM_NBody(benchmark::State& state) {
    const size_t n = state.range(0);
    auto original = generate_bodies(n);

    for (auto _ : state) {
        auto bodies = original;  // Fresh copy each iteration
        compute_forces(bodies);
        integrate_positions(bodies);
        benchmark::DoNotOptimize(bodies.data());
        benchmark::ClobberMemory();
    }

    // Report GFLOPS: ~20 FLOPs per pair interaction
    double flops_per_step = 20.0 * n * n;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops_per_step,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
    state.SetLabel("direct_pairwise");
}

// ── Benchmark: Multiple timesteps ───────────────────────────────────────────

static void BM_NBody_MultiStep(benchmark::State& state) {
    const size_t n = state.range(0);
    const int steps = 5;
    auto original = generate_bodies(n);

    for (auto _ : state) {
        auto bodies = original;
        for (int s = 0; s < steps; ++s) {
            compute_forces(bodies);
            integrate_positions(bodies);
        }
        benchmark::DoNotOptimize(bodies.data());
        benchmark::ClobberMemory();
    }

    double flops_per_step = 20.0 * n * n * steps;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops_per_step,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
    state.SetLabel("5_timesteps");
}

// ── Register ────────────────────────────────────────────────────────────────

BENCHMARK(BM_NBody)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384)
    ->MinTime(bench_utils::cpu_min_time_seconds())
    ->Repetitions(bench_utils::cpu_repetitions(10))
    ->ReportAggregatesOnly(true)
    ->DisplayAggregatesOnly(true);

BENCHMARK(BM_NBody_MultiStep)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1024)
    ->Arg(4096)
    ->MinTime(bench_utils::cpu_min_time_seconds())
    ->Repetitions(bench_utils::cpu_repetitions(10))
    ->ReportAggregatesOnly(true)
    ->DisplayAggregatesOnly(true);
