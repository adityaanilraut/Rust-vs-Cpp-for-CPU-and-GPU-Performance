//! Criterion Benchmark Harness for CPU Benchmarks
//! ================================================
//! Runs all 8 CPU benchmarks with the same input sizes
//! and trial counts as the C++ Google Benchmark suite.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rust_cpu_benchmarks::{
    binary_search, fft, hashmap, matmul, mergesort, nbody, parallel_reduction, quicksort, utils,
};
use std::time::Duration;

fn measurement_time(secs: u64) -> Duration {
    Duration::from_secs(secs)
}

fn sample_size(size: usize) -> usize {
    size
}

// ── QuickSort Benchmarks ────────────────────────────────────────────────────

fn bench_quicksort(c: &mut Criterion) {
    let mut group = c.benchmark_group("QuickSort");
    group.measurement_time(measurement_time(5));
    group.sample_size(sample_size(30));

    let sizes: &[usize] = &[1_000_000usize, 10_000_000, 100_000_000];

    for &size in sizes {
        let original = utils::generate_random_i32(size, 42);

        group.bench_with_input(
            BenchmarkId::new("sort_unstable", size),
            &size,
            |b, _| {
                b.iter_batched(
                    || original.clone(),
                    |mut data| {
                        quicksort::std_sort_unstable(&mut data);
                        data
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("manual_quicksort", size),
            &size,
            |b, _| {
                b.iter_batched(
                    || original.clone(),
                    |mut data| {
                        quicksort::manual_quicksort(&mut data);
                        data
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

// ── MergeSort Benchmarks ────────────────────────────────────────────────────

fn bench_mergesort(c: &mut Criterion) {
    let mut group = c.benchmark_group("MergeSort");
    group.measurement_time(measurement_time(5));
    group.sample_size(sample_size(30));

    let sizes: &[usize] = &[1_000_000usize, 10_000_000, 100_000_000];

    for &size in sizes {
        let original = utils::generate_random_i32(size, 42);

        group.bench_with_input(BenchmarkId::new("recursive", size), &size, |b, _| {
            b.iter_batched(
                || original.clone(),
                |mut data| {
                    mergesort::mergesort(&mut data);
                    data
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ── Matrix Multiplication Benchmarks ────────────────────────────────────────

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("MatMul");
    group.measurement_time(measurement_time(10));

    let sizes: &[usize] = &[512usize, 1024];

    for &n in sizes {
        let a = utils::generate_random_f64(n * n, 42);
        let b = utils::generate_random_f64(n * n, 43);
        let mut out = vec![0.0f64; n * n];

        group.sample_size(sample_size(10));
        group.bench_with_input(BenchmarkId::new("naive", n), &n, |bench, &n| {
            bench.iter(|| {
                matmul::matmul_naive(&a, &b, &mut out, n);
            });
        });
    }
    group.finish();
}

// ── Parallel Reduction Benchmarks ───────────────────────────────────────────

fn bench_parallel_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("ParallelReduction");
    group.measurement_time(measurement_time(5));
    group.sample_size(sample_size(30));

    let sizes: &[usize] = &[1_000_000usize, 10_000_000, 100_000_000];

    for &size in sizes {
        let data = utils::generate_random_f64(size, 42);

        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            &size,
            |b, _| {
                b.iter(|| parallel_reduction::sequential_reduce(&data));
            },
        );

        group.bench_with_input(BenchmarkId::new("rayon", size), &size, |b, _| {
            b.iter(|| parallel_reduction::parallel_reduce_rayon(&data));
        });
    }
    group.finish();

    println!(
        "Rayon thread count: {}",
        parallel_reduction::rayon_thread_count()
    );
}

// ── Binary Search Benchmarks ────────────────────────────────────────────────

fn bench_binary_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("BinarySearch");
    group.measurement_time(measurement_time(5));
    group.sample_size(sample_size(30));

    let sizes: &[usize] = &[1_000_000usize, 10_000_000, 100_000_000];

    for &size in sizes {
        let (data, queries) = binary_search::generate_search_data(size, 100_000);

        group.bench_with_input(
            BenchmarkId::new("binary_search", size),
            &size,
            |b, _| {
                b.iter(|| binary_search::binary_search_sorted(&data, &queries));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lower_bound", size),
            &size,
            |b, _| {
                b.iter(|| binary_search::lower_bound_sorted(&data, &queries));
            },
        );
    }
    group.finish();
}

// ── N-Body Simulation Benchmarks ────────────────────────────────────────────

fn bench_nbody(c: &mut Criterion) {
    let mut group = c.benchmark_group("NBody");
    group.measurement_time(measurement_time(5));

    let sizes: &[usize] = &[1024usize, 4096];

    for &n in sizes {
        let original = nbody::generate_bodies(n, 42);

        group.sample_size(sample_size(10));
        group.bench_with_input(BenchmarkId::new("single_step", n), &n, |b, _| {
            b.iter_batched(
                || original.clone(),
                |mut bodies| {
                    nbody::compute_forces(&mut bodies);
                    nbody::integrate_positions(&mut bodies);
                    bodies
                },
                criterion::BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("5_timesteps", n), &n, |b, _| {
            b.iter_batched(
                || original.clone(),
                |mut bodies| {
                    for _ in 0..5 {
                        nbody::compute_forces(&mut bodies);
                        nbody::integrate_positions(&mut bodies);
                    }
                    bodies
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ── HashMap Benchmarks ──────────────────────────────────────────────────────

fn bench_hashmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("HashMap");
    group.measurement_time(measurement_time(5));
    group.sample_size(sample_size(30));

    let int_sizes: &[usize] = &[100_000usize, 1_000_000, 10_000_000];

    for &size in int_sizes {
        let keys = hashmap::generate_int_keys(size, 42);

        group.bench_with_input(
            BenchmarkId::new("int_insert", size),
            &size,
            |b, _| {
                b.iter(|| hashmap::hashmap_insert_int(&keys));
            },
        );

        // Build map for lookup benchmark
        let map = hashmap::hashmap_insert_int(&keys);
        let queries = hashmap::generate_queries(&keys, size, 100_000, 99);

        group.bench_with_input(
            BenchmarkId::new("int_lookup", size),
            &size,
            |b, _| {
                b.iter(|| hashmap::hashmap_lookup_int(&map, &queries));
            },
        );
    }

    // String key benchmarks
    let string_sizes: &[usize] = &[100_000usize, 1_000_000];

    for &size in string_sizes {
        let keys = hashmap::generate_string_keys(size, 42);

        group.bench_with_input(
            BenchmarkId::new("string_insert", size),
            &size,
            |b, _| {
                b.iter(|| hashmap::hashmap_insert_string(&keys));
            },
        );
    }
    group.finish();
}

// ── FFT Benchmarks ──────────────────────────────────────────────────────────

fn bench_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT");
    group.measurement_time(measurement_time(5));
    group.sample_size(sample_size(30));

    let sizes: &[usize] = &[1usize << 16, 1 << 20, 1 << 24];

    for &n in sizes {
        let original = fft::generate_signal(n, 42);

        group.bench_with_input(BenchmarkId::new("forward", n), &n, |b, _| {
            b.iter_batched(
                || original.clone(),
                |mut data| {
                    fft::fft(&mut data, false);
                    data
                },
                criterion::BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("roundtrip", n), &n, |b, _| {
            b.iter_batched(
                || original.clone(),
                |mut data| {
                    fft::fft(&mut data, false);
                    fft::fft(&mut data, true);
                    data
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ── Register All Benchmark Groups ───────────────────────────────────────────

criterion_group!(
    benches,
    bench_quicksort,
    bench_mergesort,
    bench_matmul,
    bench_parallel_reduction,
    bench_binary_search,
    bench_nbody,
    bench_hashmap,
    bench_fft
);
criterion_main!(benches);
