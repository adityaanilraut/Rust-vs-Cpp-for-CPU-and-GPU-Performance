//! Binary Search Benchmark — Rust
//! ================================
//! Benchmarks `slice::binary_search` on sorted arrays.
//! Tests memory access patterns and branch prediction.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Binary search returning whether the key was found.
pub fn binary_search_sorted(data: &[i32], queries: &[i32]) -> i64 {
    let mut found: i64 = 0;
    for &q in queries {
        if data.binary_search(&q).is_ok() {
            found += 1;
        }
    }
    found
}

/// Lower-bound search returning sum of insertion indices.
pub fn lower_bound_sorted(data: &[i32], queries: &[i32]) -> i64 {
    let mut sum: i64 = 0;
    for &q in queries {
        let idx = data.partition_point(|x| *x < q);
        sum += idx as i64;
    }
    sum
}

/// Generate sorted array and random queries.
pub fn generate_search_data(n: usize, num_queries: usize) -> (Vec<i32>, Vec<i32>) {
    let data: Vec<i32> = (0..n as i32).collect();
    let mut rng = StdRng::seed_from_u64(42);
    let max_val = (n as f64 * 1.2) as i32;
    let queries: Vec<i32> = (0..num_queries).map(|_| rng.gen_range(0..max_val)).collect();
    (data, queries)
}
