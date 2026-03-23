//! Parallel Vector Reduction Benchmark — Rust
//! ============================================
//! Parallel summation of f64 vectors using Rayon.
//! Compared against C++ OpenMP implementation.

use rayon::prelude::*;

/// Sequential reduction (baseline).
pub fn sequential_reduce(data: &[f64]) -> f64 {
    data.iter().sum()
}

/// Parallel reduction using Rayon.
/// Rayon automatically determines thread count based on available cores.
pub fn parallel_reduce_rayon(data: &[f64]) -> f64 {
    data.par_iter().sum()
}

/// Get the number of Rayon threads (for reporting).
pub fn rayon_thread_count() -> usize {
    rayon::current_num_threads()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_random_f64;

    #[test]
    fn test_sequential_reduce() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sequential_reduce(&data);
        assert!((result - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_reduce() {
        let data = generate_random_f64(100_000, 42);
        let seq = sequential_reduce(&data);
        let par = parallel_reduce_rayon(&data);
        // Parallel sum may differ slightly due to floating-point ordering
        assert!(
            (seq - par).abs() < 1e-6 * seq.abs().max(1.0),
            "Sequential: {}, Parallel: {}",
            seq,
            par
        );
    }
}
