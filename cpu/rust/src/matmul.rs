//! Matrix Multiplication Benchmark — Rust
//! ========================================
//! Naive triple-loop dense matrix multiplication (f64).
//! No BLAS — ensures apples-to-apples comparison with C++.

/// Naive O(n³) matrix multiplication: C = A × B
/// Row-major layout, ikj loop order for better cache behavior.
pub fn matmul_naive(a: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    // Zero output
    c.iter_mut().for_each(|x| *x = 0.0);

    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
}

/// Compute GFLOPS for matrix multiplication.
/// Standard matmul: 2*n³ floating-point operations.
pub fn compute_gflops(n: usize, time_seconds: f64) -> f64 {
    let flops = 2.0 * (n as f64).powi(3);
    flops / (time_seconds * 1e9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_identity() {
        let n = 3;
        // A = identity
        let a = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let b = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let mut c = vec![0.0; n * n];

        matmul_naive(&a, &b, &mut c, n);

        for i in 0..n * n {
            assert!(
                (c[i] - b[i]).abs() < 1e-10,
                "Mismatch at index {}: {} != {}",
                i, c[i], b[i]
            );
        }
    }

    #[test]
    fn test_matmul_2x2() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];

        matmul_naive(&a, &b, &mut c, 2);

        assert!((c[0] - 19.0).abs() < 1e-10);
        assert!((c[1] - 22.0).abs() < 1e-10);
        assert!((c[2] - 43.0).abs() < 1e-10);
        assert!((c[3] - 50.0).abs() < 1e-10);
    }
}
