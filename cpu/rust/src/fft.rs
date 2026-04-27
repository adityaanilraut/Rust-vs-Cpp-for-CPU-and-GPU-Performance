//! FFT Benchmark — Rust
//! =====================
//! Cooley-Tukey radix-2 Fast Fourier Transform.
//! Matches the C++ implementation for fair comparison.

use std::f64::consts::PI;

#[derive(Clone, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self { Complex { re, im } }

    pub fn mul(self, other: Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    pub fn add(self, other: Complex) -> Complex {
        Complex { re: self.re + other.re, im: self.im + other.im }
    }

    pub fn sub(self, other: Complex) -> Complex {
        Complex { re: self.re - other.re, im: self.im - other.im }
    }

    pub fn scale(self, s: f64) -> Complex {
        Complex { re: self.re * s, im: self.im * s }
    }
}

/// Bit-reversal permutation.
fn bit_reverse(data: &mut [Complex]) {
    let n = data.len();
    let mut j: usize = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

/// In-place iterative radix-2 FFT (Cooley-Tukey).
pub fn fft(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    bit_reverse(data);

    let mut len = 2;
    while len <= n {
        let angle = 2.0 * PI / len as f64 * if inverse { -1.0 } else { 1.0 };
        let wlen = Complex::new(angle.cos(), angle.sin());

        let mut i = 0;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..len / 2 {
                let u = data[i + j];
                let v = data[i + j + len / 2].mul(w);
                data[i + j] = u.add(v);
                data[i + j + len / 2] = u.sub(v);
                w = w.mul(wlen);
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / n as f64;
        for x in data.iter_mut() {
            *x = x.scale(inv_n);
        }
    }
}

/// Generate random complex signal with deterministic seed.
pub fn generate_signal(n: usize, seed: u64) -> Vec<Complex> {
    let mut rng = seed;
    let mut next_normal = || -> f64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Simple uniform → rough normal approximation via Box-Muller isn't needed;
        // uniform distribution is sufficient for benchmarking purposes.
        ((rng >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    };

    (0..n).map(|_| Complex::new(next_normal(), next_normal())).collect()
}
