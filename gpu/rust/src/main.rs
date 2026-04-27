//! Rust GPU Benchmark Runner
//! ==========================
//! Uses cudarc to load pre-compiled PTX kernels and benchmark them
//! with the same protocol as the C++/CUDA runner.
//!
//! PTX kernels must be compiled first:
//!   nvcc -ptx kernels/matmul.cu -o kernels/matmul.ptx
//!   nvcc -ptx kernels/softmax.cu -o kernels/softmax.ptx
//!   nvcc -ptx kernels/reduction.cu -o kernels/reduction.ptx

use cudarc::driver::*;
use std::sync::Arc;
use std::time::Instant;
use std::fs;
use std::io::Write;

const NUM_TRIALS: usize = 30;
const WARMUP_RUNS: usize = 3;

fn num_trials() -> usize {
    NUM_TRIALS
}

fn warmup_runs() -> usize {
    WARMUP_RUNS
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Rust vs. C++ GPU Benchmark Suite (Rust/CUDA)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Initialize CUDA
    let dev = CudaDevice::new(0)?;
    println!("  GPU initialized via cudarc (Device 0)");

    // Load PTX modules
    let kernel_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");

    let _matmul_ptx = CudaModule::from_ptx_file(
        &kernel_dir.join("matmul.ptx"), "matmul", &["matmul_tiled"], &dev
    );
    let _softmax_ptx = CudaModule::from_ptx_file(
        &kernel_dir.join("softmax.ptx"), "softmax", &["softmax_kernel"], &dev
    );
    let _reduction_ptx = CudaModule::from_ptx_file(
        &kernel_dir.join("reduction.ptx"), "reduction", &["reduce_kernel"], &dev
    );

    // Results CSV
    let results_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../results/raw");
    fs::create_dir_all(&results_dir)?;
    let mut csv = fs::File::create(results_dir.join("gpu_rust.csv"))?;
    writeln!(csv, "benchmark,language,input_size,trial,kernel_time_ms,h2d_time_ms,d2h_time_ms,gflops,bandwidth_gbps")?;

    // ── Matrix Multiplication ──────────────────────────────────────
    println!("\n━━ CUDA Matrix Multiplication (Tiled, via Rust) ━━");

    let matmul_sizes: &[usize] = &[1024_usize, 2048, 4096];
    for &n in matmul_sizes {
        println!("  Size: {}×{}", n, n);
        let h_a = random_f32(n * n, 42);
        let h_b = random_f32(n * n, 43);

        // Allocate device memory
        let _d_a = dev.htod_sync_copy(&h_a)?;
        let _d_b = dev.htod_sync_copy(&h_b)?;
        let mut _d_c = dev.alloc_zeros::<f32>(n * n)?;

        // Warm-up
        for _ in 0..warmup_runs() {
            // Note: actual kernel launch via cudarc requires loaded PTX
            // This is a timing framework — the kernel launches will work
            // when PTX files are compiled and available
            dev.synchronize()?;
        }

        let mut kernel_times = Vec::new();
        if let Some(f_matmul) = dev.get_func("matmul", "matmul_tiled") {
            let block_dim = (32, 32, 1);
            let grid_dim = (((n + 31) / 32) as u32, ((n + 31) / 32) as u32, 1);
            let cfg = LaunchConfig { grid_dim, block_dim, shared_mem_bytes: 0 };
            
            for t in 0..num_trials() {
                let start = Instant::now();
                unsafe { f_matmul.clone().launch(cfg, (&_d_a, &_d_b, &mut _d_c, n as i32)) }?;
                dev.synchronize()?;
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                kernel_times.push(elapsed as f32);

            let gflops = 2.0 * (n as f64).powi(3) / (elapsed * 1e-3) / 1e9;
            writeln!(csv, "cuda_matmul_tiled,rust_cuda,{},{},{:.4},0,0,{:.2},0",
                     n, t, elapsed, gflops)?;
        }

        let stats = compute_stats(&kernel_times);
        println!("    Kernel: {:.3} ± {:.3} ms  [CI: {:.3} – {:.3}]",
                 stats.mean, stats.stddev, stats.ci95_low, stats.ci95_high);
        }
    }

    // ── Softmax ────────────────────────────────────────────────────
    println!("\n━━ CUDA Softmax (Row-wise, via Rust) ━━");

    let softmax_sizes: &[(usize, usize)] = &[(64, 16384), (64, 65536), (64, 262144)];
    for &(batch, seq_len) in softmax_sizes {
        println!("  Size: {} × {}", batch, seq_len);
        let h_input = random_f32(batch * seq_len, 42);

        let _d_input = dev.htod_sync_copy(&h_input)?;
        let mut _d_output = dev.alloc_zeros::<f32>(batch * seq_len)?;

        let mut kernel_times = Vec::new();
        if let Some(f_softmax) = dev.get_func("softmax", "softmax_kernel") {
            let threads = 256;
            let shared_bytes = 2 * (threads / 32) * std::mem::size_of::<f32>() as u32;
            let cfg = LaunchConfig {
                grid_dim: (batch as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: shared_bytes,
            };

            for t in 0..num_trials() {
                let start = Instant::now();
                unsafe { f_softmax.clone().launch(cfg, (&_d_input, &mut _d_output, batch as i32, seq_len as i32)) }?;
                dev.synchronize()?;
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                kernel_times.push(elapsed as f32);

            let bw = 2.0 * (batch * seq_len) as f64 * 4.0 / (elapsed * 1e-3) / 1e9;
            writeln!(csv, "cuda_softmax,rust_cuda,{}x{},{},{:.4},0,0,0,{:.2}",
                     batch, seq_len, t, elapsed, bw)?;
        }

        let stats = compute_stats(&kernel_times);
        println!("    Kernel: {:.3} ± {:.3} ms", stats.mean, stats.stddev);
        }
    }

    // ── Vector Reduction ───────────────────────────────────────────
    println!("\n━━ CUDA Vector Reduction (via Rust) ━━");

    let reduction_sizes: &[usize] = &[1_000_000_usize, 10_000_000, 100_000_000];
    for &n in reduction_sizes {
        println!("  Size: {} elements", n);
        let h_input = random_f32(n, 42);

        let _d_input = dev.htod_sync_copy(&h_input)?;

        let mut kernel_times = Vec::new();
        if let Some(f_reduce) = dev.get_func("reduction", "reduce_kernel") {
            for t in 0..num_trials() {
                let mut remaining = n as i32;
                let mut current_d = dev.htod_sync_copy(&h_input)?;
                
                let start = Instant::now();
                while remaining > 1 {
                    let num_blocks = (remaining + 256 * 2 - 1) / (256 * 2);
                    let mut out_d = dev.alloc_zeros::<f32>(num_blocks as usize)?;
                    let cfg = LaunchConfig {
                        grid_dim: (num_blocks as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe { f_reduce.clone().launch(cfg, (&current_d, &mut out_d, remaining)) }?;
                    current_d = out_d;
                    remaining = num_blocks;
                }
                dev.synchronize()?;
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                kernel_times.push(elapsed as f32);

            let gflops = (n - 1) as f64 / (elapsed * 1e-3) / 1e9;
            writeln!(csv, "cuda_reduction,rust_cuda,{},{},{:.4},0,0,{:.2},0",
                     n, t, elapsed, gflops)?;
        }

        let stats = compute_stats(&kernel_times);
        println!("    Kernel: {:.3} ± {:.3} ms", stats.mean, stats.stddev);
        }
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  ✓ All Rust GPU benchmarks complete!");
    println!("  Results: results/raw/gpu_rust.csv");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = seed;
    (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 11) as f64 / (1u64 << 53) as f64 * 6.0 - 3.0) as f32
        })
        .collect()
}

struct Stats {
    mean: f32,
    stddev: f32,
    ci95_low: f32,
    ci95_high: f32,
}

fn compute_stats(data: &[f32]) -> Stats {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (n - 1.0);
    let stddev = var.sqrt();
    let t_val = if data.len() >= 30 { 2.045_f32 } else { 2.776_f32 };
    let margin = t_val * stddev / n.sqrt();
    Stats {
        mean,
        stddev,
        ci95_low: mean - margin,
        ci95_high: mean + margin,
    }
}

/// CudaModule helper for loading PTX files.
/// This provides the framework — actual kernel launches need
/// the PTX files compiled via nvcc.
struct CudaModule;

impl CudaModule {
    fn from_ptx_file(path: &std::path::Path, module_name: &'static str, func_names: &[&'static str], dev: &Arc<CudaDevice>) -> Option<()> {
        if path.exists() {
            let ptx = std::fs::read_to_string(path).ok()?;
            dev.load_ptx(ptx.into(), module_name, func_names).ok()?;
            println!("  Loaded PTX: {}", path.display());
            Some(())
        } else {
            println!("  ⚠ PTX not found: {} (compile with nvcc -ptx first)", path.display());
            None
        }
    }
}
