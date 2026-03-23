//! Utility functions for loading datasets and recording results.

use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Load a binary file as a Vec<T>.
/// Files are raw binary with no header — element count inferred from file size.
pub fn load_binary<T: bytemuck::Pod>(path: &Path) -> Vec<T> {
    let mut file = File::open(path).unwrap_or_else(|_| {
        eprintln!("ERROR: Cannot open dataset: {}", path.display());
        eprintln!("       Run 'python3 scripts/generate_datasets.py' first.");
        std::process::exit(1);
    });

    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("Failed to read file");

    let elem_size = std::mem::size_of::<T>();
    assert!(
        bytes.len() % elem_size == 0,
        "File size {} is not a multiple of element size {}",
        bytes.len(),
        elem_size
    );

    bytemuck::allocation::cast_vec(bytes)
}

/// Generate a deterministic random i32 array using a simple LCG.
/// Matches the distribution from the Python generator (roughly).
pub fn generate_random_i32(size: usize, seed: u64) -> Vec<i32> {
    let mut rng = seed;
    (0..size)
        .map(|_| {
            // Simple LCG: x = (a*x + c) mod m
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 33) as i32
        })
        .collect()
}

/// Generate a deterministic random f64 array.
pub fn generate_random_f64(size: usize, seed: u64) -> Vec<f64> {
    let mut rng = seed;
    (0..size)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Map to roughly [-3, 3] range (not true normal, but deterministic)
            ((rng >> 11) as f64 / (1u64 << 53) as f64) * 6.0 - 3.0
        })
        .collect()
}

/// Get the dataset directory path.
pub fn dataset_dir() -> std::path::PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir).join("../../datasets")
}
