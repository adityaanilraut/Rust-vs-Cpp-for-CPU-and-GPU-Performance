#!/usr/bin/env python3
"""
Dataset Generator for Rust vs. C++ Benchmark Study
===================================================
Generates deterministic synthetic datasets for all benchmarks.
Uses fixed random seeds for full reproducibility.

Usage:
    python3 generate_datasets.py          # Full-size datasets
    python3 generate_datasets.py --small   # Small datasets for quick testing
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────

MASTER_SEED = 42  # Reproducibility seed

# Full-size configurations
FULL_CONFIG = {
    "int_arrays": [1_000_000, 10_000_000, 100_000_000],       # 1M, 10M, 100M
    "f64_matrices": [512, 1024, 2048, 4096],                   # NxN
    "f32_matrices": [1024, 2048, 4096],                        # GPU NxN
    "f32_vectors": [1_000_000, 10_000_000, 100_000_000],       # Reduction
    "softmax_batches": [                                        # (batch, seq_len)
        (64, 16384),    # batch × 16K
        (64, 65536),    # batch × 64K
        (64, 262144),   # batch × 256K
    ],
}

# Small configurations for quick testing
SMALL_CONFIG = {
    "int_arrays": [10_000, 100_000],
    "f64_matrices": [64, 128],
    "f32_matrices": [128, 256],
    "f32_vectors": [10_000, 100_000],
    "softmax_batches": [
        (8, 1024),
        (8, 4096),
    ],
}


def sizeof_fmt(num_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def generate_int_arrays(config: dict, output_dir: Path, rng: np.random.Generator):
    """Generate random integer arrays for sorting benchmarks."""
    print("\n  ╭─ Integer Arrays (Sorting Benchmarks)")
    for size in config["int_arrays"]:
        filename = f"int_array_{size}.bin"
        filepath = output_dir / filename
        data = rng.integers(low=0, high=2**31, size=size, dtype=np.int32)
        data.tofile(filepath)
        print(f"  │  ✓ {filename:35s} {sizeof_fmt(filepath.stat().st_size):>10s}  ({size:>12,} elements)")
    print("  ╰─ Done")


def generate_f64_matrices(config: dict, output_dir: Path, rng: np.random.Generator):
    """Generate random f64 matrices for CPU matrix multiplication."""
    print("\n  ╭─ Float64 Matrices (CPU MatMul)")
    for n in config["f64_matrices"]:
        for label in ["A", "B"]:
            filename = f"f64_matrix_{n}x{n}_{label}.bin"
            filepath = output_dir / filename
            data = rng.standard_normal((n, n)).astype(np.float64)
            data.tofile(filepath)
            print(f"  │  ✓ {filename:35s} {sizeof_fmt(filepath.stat().st_size):>10s}  ({n}×{n})")
    print("  ╰─ Done")


def generate_f32_matrices(config: dict, output_dir: Path, rng: np.random.Generator):
    """Generate random f32 matrices for GPU matrix multiplication."""
    print("\n  ╭─ Float32 Matrices (GPU MatMul)")
    for n in config["f32_matrices"]:
        for label in ["A", "B"]:
            filename = f"f32_matrix_{n}x{n}_{label}.bin"
            filepath = output_dir / filename
            data = rng.standard_normal((n, n)).astype(np.float32)
            data.tofile(filepath)
            print(f"  │  ✓ {filename:35s} {sizeof_fmt(filepath.stat().st_size):>10s}  ({n}×{n})")
    print("  ╰─ Done")


def generate_f32_vectors(config: dict, output_dir: Path, rng: np.random.Generator):
    """Generate random f32 vectors for reduction benchmarks."""
    print("\n  ╭─ Float32 Vectors (Reduction)")
    for size in config["f32_vectors"]:
        filename = f"f32_vector_{size}.bin"
        filepath = output_dir / filename
        data = rng.standard_normal(size).astype(np.float32)
        data.tofile(filepath)
        print(f"  │  ✓ {filename:35s} {sizeof_fmt(filepath.stat().st_size):>10s}  ({size:>12,} elements)")
    print("  ╰─ Done")


def generate_softmax_batches(config: dict, output_dir: Path, rng: np.random.Generator):
    """Generate random f32 matrices for softmax benchmarks."""
    print("\n  ╭─ Float32 Softmax Batches")
    for batch, seq_len in config["softmax_batches"]:
        filename = f"f32_softmax_{batch}x{seq_len}.bin"
        filepath = output_dir / filename
        data = rng.standard_normal((batch, seq_len)).astype(np.float32)
        data.tofile(filepath)
        print(f"  │  ✓ {filename:35s} {sizeof_fmt(filepath.stat().st_size):>10s}  ({batch}×{seq_len})")
    print("  ╰─ Done")


def generate_f64_vectors(config: dict, output_dir: Path, rng: np.random.Generator):
    """Generate random f64 vectors for CPU parallel reduction."""
    print("\n  ╭─ Float64 Vectors (CPU Parallel Reduction)")
    sizes = config.get("f64_vectors", config["int_arrays"])  # Same sizes as int arrays
    for size in sizes:
        filename = f"f64_vector_{size}.bin"
        filepath = output_dir / filename
        data = rng.standard_normal(size).astype(np.float64)
        data.tofile(filepath)
        print(f"  │  ✓ {filename:35s} {sizeof_fmt(filepath.stat().st_size):>10s}  ({size:>12,} elements)")
    print("  ╰─ Done")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Generate small datasets for quick testing"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: datasets/ in project root)"
    )
    parser.add_argument(
        "--seed", type=int, default=MASTER_SEED,
        help=f"Random seed (default: {MASTER_SEED})"
    )
    args = parser.parse_args()

    # Determine paths
    project_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SMALL_CONFIG if args.small else FULL_CONFIG

    print("━" * 60)
    print("  Rust vs. C++ Benchmark — Dataset Generator")
    print("━" * 60)
    print(f"  Mode:   {'SMALL (test)' if args.small else 'FULL (production)'}")
    print(f"  Seed:   {args.seed}")
    print(f"  Output: {output_dir}")

    rng = np.random.default_rng(args.seed)
    start = time.time()

    generate_int_arrays(config, output_dir, rng)
    generate_f64_matrices(config, output_dir, rng)
    generate_f32_matrices(config, output_dir, rng)
    generate_f64_vectors(config, output_dir, rng)
    generate_f32_vectors(config, output_dir, rng)
    generate_softmax_batches(config, output_dir, rng)

    elapsed = time.time() - start

    # Write metadata
    metadata = {
        "seed": args.seed,
        "mode": "small" if args.small else "full",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {k: [x if isinstance(x, int) else list(x) for x in v] for k, v in config.items()},
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    total_size = sum(f.stat().st_size for f in output_dir.iterdir() if f.suffix == ".bin")
    print(f"\n{'━' * 60}")
    print(f"  ✓ All datasets generated in {elapsed:.1f}s")
    print(f"  Total size: {sizeof_fmt(total_size)}")
    print(f"  Metadata:   {meta_path}")
    print(f"{'━' * 60}")


if __name__ == "__main__":
    main()
