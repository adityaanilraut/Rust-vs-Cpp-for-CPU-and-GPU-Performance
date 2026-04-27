# Rust vs. C++ for CPU and GPU Performance

A controlled benchmark study comparing Rust and C++ across CPU workloads and CUDA-accelerated GPU workloads. The repository ships source implementations, dataset-generation utilities, benchmark automation, statistical analysis, and the raw outputs used for the accompanying paper.

## Research Scope

The study addresses three questions:

1. How does Rust perform relative to C++ on CPU computational benchmarks?
2. How does Rust compare to C++ for CUDA-based GPU workloads?
3. What trade-offs appear between Rust's safety guarantees and C++'s mature performance ecosystem?

## Repository Layout

```
.
├── cpu/
│   ├── cpp/        # C++ CPU benchmarks (CMake + Google Benchmark)
│   └── rust/       # Rust CPU benchmarks (Cargo + Criterion)
├── gpu/
│   ├── cpp/        # CUDA/C++ GPU benchmarks (CMake)
│   └── rust/       # Rust GPU driver + CUDA kernels (cudarc, .ptx)
├── scripts/        # Dataset generation, analysis, plotting, safety metrics
├── results/
│   ├── raw/        # Committed CSV / JSON / TeX analysis outputs
│   └── figures/    # Committed PNG / PDF plots
├── papers/         # LaTeX sources, figures, bibliography, prior drafts
├── datasets/       # metadata.json only — .bin files are regenerated locally
├── Makefile        # Top-level orchestration
└── run_benchmarks.ps1
```

## Prerequisites

| Component       | Tested version                          |
|-----------------|-----------------------------------------|
| OS              | Ubuntu 24.04 (WSL2) or native Linux     |
| Rust toolchain  | rustc 1.95.0 (cargo + rustup)           |
| C++ toolchain   | gcc 13.3.0 / clang 18+                  |
| CMake           | 3.28+                                   |
| CUDA Toolkit    | 12.6 (`nvcc` 12.6.85)                   |
| NVIDIA driver   | 592.01 (compute capability >= 7.0)      |
| Python          | 3.10+                                   |

GPU benchmarks require a CUDA-capable NVIDIA GPU. Under WSL2, ensure the appropriate Windows-side driver is installed and that `LD_LIBRARY_PATH` includes `/usr/lib/wsl/lib`.

## Quick Start

```bash
# 1. Install Python tooling
make setup

# 2. Regenerate the synthetic input datasets (~2.3 GB; deterministic)
make datasets

# 3. Build everything
make cpu-cpp cpu-rust gpu-cpp gpu-rust

# 4. Run the full suite (writes to results/raw/)
make benchmarks-all

# 5. Analyse and plot
make analyze
make plots
make safety
```

Run `make help` for the full list of targets.

## Reproducibility Notes

- **Datasets are not tracked.** They are deterministic outputs of `scripts/generate_datasets.py` (seed 42) and several files exceed GitHub's per-file size limits. `datasets/metadata.json` is committed so the exact configuration used for the paper is preserved.
- **Build trees are not tracked.** `cpu/{cpp,rust}/build|target/` and `gpu/{cpp,rust}/build|target/` are produced on demand; remove them with `make clean`.
- **Pre-compiled PTX kernels are tracked.** `gpu/rust/kernels/*.ptx` is shipped so the Rust GPU driver runs without requiring `nvcc` at startup. Rebuild them with `make gpu-rust`.
- **Raw results are tracked.** Everything under `results/raw/` and `results/figures/` reflects the latest run reported in the paper, so consumers can inspect outcomes without re-running benchmarks.
- **Paper artefacts.** `papers/main.tex`, `papers/preamble.tex`, `papers/paper.bib`, and `papers/figures/` are the inputs for the Overleaf build. `papers/main.pdf` is the latest compiled PDF; older drafts in `papers/` are kept for historical reference.

## Cleaning

```bash
make clean       # remove build artefacts
make clean-all   # also remove datasets and results
```

## License

See repository metadata for licensing details.
