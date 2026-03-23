# Rust vs. C++ for CPU and GPU Performance

This repository contains a controlled benchmark study comparing Rust and C++ across CPU workloads and CUDA-accelerated GPU workloads. The project includes source implementations, dataset-generation utilities, benchmark automation, and the raw analysis outputs used for the study.

## Research Scope

The study focuses on three questions:

1. How does Rust performance compare to C++ on CPU computational benchmarks?
2. How does Rust performance compare to C++ for CUDA-based GPU workloads?
3. What trade-offs appear between Rust's safety guarantees and C++'s mature performance ecosystem?

## Repository Layout

- `cpu/cpp`: C++ CPU benchmarks built with CMake and Google Benchmark
- `cpu/rust`: Rust CPU benchmarks built with Cargo
- `gpu/cpp`: CUDA/C++ GPU benchmarks
- `gpu/rust`: Rust GPU driver code plus CUDA kernels compiled to PTX
- `scripts`: dataset generation, analysis, export, plotting, and safety-analysis helpers
- `results/raw`: committed raw benchmark and analysis outputs
- `datasets`: generated local input datasets for reproducible runs
- `papers`: paper-related materials from the original repository history

## Included Results

The repository keeps the current benchmark outputs and raw analysis artifacts under `results/raw`, including:

- CPU benchmark exports for Rust and C++
- GPU benchmark exports for Rust and C++
- Google Benchmark JSON output
- Statistical summary tables
- Safety and developer-metrics analysis output

These files are committed so the current study outputs can be inspected without rerunning benchmarks.

## Dataset Policy

The large binary files in `datasets/` are generated inputs rather than source files. They are intentionally excluded from normal Git tracking because they are reproducible and several exceed GitHub's regular file-size limits. Use the provided generation scripts to recreate them locally when needed.

## Quick Start

Install Python dependencies:

```bash
make setup
```

Generate datasets locally:

```bash
make datasets
```

Build CPU and GPU benchmarks:

```bash
make cpu-cpp
make cpu-rust
make gpu-cpp
make gpu-rust
```

Run analysis on existing results:

```bash
make analyze
make plots
```

## Notes

- No benchmarks need to be rerun to inspect the committed outputs already present in `results/raw`.
- Build directories and Cargo targets are ignored so the repository stays focused on source, scripts, and results.
