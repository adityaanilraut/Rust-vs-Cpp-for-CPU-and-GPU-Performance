# Rust vs. C++ for CPU and GPU Performance: A Controlled Empirical Benchmark Study

**Name:** Aditya Raut  
**GitHub Username:** adityaanilraut

## Abstract

High-performance computing (HPC) systems are fundamental to modern artificial intelligence, scientific simulation, and large-scale numerical workloads. C++ has traditionally dominated performance-critical systems due to mature compiler optimizations and CUDA-based GPU ecosystems. However, Rust has emerged as a systems programming language promising memory safety and data-race freedom without garbage collection or runtime overhead.

This research investigates how Rust compares to C++ across both CPU-based and CUDA-accelerated GPU workloads within a unified experimental framework. The primary research questions are: (1) How does Rust performance compare to C++ on CPU computational benchmarks? (2) How does Rust performance compare to C++ for CUDA-based GPU workloads? (3) What measurable trade-offs exist between safety guarantees and raw performance? The study employs controlled benchmarking, statistical validation, and developer-effort metrics to evaluate performance, safety, and feasibility for AI system development.
