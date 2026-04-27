# ============================================================================
# Rust vs. C++ Benchmark Study — Top-Level Makefile
# ============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Colors for pretty output
CYAN  := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

# ── Directories ──────────────────────────────────────────────────────────────
DATASETS_DIR   := datasets
RESULTS_RAW    := results/raw
RESULTS_FIG    := results/figures
CPU_CPP_DIR    := cpu/cpp
CPU_RUST_DIR   := cpu/rust
GPU_CPP_DIR    := gpu/cpp
GPU_RUST_DIR   := gpu/rust
SCRIPTS_DIR    := scripts

# ── Phony Targets ────────────────────────────────────────────────────────────
.PHONY: help setup datasets cpu-cpp cpu-rust gpu-cpp gpu-rust \
        benchmarks-cpu benchmarks-gpu benchmarks-all \
        analyze plots clean clean-all

help: ## Show this help
	@echo ""
	@echo "$(CYAN)Rust vs. C++ Benchmark Study$(RESET)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────────────
setup: ## Install Python dependencies
	pip install -r $(SCRIPTS_DIR)/requirements.txt

# ── Dataset Generation ───────────────────────────────────────────────────────
datasets: ## Generate all synthetic datasets
	@echo "$(YELLOW)▸ Generating datasets...$(RESET)"
	python3 $(SCRIPTS_DIR)/generate_datasets.py
	@echo "$(GREEN)✓ Datasets generated in $(DATASETS_DIR)/$(RESET)"

datasets-small: ## Generate small datasets for testing
	@echo "$(YELLOW)▸ Generating small test datasets...$(RESET)"
	python3 $(SCRIPTS_DIR)/generate_datasets.py --small
	@echo "$(GREEN)✓ Small datasets generated$(RESET)"

# ── CPU Benchmarks ───────────────────────────────────────────────────────────
cpu-cpp: ## Build C++ CPU benchmarks
	@echo "$(YELLOW)▸ Building C++ CPU benchmarks...$(RESET)"
	mkdir -p $(CPU_CPP_DIR)/build
	cd $(CPU_CPP_DIR)/build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$$(nproc)
	@echo "$(GREEN)✓ C++ CPU benchmarks built$(RESET)"

cpu-rust: ## Build Rust CPU benchmarks
	@echo "$(YELLOW)▸ Building Rust CPU benchmarks...$(RESET)"
	cd $(CPU_RUST_DIR) && cargo build --release
	@echo "$(GREEN)✓ Rust CPU benchmarks built$(RESET)"

# ── GPU Benchmarks ───────────────────────────────────────────────────────────
gpu-cpp: ## Build C++/CUDA GPU benchmarks
	@echo "$(YELLOW)▸ Building C++/CUDA GPU benchmarks...$(RESET)"
	mkdir -p $(GPU_CPP_DIR)/build
	cd $(GPU_CPP_DIR)/build && cmake .. && make -j$$(nproc)
	@echo "$(GREEN)✓ C++/CUDA GPU benchmarks built$(RESET)"

gpu-rust: ## Build Rust/CUDA GPU benchmarks
	@echo "$(YELLOW)▸ Compiling PTX kernels...$(RESET)"
	cd $(GPU_RUST_DIR) && \
		nvcc -ptx kernels/matmul.cu -o kernels/matmul.ptx && \
		nvcc -ptx kernels/softmax.cu -o kernels/softmax.ptx && \
		nvcc -ptx kernels/reduction.cu -o kernels/reduction.ptx
	@echo "$(YELLOW)▸ Building Rust GPU benchmarks...$(RESET)"
	cd $(GPU_RUST_DIR) && cargo build --release
	@echo "$(GREEN)✓ Rust/CUDA GPU benchmarks built$(RESET)"

# ── Run Benchmarks ───────────────────────────────────────────────────────────
benchmarks-cpu: cpu-cpp cpu-rust ## Build and run all CPU benchmarks
	@echo "$(YELLOW)▸ Running CPU benchmarks (30 trials)...$(RESET)"
	mkdir -p $(RESULTS_RAW)
	cd $(CPU_CPP_DIR)/build && ./cpu_benchmarks \
		--benchmark_out=../../../$(RESULTS_RAW)/cpu_cpp.json \
		--benchmark_out_format=json
	cd $(CPU_RUST_DIR) && cargo bench
	@echo "$(GREEN)✓ CPU benchmarks complete$(RESET)"

benchmarks-gpu: gpu-cpp gpu-rust ## Build and run all GPU benchmarks
	@echo "$(YELLOW)▸ Running GPU benchmarks (30 trials)...$(RESET)"
	mkdir -p $(RESULTS_RAW)
	cd $(GPU_CPP_DIR)/build && ./gpu_benchmarks
	cd $(GPU_RUST_DIR) && cargo run --release
	@echo "$(GREEN)✓ GPU benchmarks complete$(RESET)"

benchmarks-all: benchmarks-cpu benchmarks-gpu ## Run all benchmarks

# ── Analysis ─────────────────────────────────────────────────────────────────
analyze: ## Run statistical analysis on results
	@echo "$(YELLOW)▸ Running statistical analysis...$(RESET)"
	python3 $(SCRIPTS_DIR)/analyze_results.py
	@echo "$(GREEN)✓ Analysis complete — see $(RESULTS_RAW)/$(RESET)"

plots: ## Generate publication-quality plots
	@echo "$(YELLOW)▸ Generating plots...$(RESET)"
	mkdir -p $(RESULTS_FIG)
	python3 $(SCRIPTS_DIR)/generate_plots.py
	@echo "$(GREEN)✓ Plots saved to $(RESULTS_FIG)/$(RESET)"

safety: ## Run safety & developer metrics analysis
	@echo "$(YELLOW)▸ Analyzing safety metrics...$(RESET)"
	python3 $(SCRIPTS_DIR)/safety_analysis.py
	@echo "$(GREEN)✓ Safety analysis complete$(RESET)"

# ── Paper ────────────────────────────────────────────────────────────────────
paper: ## Compile LaTeX paper
	@echo "$(YELLOW)▸ Compiling paper...$(RESET)"
	cd papers && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
	@echo "$(GREEN)✓ papers/main.pdf generated$(RESET)"

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts
	rm -rf $(CPU_CPP_DIR)/build
	rm -rf $(GPU_CPP_DIR)/build
	cd $(CPU_RUST_DIR) && cargo clean 2>/dev/null || true
	cd $(GPU_RUST_DIR) && cargo clean 2>/dev/null || true

clean-all: clean ## Remove everything (builds + datasets + results)
	rm -rf $(DATASETS_DIR)/*.bin $(DATASETS_DIR)/*.npy
	rm -rf $(RESULTS_RAW)/*.csv $(RESULTS_RAW)/*.json
	rm -rf $(RESULTS_FIG)/*.png $(RESULTS_FIG)/*.pdf
