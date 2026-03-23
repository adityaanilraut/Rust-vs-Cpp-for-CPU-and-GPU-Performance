cmake_minimum_required(VERSION 3.18)
project(RustVsCpp_GPU_Benchmarks LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ── CUDA Architecture ───────────────────────────────────────────────────────
# Auto-detect GPU architecture, or set manually:
# set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3090
# set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4090
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# ── Optimization Flags ──────────────────────────────────────────────────────
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")

# ── Dataset path ─────────────────────────────────────────────────────────────
set(DATASET_DIR "${CMAKE_SOURCE_DIR}/../../datasets" CACHE PATH "Path to datasets")
set(RESULTS_DIR "${CMAKE_SOURCE_DIR}/../../results/raw" CACHE PATH "Path to results")
add_compile_definitions(DATASET_DIR="${DATASET_DIR}" RESULTS_DIR="${RESULTS_DIR}")

# ── GPU Benchmark Executable ────────────────────────────────────────────────
add_executable(gpu_benchmarks
    src/matmul_kernel.cu
    src/softmax_kernel.cu
    src/reduction_kernel.cu
    src/benchmark_runner.cu
)

target_include_directories(gpu_benchmarks PRIVATE ${CMAKE_SOURCE_DIR}/../cpp/include)
