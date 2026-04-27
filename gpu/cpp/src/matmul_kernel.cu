/**
 * CUDA Matrix Multiplication Kernel
 * ==================================
 * Tiled matrix multiplication using shared memory.
 * Sizes: 1024×1024, 2048×2048, 4096×4096 (f32).
 */

#include <cuda_runtime.h>
#include <cstdio>

#define TILE_SIZE 32

// ── Tiled MatMul Kernel ─────────────────────────────────────────────────────

__global__ void matmul_tiled(const float* A, const float* B, float* C,
                              int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B
        if ((t * TILE_SIZE + threadIdx.y) < N && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ── Naive MatMul Kernel (for comparison) ────────────────────────────────────

__global__ void matmul_naive(const float* A, const float* B, float* C,
                              int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ── Host Interface ──────────────────────────────────────────────────────────

struct MatMulResult {
    float kernel_time_ms;
    float h2d_time_ms;
    float d2h_time_ms;
    float gflops;
};

extern "C" MatMulResult run_cuda_matmul(const float* h_A, const float* h_B,
                                         float* h_C, int N, bool use_tiled) {
    MatMulResult result = {};
    size_t bytes = N * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ── Host to Device transfer ──
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.h2d_time_ms, start, stop);

    // ── Kernel execution ──
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEventRecord(start);
    if (use_tiled) {
        matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    } else {
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.kernel_time_ms, start, stop);

    // ── Device to Host transfer ──
    cudaEventRecord(start);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.d2h_time_ms, start, stop);

    // ── GFLOPS ──
    double flops = 2.0 * (double)N * N * N;
    result.gflops = (float)(flops / (result.kernel_time_ms * 1e-3) / 1e9);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}
