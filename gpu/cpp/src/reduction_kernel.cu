/**
 * CUDA Vector Reduction Kernel
 * =============================
 * Parallel reduction using shared memory + warp-level primitives.
 * Input sizes: 1M, 10M, 100M f32 elements.
 * Records kernel time and memory transfer time separately.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 256

// ── Parallel Reduction Kernel (shared memory + warp unroll) ─────────────────

__global__ void reduce_kernel(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;

    // Load two elements per thread (grid-stride)
    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + BLOCK_SIZE < n) sum += input[i + BLOCK_SIZE];
    sdata[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory
    for (unsigned int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction (no sync needed within a warp)
    if (tid < 32) {
        volatile float* vs = sdata;
        vs[tid] += vs[tid + 32];
        vs[tid] += vs[tid + 16];
        vs[tid] += vs[tid + 8];
        vs[tid] += vs[tid + 4];
        vs[tid] += vs[tid + 2];
        vs[tid] += vs[tid + 1];
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ── Multi-stage Reduction ───────────────────────────────────────────────────

struct ReductionResult {
    float kernel_time_ms;
    float h2d_time_ms;
    float d2h_time_ms;
    float gflops;
    float result_value;
};

extern "C" ReductionResult run_cuda_reduction(const float* h_input, int n) {
    ReductionResult result = {};
    size_t bytes = n * sizeof(float);

    float* d_input;
    cudaMalloc(&d_input, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // H2D
    cudaEventRecord(start);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.h2d_time_ms, start, stop);

    // Kernel — multi-stage reduction
    int remaining = n;
    float* d_current = d_input;
    float* d_output = nullptr;

    cudaEventRecord(start);

    while (remaining > 1) {
        int num_blocks = (remaining + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

        float* d_out;
        cudaMalloc(&d_out, num_blocks * sizeof(float));

        reduce_kernel<<<num_blocks, BLOCK_SIZE>>>(d_current, d_out, remaining);

        if (d_current != d_input) {
            cudaFree(d_current);
        }
        d_current = d_out;
        remaining = num_blocks;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.kernel_time_ms, start, stop);

    // D2H — just the single result value
    cudaEventRecord(start);
    cudaMemcpy(&result.result_value, d_current, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.d2h_time_ms, start, stop);

    // GFLOPS: n-1 additions
    result.gflops = (float)((double)(n - 1) / (result.kernel_time_ms * 1e-3) / 1e9);

    // Cleanup
    if (d_current != d_input) cudaFree(d_current);
    cudaFree(d_input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}
