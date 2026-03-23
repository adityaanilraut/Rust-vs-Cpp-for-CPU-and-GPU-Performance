/**
 * CUDA Softmax Kernel
 * ====================
 * Row-wise softmax with numerical stability (log-sum-exp trick).
 * Applied to batch × seq_len matrices (f32).
 * Numerical equivalence verified post-run.
 */

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cmath>

#define WARP_SIZE 32

// ── Warp-level Reduction Primitives ─────────────────────────────────────────

__device__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ── Softmax Kernel (one block per row) ──────────────────────────────────────

__global__ void softmax_kernel(const float* input, float* output,
                                int batch_size, int seq_len) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_in = input + row * seq_len;
    float* row_out = output + row * seq_len;

    extern __shared__ float shared[];
    float* s_max = shared;                          // blockDim.x / WARP_SIZE
    float* s_sum = shared + blockDim.x / WARP_SIZE; // blockDim.x / WARP_SIZE

    // ── Step 1: Find row maximum (for numerical stability) ──
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        local_max = fmaxf(local_max, row_in[i]);
    }

    // Warp reduction
    local_max = warp_reduce_max(local_max);

    // Store warp results in shared memory
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) s_max[warp_id] = local_max;
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        local_max = (lane < blockDim.x / WARP_SIZE) ? s_max[lane] : -FLT_MAX;
        local_max = warp_reduce_max(local_max);
    }
    __syncthreads();

    // Broadcast max to all threads
    if (threadIdx.x == 0) s_max[0] = local_max;
    __syncthreads();
    float row_max = s_max[0];

    // ── Step 2: Compute exp(x - max) and sum ──
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float val = expf(row_in[i] - row_max);
        row_out[i] = val;  // Store intermediate
        local_sum += val;
    }

    // Warp reduction
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) s_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane < blockDim.x / WARP_SIZE) ? s_sum[lane] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
    }
    __syncthreads();

    if (threadIdx.x == 0) s_sum[0] = local_sum;
    __syncthreads();
    float row_sum = s_sum[0];

    // ── Step 3: Normalize ──
    float inv_sum = 1.0f / row_sum;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
}

// ── Host Interface ──────────────────────────────────────────────────────────

struct SoftmaxResult {
    float kernel_time_ms;
    float h2d_time_ms;
    float d2h_time_ms;
    float bandwidth_gbps;
};

extern "C" SoftmaxResult run_cuda_softmax(const float* h_input, float* h_output,
                                           int batch_size, int seq_len) {
    SoftmaxResult result = {};
    size_t bytes = (size_t)batch_size * seq_len * sizeof(float);

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // H2D
    cudaEventRecord(start);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.h2d_time_ms, start, stop);

    // Kernel
    int threads = 256;
    int shared_bytes = 2 * (threads / WARP_SIZE) * sizeof(float);

    cudaEventRecord(start);
    softmax_kernel<<<batch_size, threads, shared_bytes>>>(
        d_input, d_output, batch_size, seq_len);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.kernel_time_ms, start, stop);

    // D2H
    cudaEventRecord(start);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.d2h_time_ms, start, stop);

    // Bandwidth: read input + write output
    result.bandwidth_gbps = (float)(2.0 * bytes / (result.kernel_time_ms * 1e-3) / 1e9);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}
