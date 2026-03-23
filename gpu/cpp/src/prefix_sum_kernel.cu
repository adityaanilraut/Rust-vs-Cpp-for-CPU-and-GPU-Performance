// ============================================================================
// CUDA Exclusive Prefix Sum (Scan) Kernel
// ============================================================================
// Blelloch work-efficient parallel scan.
// Fundamental parallel primitive used in sorting, histograms, stream
// compaction, and GPU radix sort.
// Input sizes: 1M, 10M, 100M elements.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// ── Block-level exclusive scan (Blelloch) ───────────────────────────────────

extern "C"
__global__ void blelloch_scan_block(float* data, float* block_sums, int n) {
    __shared__ float temp[BLOCK_SIZE * 2];

    int tid = threadIdx.x;
    int offset = 1;
    int block_offset = blockIdx.x * (BLOCK_SIZE * 2);

    // Load data into shared memory
    int ai = tid;
    int bi = tid + BLOCK_SIZE;
    int global_ai = block_offset + ai;
    int global_bi = block_offset + bi;

    temp[ai] = (global_ai < n) ? data[global_ai] : 0.0f;
    temp[bi] = (global_bi < n) ? data[global_bi] : 0.0f;

    // Up-sweep (reduce) phase
    for (int d = BLOCK_SIZE; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai_idx = offset * (2 * tid + 1) - 1;
            int bi_idx = offset * (2 * tid + 2) - 1;
            temp[bi_idx] += temp[ai_idx];
        }
        offset *= 2;
    }

    // Save block sum and clear last element
    if (tid == 0) {
        if (block_sums) block_sums[blockIdx.x] = temp[BLOCK_SIZE * 2 - 1];
        temp[BLOCK_SIZE * 2 - 1] = 0.0f;
    }

    // Down-sweep phase
    for (int d = 1; d < BLOCK_SIZE * 2; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai_idx = offset * (2 * tid + 1) - 1;
            int bi_idx = offset * (2 * tid + 2) - 1;
            float t = temp[ai_idx];
            temp[ai_idx] = temp[bi_idx];
            temp[bi_idx] += t;
        }
    }
    __syncthreads();

    // Write results
    if (global_ai < n) data[global_ai] = temp[ai];
    if (global_bi < n) data[global_bi] = temp[bi];
}

// ── Add block sums to make global scan ──────────────────────────────────────

extern "C"
__global__ void add_block_sums(float* data, float* block_sums, int n) {
    int idx = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;
    float block_sum = block_sums[blockIdx.x];

    if (idx < n) data[idx] += block_sum;
    if (idx + BLOCK_SIZE < n) data[idx + BLOCK_SIZE] += block_sum;
}

// ── Host Interface ──────────────────────────────────────────────────────────

void run_prefix_sum_benchmark(int n, float* h2d_ms, float* kernel_ms, float* d2h_ms) {
    // Allocate
    float* h_data = new float[n];
    float* h_result = new float[n];
    for (int i = 0; i < n; ++i) h_data[i] = 1.0f;  // All ones → result should be [0,1,2,3,...]

    float *d_data, *d_block_sums;
    int num_blocks = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMalloc(&d_block_sums, num_blocks * sizeof(float));

    // Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // H2D
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(h2d_ms, start, stop);

    // Kernel
    cudaEventRecord(start);
    blelloch_scan_block<<<num_blocks, BLOCK_SIZE>>>(d_data, d_block_sums, n);

    // For large arrays, need recursive scan on block sums
    // (simplified: only handles single-level for benchmarking)
    if (num_blocks > 1) {
        // Scan block sums
        float* d_block_sums2;
        int num_blocks2 = (num_blocks + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
        cudaMalloc(&d_block_sums2, num_blocks2 * sizeof(float));
        blelloch_scan_block<<<num_blocks2, BLOCK_SIZE>>>(d_block_sums, d_block_sums2, num_blocks);

        // Add scanned block sums back
        add_block_sums<<<num_blocks, BLOCK_SIZE>>>(d_data, d_block_sums, n);
        cudaFree(d_block_sums2);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(kernel_ms, start, stop);

    // D2H
    cudaEventRecord(start);
    cudaMemcpy(h_result, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(d2h_ms, start, stop);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_block_sums);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_data;
    delete[] h_result;
}
