// ============================================================================
// CUDA Prefix Sum / Scan Kernel (PTX source for Rust driver)
// ============================================================================
// Blelloch scan for PTX compilation. Loaded by Rust at runtime.
// nvcc -ptx prefix_sum.cu -o prefix_sum.ptx
// ============================================================================

#define BLOCK_SIZE 256

extern "C"
__global__ void blelloch_scan_block(float* data, float* block_sums, int n) {
    __shared__ float temp[BLOCK_SIZE * 2];

    int tid = threadIdx.x;
    int offset = 1;
    int block_offset = blockIdx.x * (BLOCK_SIZE * 2);

    int ai = tid;
    int bi = tid + BLOCK_SIZE;
    int global_ai = block_offset + ai;
    int global_bi = block_offset + bi;

    temp[ai] = (global_ai < n) ? data[global_ai] : 0.0f;
    temp[bi] = (global_bi < n) ? data[global_bi] : 0.0f;

    // Up-sweep
    for (int d = BLOCK_SIZE; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai_idx = offset * (2 * tid + 1) - 1;
            int bi_idx = offset * (2 * tid + 2) - 1;
            temp[bi_idx] += temp[ai_idx];
        }
        offset *= 2;
    }

    if (tid == 0) {
        if (block_sums) block_sums[blockIdx.x] = temp[BLOCK_SIZE * 2 - 1];
        temp[BLOCK_SIZE * 2 - 1] = 0.0f;
    }

    // Down-sweep
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

    if (global_ai < n) data[global_ai] = temp[ai];
    if (global_bi < n) data[global_bi] = temp[bi];
}

extern "C"
__global__ void add_block_sums(float* data, float* block_sums, int n) {
    int idx = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;
    float block_sum = block_sums[blockIdx.x];

    if (idx < n) data[idx] += block_sum;
    if (idx + BLOCK_SIZE < n) data[idx + BLOCK_SIZE] += block_sum;
}
