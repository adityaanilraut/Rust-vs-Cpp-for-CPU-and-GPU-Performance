// ============================================================================
// CUDA Vector Reduction Kernel (PTX source for Rust driver)
// ============================================================================

#define BLOCK_SIZE 256

extern "C"
__global__ void reduce_kernel(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;

    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + BLOCK_SIZE < n) sum += input[i + BLOCK_SIZE];
    sdata[tid] = sum;
    __syncthreads();

    // Tree reduction
    for (unsigned int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Warp-level unrolled reduction
    if (tid < 32) {
        volatile float* vs = sdata;
        vs[tid] += vs[tid + 32];
        vs[tid] += vs[tid + 16];
        vs[tid] += vs[tid + 8];
        vs[tid] += vs[tid + 4];
        vs[tid] += vs[tid + 2];
        vs[tid] += vs[tid + 1];
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}
