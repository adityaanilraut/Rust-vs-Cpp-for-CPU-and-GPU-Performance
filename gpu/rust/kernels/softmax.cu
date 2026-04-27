// ============================================================================
// CUDA Softmax Kernel (PTX source for Rust driver)
// ============================================================================

#define WARP_SIZE 32

__device__ float warp_reduce_max_ptx(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ float warp_reduce_sum_ptx(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

extern "C"
__global__ void softmax_kernel(const float* input, float* output,
                                int batch_size, int seq_len) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_in  = input  + row * seq_len;
    float*       row_out = output + row * seq_len;

    extern __shared__ float shared[];
    float* s_max = shared;
    float* s_sum = shared + blockDim.x / WARP_SIZE;

    // Find row max
    float local_max = -3.402823466e+38f; // -FLT_MAX
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x)
        local_max = fmaxf(local_max, row_in[i]);

    local_max = warp_reduce_max_ptx(local_max);
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) s_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane < blockDim.x / WARP_SIZE) ? s_max[lane] : -3.402823466e+38f;
        local_max = warp_reduce_max_ptx(local_max);
    }
    __syncthreads();
    if (threadIdx.x == 0) s_max[0] = local_max;
    __syncthreads();
    float row_max = s_max[0];

    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float val = expf(row_in[i] - row_max);
        row_out[i] = val;
        local_sum += val;
    }

    local_sum = warp_reduce_sum_ptx(local_sum);
    if (lane == 0) s_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        local_sum = (lane < blockDim.x / WARP_SIZE) ? s_sum[lane] : 0.0f;
        local_sum = warp_reduce_sum_ptx(local_sum);
    }
    __syncthreads();
    if (threadIdx.x == 0) s_sum[0] = local_sum;
    __syncthreads();
    float row_sum = s_sum[0];

    // Normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x)
        row_out[i] *= inv_sum;
}
