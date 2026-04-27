// ============================================================================
// CUDA 2D Convolution Kernel (PTX source for Rust driver)
// ============================================================================
// Naive convolution for PTX compilation. Loaded by Rust at runtime.
// nvcc -ptx conv2d.cu -o conv2d.ptx
// ============================================================================

extern "C"
__global__ void conv2d_naive(const float* input, const float* filter,
                              float* output, int width, int height,
                              int filter_radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width) return;

    float sum = 0.0f;
    int fsize = 2 * filter_radius + 1;

    for (int fy = -filter_radius; fy <= filter_radius; ++fy) {
        for (int fx = -filter_radius; fx <= filter_radius; ++fx) {
            int iy = min(max(row + fy, 0), height - 1);
            int ix = min(max(col + fx, 0), width - 1);
            int fidx = (fy + filter_radius) * fsize + (fx + filter_radius);
            sum += input[iy * width + ix] * filter[fidx];
        }
    }

    output[row * width + col] = sum;
}
