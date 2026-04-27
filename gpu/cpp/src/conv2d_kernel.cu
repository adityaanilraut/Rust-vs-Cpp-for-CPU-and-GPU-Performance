// ============================================================================
// CUDA 2D Convolution Kernel
// ============================================================================
// Tiled 2D convolution using shared memory — AI-relevant kernel.
// 3×3, 5×5, and 7×7 filter sizes on 1024×1024, 2048×2048 images.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define MAX_FILTER_RADIUS 3  // max 7×7 filter

// ── Naive Convolution (baseline) ────────────────────────────────────────────

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

// ── Tiled Convolution (constant memory filter) ─────────────────────────────

__constant__ float d_filter[49];  // Max 7×7

extern "C"
__global__ void conv2d_tiled(const float* input, float* output,
                              int width, int height, int filter_radius) {
    int tile_w = TILE_SIZE + 2 * filter_radius;
    extern __shared__ float tile[];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load tile including halo
    for (int ty = threadIdx.y; ty < tile_w; ty += blockDim.y) {
        for (int tx = threadIdx.x; tx < tile_w; tx += blockDim.x) {
            int iy = blockIdx.y * TILE_SIZE + ty - filter_radius;
            int ix = blockIdx.x * TILE_SIZE + tx - filter_radius;
            iy = min(max(iy, 0), height - 1);
            ix = min(max(ix, 0), width - 1);
            tile[ty * tile_w + tx] = input[iy * width + ix];
        }
    }
    __syncthreads();

    if (row >= height || col >= width) return;

    float sum = 0.0f;
    int fsize = 2 * filter_radius + 1;

    for (int fy = 0; fy < fsize; ++fy) {
        for (int fx = 0; fx < fsize; ++fx) {
            sum += tile[(threadIdx.y + fy) * tile_w + (threadIdx.x + fx)]
                 * d_filter[fy * fsize + fx];
        }
    }

    output[row * width + col] = sum;
}


// ── Host Interface ──────────────────────────────────────────────────────────

void run_conv2d_benchmark(int width, int height, int filter_radius,
                           float* h2d_ms, float* kernel_ms, float* d2h_ms) {
    int n = width * height;
    int fsize = 2 * filter_radius + 1;

    // Allocate and initialize
    float* h_input  = new float[n];
    float* h_output = new float[n];
    float* h_filter = new float[fsize * fsize];

    for (int i = 0; i < n; ++i) h_input[i] = (float)(rand() % 256) / 255.0f;
    float filter_sum = 0.0f;
    for (int i = 0; i < fsize * fsize; ++i) {
        h_filter[i] = 1.0f / (fsize * fsize);  // Box filter (normalized)
        filter_sum += h_filter[i];
    }

    // Device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // Upload filter to constant memory
    cudaMemcpyToSymbol(d_filter, h_filter, fsize * fsize * sizeof(float));

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // H2D
    cudaEventRecord(start);
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(h2d_ms, start, stop);

    // Kernel
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE,
              (height + TILE_SIZE - 1) / TILE_SIZE);
    int tile_w = TILE_SIZE + 2 * filter_radius;
    int shared_mem = tile_w * tile_w * sizeof(float);

    cudaEventRecord(start);
    conv2d_tiled<<<grid, block, shared_mem>>>(d_input, d_output, width, height, filter_radius);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(kernel_ms, start, stop);

    // D2H
    cudaEventRecord(start);
    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(d2h_ms, start, stop);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_input;
    delete[] h_output;
    delete[] h_filter;
}
