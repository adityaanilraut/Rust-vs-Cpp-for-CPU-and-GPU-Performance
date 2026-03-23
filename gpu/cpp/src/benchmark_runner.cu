/**
 * GPU Benchmark Runner
 * =====================
 * Orchestrates all GPU benchmarks with proper timing, warm-up,
 * thermal monitoring, and CSV output.
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <chrono>
#include <thread>
#include <vector>

// ── External kernel interfaces ──────────────────────────────────────────────

struct MatMulResult {
    float kernel_time_ms;
    float h2d_time_ms;
    float d2h_time_ms;
    float gflops;
};

struct SoftmaxResult {
    float kernel_time_ms;
    float h2d_time_ms;
    float d2h_time_ms;
    float bandwidth_gbps;
};

struct ReductionResult {
    float kernel_time_ms;
    float h2d_time_ms;
    float d2h_time_ms;
    float gflops;
    float result_value;
};

extern "C" MatMulResult run_cuda_matmul(const float* h_A, const float* h_B,
                                         float* h_C, int N, bool use_tiled);
extern "C" SoftmaxResult run_cuda_softmax(const float* h_input, float* h_output,
                                           int batch_size, int seq_len);
extern "C" ReductionResult run_cuda_reduction(const float* h_input, int n);

// ── Configuration ───────────────────────────────────────────────────────────

constexpr int NUM_TRIALS = 30;
constexpr int WARMUP_RUNS = 3;
constexpr float TEMP_THRESHOLD = 80.0f;
constexpr int COOLDOWN_MS = 10000;

int num_trials() {
    return NUM_TRIALS;
}

int warmup_runs() {
    return WARMUP_RUNS;
}

// ── Utilities ───────────────────────────────────────────────────────────────

void print_header(const char* name) {
    printf("\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  %s\n", name);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

void print_gpu_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\n  GPU: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM Count: %d\n", prop.multiProcessorCount);
    printf("  Memory: %.0f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("  Memory bus: %d-bit\n", prop.memoryBusWidth);
    printf("\n");
}

float get_gpu_temp() {
    // Try nvidia-smi to get temperature
    FILE* pipe = popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null", "r");
    if (pipe) {
        char buffer[32];
        if (fgets(buffer, sizeof(buffer), pipe)) {
            pclose(pipe);
            return atof(buffer);
        }
        pclose(pipe);
    }
    return 0.0f; // Unknown
}

void enforce_cooldown() {
    float temp = get_gpu_temp();
    if (temp > TEMP_THRESHOLD) {
        printf("  ⚠ GPU temp %.0f°C > %.0f°C — cooling down %ds...\n",
               temp, TEMP_THRESHOLD, COOLDOWN_MS / 1000);
        cudaDeviceSynchronize();

        // Busy wait with monitoring
        for (int i = 0; i < COOLDOWN_MS / 1000; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            temp = get_gpu_temp();
            if (temp <= TEMP_THRESHOLD - 5) break;
        }
        printf("  ✓ GPU temp now %.0f°C\n", temp);
    }
}

struct Stats {
    float mean;
    float stddev;
    float ci95_low;
    float ci95_high;
};

Stats compute_stats(const std::vector<float>& data) {
    Stats s = {};
    int n = data.size();
    if (n == 0) return s;

    double sum = 0;
    for (auto v : data) sum += v;
    s.mean = sum / n;

    double var_sum = 0;
    for (auto v : data) var_sum += (v - s.mean) * (v - s.mean);
    s.stddev = sqrt(var_sum / (n - 1));

    float t_val = n >= 30 ? 2.045f : 2.776f;
    float margin = t_val * s.stddev / sqrt((float)n);
    s.ci95_low = s.mean - margin;
    s.ci95_high = s.mean + margin;

    return s;
}

// ── CSV Output ──────────────────────────────────────────────────────────────

class CSVWriter {
    std::ofstream file;
public:
    CSVWriter(const std::string& path) {
        file.open(path);
        file << "benchmark,language,input_size,trial,kernel_time_ms,"
             << "h2d_time_ms,d2h_time_ms,gflops,bandwidth_gbps\n";
    }

    void write(const std::string& benchmark, const std::string& input_size,
               int trial, float kernel, float h2d, float d2h,
               float gflops, float bw) {
        file << benchmark << ",cpp_cuda," << input_size << "," << trial << ","
             << std::fixed << std::setprecision(4)
             << kernel << "," << h2d << "," << d2h << ","
             << std::setprecision(2) << gflops << "," << bw << "\n";
    }

    ~CSVWriter() { file.close(); }
};

// ── Random Data Generation ──────────────────────────────────────────────────

std::vector<float> random_f32(size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(n);
    for (auto& v : data) v = dist(rng);
    return data;
}

// ── Main ────────────────────────────────────────────────────────────────────

int main() {
    print_header("Rust vs. C++ GPU Benchmark Suite (C++/CUDA)");
    print_gpu_info();

    std::string results_dir = "../../results/raw/";
    #ifdef RESULTS_DIR
    results_dir = std::string(RESULTS_DIR) + "/";
    #endif

    CSVWriter csv(results_dir + "gpu_cpp.csv");

    // ────────────────────────────────────────────────────────────────
    // 1. Matrix Multiplication
    // ────────────────────────────────────────────────────────────────
    {
        print_header("CUDA Matrix Multiplication (Tiled)");

        const std::vector<int> matmul_sizes{1024, 2048, 4096};

        for (int N : matmul_sizes) {
            printf("  Size: %d×%d\n", N, N);

            auto h_A = random_f32(N * N, 42);
            auto h_B = random_f32(N * N, 43);
            std::vector<float> h_C(N * N);

            // Warm-up
            for (int i = 0; i < warmup_runs(); ++i) {
                run_cuda_matmul(h_A.data(), h_B.data(), h_C.data(), N, true);
            }

            // Benchmark
            std::vector<float> kernel_times;
            for (int t = 0; t < num_trials(); ++t) {
                enforce_cooldown();
                auto r = run_cuda_matmul(h_A.data(), h_B.data(), h_C.data(), N, true);
                kernel_times.push_back(r.kernel_time_ms);
                csv.write("cuda_matmul_tiled", std::to_string(N),
                          t, r.kernel_time_ms, r.h2d_time_ms, r.d2h_time_ms,
                          r.gflops, 0);
            }

            auto stats = compute_stats(kernel_times);
            printf("    Kernel: %.3f ± %.3f ms  [CI: %.3f – %.3f]\n",
                   stats.mean, stats.stddev, stats.ci95_low, stats.ci95_high);

            float variance_pct = (stats.stddev / stats.mean) * 100.0f;
            if (variance_pct > 5.0f) {
                printf("    ⚠ Variance %.1f%% > 5%% — consider increasing trials\n",
                       variance_pct);
            }
        }
    }

    // ────────────────────────────────────────────────────────────────
    // 2. Softmax
    // ────────────────────────────────────────────────────────────────
    {
        print_header("CUDA Softmax (Row-wise)");

        struct SoftmaxConfig { int batch; int seq_len; };
        const std::vector<SoftmaxConfig> configs{{64, 16384}, {64, 65536}, {64, 262144}};

        for (auto& cfg : configs) {
            printf("  Size: %d × %d\n", cfg.batch, cfg.seq_len);

            auto h_input = random_f32((size_t)cfg.batch * cfg.seq_len, 42);
            std::vector<float> h_output(h_input.size());

            // Warm-up
            for (int i = 0; i < warmup_runs(); ++i) {
                run_cuda_softmax(h_input.data(), h_output.data(),
                                  cfg.batch, cfg.seq_len);
            }

            // Benchmark
            std::vector<float> kernel_times;
            for (int t = 0; t < num_trials(); ++t) {
                enforce_cooldown();
                auto r = run_cuda_softmax(h_input.data(), h_output.data(),
                                           cfg.batch, cfg.seq_len);
                kernel_times.push_back(r.kernel_time_ms);

                std::string size_str = std::to_string(cfg.batch) + "x" +
                                       std::to_string(cfg.seq_len);
                csv.write("cuda_softmax", size_str,
                          t, r.kernel_time_ms, r.h2d_time_ms, r.d2h_time_ms,
                          0, r.bandwidth_gbps);
            }

            auto stats = compute_stats(kernel_times);
            printf("    Kernel: %.3f ± %.3f ms  [CI: %.3f – %.3f]\n",
                   stats.mean, stats.stddev, stats.ci95_low, stats.ci95_high);

            // Verify numerical correctness (spot check)
            float row_sum = 0;
            for (int j = 0; j < cfg.seq_len; ++j) {
                row_sum += h_output[j];
            }
            printf("    Verification: row[0] sum = %.6f (expected ≈ 1.0)\n", row_sum);
        }
    }

    // ────────────────────────────────────────────────────────────────
    // 3. Vector Reduction
    // ────────────────────────────────────────────────────────────────
    {
        print_header("CUDA Vector Reduction");

        const std::vector<int> reduction_sizes{1000000, 10000000, 100000000};

        for (int n : reduction_sizes) {
            printf("  Size: %d elements\n", n);

            auto h_input = random_f32(n, 42);

            // Warm-up
            for (int i = 0; i < warmup_runs(); ++i) {
                run_cuda_reduction(h_input.data(), n);
            }

            // Benchmark
            std::vector<float> kernel_times;
            for (int t = 0; t < num_trials(); ++t) {
                enforce_cooldown();
                auto r = run_cuda_reduction(h_input.data(), n);
                kernel_times.push_back(r.kernel_time_ms);
                csv.write("cuda_reduction", std::to_string(n),
                          t, r.kernel_time_ms, r.h2d_time_ms, r.d2h_time_ms,
                          r.gflops, 0);
            }

            auto stats = compute_stats(kernel_times);
            printf("    Kernel: %.3f ± %.3f ms  [CI: %.3f – %.3f]\n",
                   stats.mean, stats.stddev, stats.ci95_low, stats.ci95_high);
        }
    }

    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  ✓ All GPU benchmarks complete!\n");
    printf("  Results: %s/gpu_cpp.csv\n", results_dir.c_str());
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    return 0;
}
