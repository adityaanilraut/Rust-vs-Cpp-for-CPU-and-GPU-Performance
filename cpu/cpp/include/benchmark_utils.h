#pragma once
/**
 * benchmark_utils.h — Shared utilities for benchmark suite
 * 
 * Provides dataset loading from binary files, CSV output helpers,
 * and timing utilities for the Rust vs. C++ benchmark study.
 */

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace bench_utils {

inline double cpu_min_time_seconds() {
    return 2.0;
}

inline int cpu_repetitions(int reps) {
    return reps;
}

// ── Dataset Loading ─────────────────────────────────────────────────────────

/**
 * Load a binary file containing an array of type T.
 * Files are raw binary (no header) — element count inferred from file size.
 */
template <typename T>
std::vector<T> load_binary(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open dataset: " << filepath << std::endl;
        std::cerr << "       Run 'python3 scripts/generate_datasets.py' first." << std::endl;
        std::exit(1);
    }
    
    auto file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t count = static_cast<size_t>(file_size) / sizeof(T);
    std::vector<T> data(count);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    
    return data;
}

/**
 * Load a binary file as a flattened NxN matrix of type T.
 */
template <typename T>
std::vector<T> load_matrix(const std::string& filepath, size_t n) {
    auto data = load_binary<T>(filepath);
    if (data.size() != n * n) {
        std::cerr << "ERROR: Matrix file size mismatch. Expected "
                  << n * n << " elements, got " << data.size() << std::endl;
        std::exit(1);
    }
    return data;
}

/**
 * Generate a random array in-memory (fallback if dataset files not available).
 */
template <typename T>
std::vector<T> generate_random_array(size_t size, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::vector<T> data(size);
    
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(0, std::numeric_limits<T>::max());
        std::generate(data.begin(), data.end(), [&]() { return dist(rng); });
    } else {
        std::normal_distribution<T> dist(0.0, 1.0);
        std::generate(data.begin(), data.end(), [&]() { return dist(rng); });
    }
    
    return data;
}

// ── Result Recording ────────────────────────────────────────────────────────

struct BenchmarkResult {
    std::string benchmark_name;
    std::string language;
    std::string input_size;
    double time_ms;
    double stddev_ms;
    int num_trials;
};

/**
 * Append a benchmark result to a CSV file.
 */
inline void append_csv(const std::string& filepath, const BenchmarkResult& result) {
    bool file_exists = std::ifstream(filepath).good();
    
    std::ofstream file(filepath, std::ios::app);
    if (!file_exists) {
        file << "benchmark,language,input_size,time_ms,stddev_ms,num_trials\n";
    }
    file << result.benchmark_name << ","
         << result.language << ","
         << result.input_size << ","
         << result.time_ms << ","
         << result.stddev_ms << ","
         << result.num_trials << "\n";
}

// ── Path Helpers ────────────────────────────────────────────────────────────

inline std::string dataset_path(const std::string& filename) {
    #ifdef DATASET_DIR
        return std::string(DATASET_DIR) + "/" + filename;
    #else
        return "../../datasets/" + filename;
    #endif
}

inline std::string results_path(const std::string& filename) {
    return "../../results/raw/" + filename;
}

} // namespace bench_utils
