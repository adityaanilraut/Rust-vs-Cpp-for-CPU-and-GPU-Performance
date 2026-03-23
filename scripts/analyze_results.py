#!/usr/bin/env python3
"""
Statistical Analysis Script
============================
Processes raw benchmark CSV data and computes:
- Mean, standard deviation, 95% CI (t-distribution)
- Welch's t-test (Rust vs C++)
- Cohen's d effect size
- Variance check (>5% → flag for 50 trials)

Outputs: summary CSV, LaTeX tables, console report.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all CSV result files and merge."""
    frames = []
    for csv_file in results_dir.glob("*.csv"):
        if csv_file.name.startswith("analysis_"):
            continue
        try:
            df = pd.read_csv(csv_file)
            frames.append(df)
        except Exception as e:
            print(f"  ⚠ Skipping {csv_file.name}: {e}")

    if not frames:
        print("ERROR: No CSV files found in", results_dir)
        print("       Run benchmarks first, or use --sample for demo data.")
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


def generate_sample_data() -> pd.DataFrame:
    """Generate realistic sample data for testing the analysis pipeline."""
    np.random.seed(42)
    records = []

    benchmarks = {
        "quicksort": {"sizes": ["1000000", "10000000", "100000000"],
                       "cpp_base": [45, 520, 6200], "rust_ratio": [1.03, 1.05, 1.02]},
        "mergesort": {"sizes": ["1000000", "10000000", "100000000"],
                       "cpp_base": [55, 620, 7100], "rust_ratio": [1.06, 1.08, 1.04]},
        "matmul":    {"sizes": ["512", "1024", "2048"],
                       "cpp_base": [150, 1800, 18000], "rust_ratio": [1.10, 1.12, 1.09]},
        "parallel_reduction": {"sizes": ["1000000", "10000000", "100000000"],
                                "cpp_base": [2, 18, 180], "rust_ratio": [0.98, 1.01, 1.03]},
    }

    gpu_benchmarks = {
        "cuda_matmul_tiled": {"sizes": ["1024", "2048", "4096"],
                               "cpp_base": [0.8, 5.2, 38.0], "rust_ratio": [1.02, 1.03, 1.01]},
        "cuda_softmax": {"sizes": ["64x16384", "64x65536", "64x262144"],
                          "cpp_base": [0.12, 0.45, 1.8], "rust_ratio": [1.01, 1.02, 1.01]},
        "cuda_reduction": {"sizes": ["1000000", "10000000", "100000000"],
                            "cpp_base": [0.08, 0.5, 4.2], "rust_ratio": [1.015, 1.02, 1.01]},
    }

    for bench, cfg in {**benchmarks, **gpu_benchmarks}.items():
        for i, size in enumerate(cfg["sizes"]):
            cpp_base = cfg["cpp_base"][i]
            rust_base = cpp_base * cfg["rust_ratio"][i]
            cpp_std = cpp_base * 0.02
            rust_std = rust_base * 0.025

            for trial in range(30):
                records.append({
                    "benchmark": bench, "language": "cpp",
                    "input_size": size, "trial": trial,
                    "kernel_time_ms": max(0.01, np.random.normal(cpp_base, cpp_std)),
                    "h2d_time_ms": 0, "d2h_time_ms": 0,
                    "gflops": 0, "bandwidth_gbps": 0
                })
                records.append({
                    "benchmark": bench, "language": "rust",
                    "input_size": size, "trial": trial,
                    "kernel_time_ms": max(0.01, np.random.normal(rust_base, rust_std)),
                    "h2d_time_ms": 0, "d2h_time_ms": 0,
                    "gflops": 0, "bandwidth_gbps": 0
                })

    return pd.DataFrame(records)


def analyze(df: pd.DataFrame, output_dir: Path):
    """Run full statistical analysis."""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Statistical Analysis Report")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    results = []

    # Identify unique (benchmark, language) key from the 'language' column
    # or infer from naming convention
    if 'language' not in df.columns:
        df['language'] = df['benchmark'].apply(
            lambda x: 'rust' if 'rust' in x.lower() else 'cpp'
        )

    benchmarks = df['benchmark'].unique()
    for bench in sorted(benchmarks):
        bench_df = df[df['benchmark'] == bench]
        sizes = bench_df['input_size'].unique()

        print(f"  ┌─ {bench}")
        for size in sorted(sizes, key=str):
            size_df = bench_df[bench_df['input_size'] == size]
            size_str = str(size)

            cpp_data = size_df[size_df['language'].str.contains('cpp', case=False)]['kernel_time_ms'].values
            rust_data = size_df[size_df['language'].str.contains('rust', case=False)]['kernel_time_ms'].values

            if len(cpp_data) == 0 or len(rust_data) == 0:
                continue

            # Basic stats
            cpp_mean, cpp_std = np.mean(cpp_data), np.std(cpp_data, ddof=1)
            rust_mean, rust_std = np.mean(rust_data), np.std(rust_data, ddof=1)

            # 95% CI (t-distribution)
            cpp_ci = stats.t.interval(0.95, len(cpp_data)-1, loc=cpp_mean,
                                       scale=stats.sem(cpp_data))
            rust_ci = stats.t.interval(0.95, len(rust_data)-1, loc=rust_mean,
                                        scale=stats.sem(rust_data))

            # Welch's t-test
            t_stat, p_value = stats.ttest_ind(cpp_data, rust_data, equal_var=False)

            # Cohen's d effect size
            pooled_std = np.sqrt((cpp_std**2 + rust_std**2) / 2)
            cohens_d = (rust_mean - cpp_mean) / pooled_std if pooled_std > 0 else 0

            # Overhead percentage
            overhead_pct = ((rust_mean - cpp_mean) / cpp_mean) * 100

            # Variance check
            cpp_cv = (cpp_std / cpp_mean) * 100 if cpp_mean > 0 else 0
            rust_cv = (rust_std / rust_mean) * 100 if rust_mean > 0 else 0
            high_variance = cpp_cv > 5 or rust_cv > 5

            result = {
                "benchmark": bench, "input_size": size,
                "cpp_mean_ms": round(cpp_mean, 4), "cpp_std_ms": round(cpp_std, 4),
                "cpp_ci_low": round(cpp_ci[0], 4), "cpp_ci_high": round(cpp_ci[1], 4),
                "rust_mean_ms": round(rust_mean, 4), "rust_std_ms": round(rust_std, 4),
                "rust_ci_low": round(rust_ci[0], 4), "rust_ci_high": round(rust_ci[1], 4),
                "overhead_pct": round(overhead_pct, 2),
                "t_statistic": round(t_stat, 4), "p_value": round(p_value, 6),
                "cohens_d": round(cohens_d, 4),
                "cpp_cv_pct": round(cpp_cv, 2), "rust_cv_pct": round(rust_cv, 2),
                "high_variance": high_variance,
            }
            results.append(result)

            # Console output
            sign = "+" if overhead_pct > 0 else ""
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            flag = " ⚠ HIGH VARIANCE" if high_variance else ""
            print(f"  │  Size {size_str:>12s}:  C++ {cpp_mean:>10.3f}ms  "
                  f"Rust {rust_mean:>10.3f}ms  "
                  f"({sign}{overhead_pct:.1f}%)  p={p_value:.4f} {sig}{flag}")

        print(f"  └─")

    # Save summary
    summary_df = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_dir / "analysis_summary.csv", index=False)

    # Generate LaTeX table
    generate_latex_table(summary_df, output_dir / "analysis_table.tex")

    print(f"\n  ✓ Summary saved to {output_dir / 'analysis_summary.csv'}")
    print(f"  ✓ LaTeX table saved to {output_dir / 'analysis_table.tex'}")

    # Hypothesis evaluation
    print("\n━━ Hypothesis Evaluation ━━━━━━━━━━━━━━━━━━━━━━━━━━")
    cpu_results = summary_df[~summary_df['benchmark'].str.contains('cuda')]
    gpu_results = summary_df[summary_df['benchmark'].str.contains('cuda')]

    if len(cpu_results) > 0:
        avg_cpu_overhead = cpu_results['overhead_pct'].mean()
        h1 = "SUPPORTED" if -15 <= avg_cpu_overhead <= 15 else "REFUTED"
        print(f"  H1 (Rust within 5-15% on CPU): {h1}")
        print(f"      Average overhead: {avg_cpu_overhead:+.1f}%")

    if len(gpu_results) > 0:
        avg_gpu_overhead = gpu_results['overhead_pct'].mean()
        h2 = "SUPPORTED" if avg_gpu_overhead < 5 else "REFUTED"
        print(f"  H2 (GPU near-parity): {h2}")
        print(f"      Average overhead: {avg_gpu_overhead:+.1f}%")

    return summary_df


def generate_latex_table(df: pd.DataFrame, output_path: Path):
    """Generate IEEE-formatted LaTeX table."""
    with open(output_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\caption{Performance Comparison: C++ vs. Rust}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{llrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Benchmark & Size & C++ (ms) & Rust (ms) & Overhead & p-value & Cohen's d \\\\\n")
        f.write("\\midrule\n")

        for _, row in df.iterrows():
            sig = "$^{***}$" if row['p_value'] < 0.001 else \
                  "$^{**}$" if row['p_value'] < 0.01 else \
                  "$^{*}$" if row['p_value'] < 0.05 else ""
            f.write(f"{row['benchmark']} & {row['input_size']} & "
                    f"{row['cpp_mean_ms']:.2f} $\\pm$ {row['cpp_std_ms']:.2f} & "
                    f"{row['rust_mean_ms']:.2f} $\\pm$ {row['rust_std_ms']:.2f} & "
                    f"{row['overhead_pct']:+.1f}\\% & "
                    f"{row['p_value']:.4f}{sig} & "
                    f"{row['cohens_d']:.2f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark statistical analysis")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing raw CSV results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for analysis files")
    parser.add_argument("--sample", action="store_true",
                        help="Use generated sample data for testing")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else project_root / "results" / "raw"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "results" / "raw"

    if args.sample:
        print("  Using generated sample data...")
        df = generate_sample_data()
    else:
        df = load_results(results_dir)

    analyze(df, output_dir)


if __name__ == "__main__":
    main()
