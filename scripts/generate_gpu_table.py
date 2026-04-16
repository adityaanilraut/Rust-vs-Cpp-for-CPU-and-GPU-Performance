#!/usr/bin/env python3
"""
GPU Benchmark Statistical Analysis
====================================
Reads gpu_cpp.csv and gpu_rust.csv from results/raw/, computes per-benchmark
statistics (mean, std, 95% CI, Welch's t-test, Cohen's d, CV%), and outputs
a LaTeX table and a summary CSV.

Outputs:
  results/raw/gpu_analysis_table.tex
  results/raw/gpu_analysis_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def analyze_gpu(results_dir: Path, output_dir: Path):
    cpp_df = pd.read_csv(results_dir / "gpu_cpp.csv")
    rust_df = pd.read_csv(results_dir / "gpu_rust.csv")

    df = pd.concat([cpp_df, rust_df], ignore_index=True)

    # Normalise language column  (cpp_cuda → cpp, rust_cuda → rust)
    df["lang"] = df["language"].str.replace(r"_cuda$", "", regex=True)

    benchmarks = sorted(df["benchmark"].unique())
    rows = []

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  GPU Statistical Analysis")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    for bench in benchmarks:
        bdf = df[df["benchmark"] == bench]
        sizes = bdf["input_size"].unique()
        print(f"\n  ┌─ {bench}")

        for size in sizes:
            sdf = bdf[bdf["input_size"] == size]
            cpp_vals = sdf[sdf["lang"] == "cpp"]["kernel_time_ms"].values
            rust_vals = sdf[sdf["lang"] == "rust"]["kernel_time_ms"].values

            if len(cpp_vals) < 2 or len(rust_vals) < 2:
                continue

            cpp_mean, cpp_std = np.mean(cpp_vals), np.std(cpp_vals, ddof=1)
            rust_mean, rust_std = np.mean(rust_vals), np.std(rust_vals, ddof=1)

            cpp_ci = stats.t.interval(0.95, len(cpp_vals) - 1,
                                      loc=cpp_mean, scale=stats.sem(cpp_vals))
            rust_ci = stats.t.interval(0.95, len(rust_vals) - 1,
                                       loc=rust_mean, scale=stats.sem(rust_vals))

            t_stat, p_value = stats.ttest_ind(cpp_vals, rust_vals, equal_var=False)

            pooled_std = np.sqrt((cpp_std ** 2 + rust_std ** 2) / 2)
            cohens_d = (rust_mean - cpp_mean) / pooled_std if pooled_std > 0 else 0

            overhead_pct = ((rust_mean - cpp_mean) / cpp_mean) * 100 if cpp_mean > 0 else 0

            cpp_cv = (cpp_std / cpp_mean) * 100 if cpp_mean > 0 else 0
            rust_cv = (rust_std / rust_mean) * 100 if rust_mean > 0 else 0
            high_variance = cpp_cv > 5 or rust_cv > 5

            sig = ("***" if p_value < 0.001 else
                   "**" if p_value < 0.01 else
                   "*" if p_value < 0.05 else "ns")
            flag = " ⚠ HIGH VARIANCE" if high_variance else ""
            sign = "+" if overhead_pct > 0 else ""
            print(f"  │  Size {str(size):>12s}:  C++ {cpp_mean:>8.4f}ms  "
                  f"Rust {rust_mean:>8.4f}ms  "
                  f"({sign}{overhead_pct:.1f}%)  p={p_value:.4f} {sig}{flag}")

            rows.append({
                "benchmark": bench,
                "input_size": size,
                "cpp_mean_ms": round(cpp_mean, 4),
                "cpp_std_ms": round(cpp_std, 4),
                "cpp_ci_low": round(cpp_ci[0], 4),
                "cpp_ci_high": round(cpp_ci[1], 4),
                "rust_mean_ms": round(rust_mean, 4),
                "rust_std_ms": round(rust_std, 4),
                "rust_ci_low": round(rust_ci[0], 4),
                "rust_ci_high": round(rust_ci[1], 4),
                "overhead_pct": round(overhead_pct, 2),
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_value, 6),
                "cohens_d": round(cohens_d, 4),
                "cpp_cv_pct": round(cpp_cv, 2),
                "rust_cv_pct": round(rust_cv, 2),
                "high_variance": high_variance,
            })
        print("  └─")

    summary_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "gpu_analysis_summary.csv", index=False)
    _write_latex_table(summary_df, output_dir / "gpu_analysis_table.tex")

    print(f"\n  ✓ GPU summary → {output_dir / 'gpu_analysis_summary.csv'}")
    print(f"  ✓ GPU LaTeX table → {output_dir / 'gpu_analysis_table.tex'}")
    return summary_df


def _write_latex_table(df: pd.DataFrame, path: Path):
    with open(path, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\caption{GPU Kernel Performance: C++ vs.\\ Rust (kernel time only, host transfer excluded)}\n")
        f.write("\\label{tab:gpu_results}\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{llrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Benchmark & Size & C++ (ms) & Rust (ms) & Overhead & "
                "p-value & Cohen's $d$ & CV\\% (C++) & CV\\% (Rust) \\\\\n")
        f.write("\\midrule\n")

        for _, row in df.iterrows():
            sig = ("$^{***}$" if row["p_value"] < 0.001 else
                   "$^{**}$" if row["p_value"] < 0.01 else
                   "$^{*}$" if row["p_value"] < 0.05 else "")
            overhead_str = f"{row['overhead_pct']:+.1f}\\%"
            bench_name = row["benchmark"].replace("_", "\\_")
            size_str = str(row["input_size"])
            f.write(
                f"{bench_name} & {size_str} & "
                f"{row['cpp_mean_ms']:.4f} $\\pm$ {row['cpp_std_ms']:.4f} & "
                f"{row['rust_mean_ms']:.4f} $\\pm$ {row['rust_std_ms']:.4f} & "
                f"{overhead_str} & "
                f"{row['p_value']:.4f}{sig} & "
                f"{row['cohens_d']:.2f} & "
                f"{row['cpp_cv_pct']:.1f} & "
                f"{row['rust_cv_pct']:.1f} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\smallskip\n")
        f.write("{\\small $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$. "
                "CV\\% = coefficient of variation; overhead positive means Rust is slower.}\n")
        f.write("\\end{table}\n")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results" / "raw"
    analyze_gpu(results_dir, results_dir)
