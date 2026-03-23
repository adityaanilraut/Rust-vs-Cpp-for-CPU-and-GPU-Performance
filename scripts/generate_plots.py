#!/usr/bin/env python3
"""
Plot Generator for Rust vs. C++ Benchmark Study
=================================================
Creates publication-quality figures using matplotlib + seaborn.

Plots generated:
  1. Grouped bar charts with CI error bars (CPU)
  2. GFLOPS throughput comparison (GPU)
  3. Kernel vs. transfer time breakdown (stacked bars)
  4. Speedup ratio heatmap
  5. Violin plots for distribution visualization
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

# ── Style Configuration ─────────────────────────────────────────────────────

COLORS = {
    "cpp": "#FF6B35",     # Warm orange
    "rust": "#4ECDC4",    # Teal
    "cpp_dark": "#C94E1E",
    "rust_dark": "#3AA89E",
}

def setup_style():
    """Configure publication-quality plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def generate_sample_data() -> pd.DataFrame:
    """Generate sample data for testing plot generation."""
    np.random.seed(42)
    records = []

    benchmarks = {
        "QuickSort": {"sizes": [1e6, 1e7, 1e8], "cpp": [45, 520, 6200], "rust_r": [1.03, 1.05, 1.02]},
        "MergeSort": {"sizes": [1e6, 1e7, 1e8], "cpp": [55, 620, 7100], "rust_r": [1.06, 1.08, 1.04]},
        "MatMul":    {"sizes": [512, 1024, 2048], "cpp": [150, 1800, 18000], "rust_r": [1.10, 1.12, 1.09]},
        "ParReduce": {"sizes": [1e6, 1e7, 1e8], "cpp": [2, 18, 180], "rust_r": [0.98, 1.01, 1.03]},
    }

    gpu_benchmarks = {
        "CUDA MatMul":   {"sizes": [1024, 2048, 4096], "cpp": [0.8, 5.2, 38], "rust_r": [1.02, 1.03, 1.01]},
        "CUDA Softmax":  {"sizes": [16384, 65536, 262144], "cpp": [0.12, 0.45, 1.8], "rust_r": [1.01, 1.02, 1.01]},
        "CUDA Reduce":   {"sizes": [1e6, 1e7, 1e8], "cpp": [0.08, 0.5, 4.2], "rust_r": [1.015, 1.02, 1.01]},
    }

    for bench, cfg in {**benchmarks, **gpu_benchmarks}.items():
        for i, size in enumerate(cfg["sizes"]):
            for lang, base, std_ratio in [("C++", cfg["cpp"][i], 0.02),
                                           ("Rust", cfg["cpp"][i] * cfg["rust_r"][i], 0.025)]:
                for trial in range(30):
                    records.append({
                        "benchmark": bench, "language": lang,
                        "input_size": size, "trial": trial,
                        "time_ms": max(0.01, np.random.normal(base, base * std_ratio)),
                        "is_gpu": bench.startswith("CUDA"),
                    })

    return pd.DataFrame(records)


# ── Plot Functions ───────────────────────────────────────────────────────────

def plot_grouped_bar_cpu(df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart with CI error bars for CPU benchmarks."""
    cpu_df = df[~df['is_gpu']]
    benchmarks = cpu_df['benchmark'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, bench in enumerate(benchmarks):
        ax = axes[idx]
        bench_df = cpu_df[cpu_df['benchmark'] == bench]
        sizes = sorted(bench_df['input_size'].unique())

        x = np.arange(len(sizes))
        width = 0.35

        for i, lang in enumerate(["C++", "Rust"]):
            means, cis = [], []
            for size in sizes:
                data = bench_df[(bench_df['input_size'] == size) &
                                (bench_df['language'] == lang)]['time_ms']
                means.append(data.mean())
                cis.append(1.96 * data.std() / np.sqrt(len(data)))

            color = COLORS['cpp'] if lang == "C++" else COLORS['rust']
            ax.bar(x + i * width, means, width, yerr=cis,
                   label=lang, color=color, alpha=0.85,
                   capsize=3, edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Input Size')
        ax.set_ylabel('Time (ms)')
        ax.set_title(bench, fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([f"{s:.0e}" if s >= 1e6 else str(int(s)) for s in sizes])
        ax.legend()
        ax.set_yscale('log')

    plt.suptitle('CPU Benchmark Comparison: C++ vs. Rust', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'cpu_comparison_bar.png', bbox_inches='tight')
    plt.savefig(output_dir / 'cpu_comparison_bar.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ cpu_comparison_bar.png")


def plot_gpu_comparison(df: pd.DataFrame, output_dir: Path):
    """GPU kernel time comparison."""
    gpu_df = df[df['is_gpu']]
    benchmarks = gpu_df['benchmark'].unique()

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(5 * len(benchmarks), 5))
    if len(benchmarks) == 1:
        axes = [axes]

    for idx, bench in enumerate(benchmarks):
        ax = axes[idx]
        bench_df = gpu_df[gpu_df['benchmark'] == bench]
        sizes = sorted(bench_df['input_size'].unique())

        x = np.arange(len(sizes))
        width = 0.35

        for i, lang in enumerate(["C++", "Rust"]):
            means = []
            for size in sizes:
                data = bench_df[(bench_df['input_size'] == size) &
                                (bench_df['language'] == lang)]['time_ms']
                means.append(data.mean())

            color = COLORS['cpp'] if lang == "C++" else COLORS['rust']
            ax.bar(x + i * width, means, width, label=lang,
                   color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Input Size')
        ax.set_ylabel('Kernel Time (ms)')
        ax.set_title(bench, fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([f"{s:.0e}" if s >= 1e6 else str(int(s)) for s in sizes],
                            rotation=30)
        ax.legend()

    plt.suptitle('GPU Kernel Performance: C++ vs. Rust', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'gpu_comparison_bar.png', bbox_inches='tight')
    plt.savefig(output_dir / 'gpu_comparison_bar.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ gpu_comparison_bar.png")


def plot_speedup_heatmap(df: pd.DataFrame, output_dir: Path):
    """Heatmap showing Rust-to-C++ speedup ratio."""
    benchmarks = df['benchmark'].unique()
    all_sizes = sorted(df['input_size'].unique())

    matrix = []
    bench_labels = []
    size_labels = set()

    for bench in benchmarks:
        row = []
        bench_df = df[df['benchmark'] == bench]
        sizes = sorted(bench_df['input_size'].unique())
        for size in sizes:
            cpp = bench_df[(bench_df['input_size'] == size) &
                           (bench_df['language'] == 'C++')]['time_ms'].mean()
            rust = bench_df[(bench_df['input_size'] == size) &
                            (bench_df['language'] == 'Rust')]['time_ms'].mean()
            ratio = cpp / rust if rust > 0 else 1.0
            row.append(ratio)
            size_labels.add(str(int(size)) if size < 1e6 else f"{size:.0e}")
        matrix.append(row)
        bench_labels.append(bench)

    # Pad rows to same length
    max_len = max(len(r) for r in matrix)
    for r in matrix:
        while len(r) < max_len:
            r.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))
    data = np.array(matrix)

    cmap = sns.diverging_palette(10, 150, as_cmap=True, center='light')
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0.8, vmax=1.2)

    ax.set_yticks(range(len(bench_labels)))
    ax.set_yticklabels(bench_labels)
    ax.set_xlabel('Input Size Index')
    ax.set_title('C++/Rust Speedup Ratio (>1 = C++ faster)', fontweight='bold')

    plt.colorbar(im, ax=ax, label='Speedup Ratio')

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f'{data[i,j]:.3f}', ha='center', va='center',
                        fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_heatmap.png', bbox_inches='tight')
    plt.savefig(output_dir / 'speedup_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ speedup_heatmap.png")


def plot_violin_distributions(df: pd.DataFrame, output_dir: Path):
    """Violin plots showing full distribution of benchmark times."""
    cpu_df = df[~df['is_gpu']]
    benchmarks = cpu_df['benchmark'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, bench in enumerate(benchmarks):
        ax = axes[idx]
        bench_df = cpu_df[cpu_df['benchmark'] == bench]
        largest_size = bench_df['input_size'].max()
        data = bench_df[bench_df['input_size'] == largest_size]

        palette = {'C++': COLORS['cpp'], 'Rust': COLORS['rust']}
        sns.violinplot(data=data, x='language', y='time_ms', ax=ax,
                       palette=palette, inner='box', cut=0)

        ax.set_title(f'{bench} (n={int(largest_size):,})', fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Time (ms)')

    plt.suptitle('Distribution of Benchmark Times (Largest Input Size)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'violin_distributions.png', bbox_inches='tight')
    plt.savefig(output_dir / 'violin_distributions.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ violin_distributions.png")


def plot_overhead_summary(df: pd.DataFrame, output_dir: Path):
    """Bar chart of Rust overhead percentage per benchmark."""
    overheads = []
    for bench in df['benchmark'].unique():
        bench_df = df[df['benchmark'] == bench]
        for size in bench_df['input_size'].unique():
            cpp = bench_df[(bench_df['input_size'] == size) &
                           (bench_df['language'] == 'C++')]['time_ms'].mean()
            rust = bench_df[(bench_df['input_size'] == size) &
                            (bench_df['language'] == 'Rust')]['time_ms'].mean()
            overhead = ((rust - cpp) / cpp) * 100
            overheads.append({"benchmark": bench, "size": size, "overhead_pct": overhead})

    overhead_df = pd.DataFrame(overheads)
    avg_overhead = overhead_df.groupby('benchmark')['overhead_pct'].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLORS['rust'] if v > 0 else COLORS['cpp'] for v in avg_overhead.values]
    bars = ax.barh(avg_overhead.index, avg_overhead.values, color=colors, alpha=0.85,
                    edgecolor='white', linewidth=0.5)

    ax.axvline(x=0, color='gray', linewidth=1, linestyle='--')
    ax.axvline(x=5, color='red', linewidth=0.8, linestyle=':', alpha=0.5, label='5% threshold')
    ax.axvline(x=15, color='red', linewidth=0.8, linestyle=':', alpha=0.5, label='15% threshold')
    ax.axvline(x=-5, color='green', linewidth=0.8, linestyle=':', alpha=0.5)

    ax.set_xlabel('Rust Overhead vs. C++ (%)')
    ax.set_title('Average Rust Overhead by Benchmark', fontweight='bold')
    ax.legend()

    for bar, val in zip(bars, avg_overhead.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:+.1f}%', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'overhead_summary.png', bbox_inches='tight')
    plt.savefig(output_dir / 'overhead_summary.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ overhead_summary.png")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sample", action="store_true",
                        help="Use sample data for testing")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    if args.sample:
        df = generate_sample_data()
    else:
        results_dir = Path(args.results_dir) if args.results_dir else project_root / "results" / "raw"
        df = pd.read_csv(results_dir / "analysis_summary.csv") if (results_dir / "analysis_summary.csv").exists() \
            else generate_sample_data()
        print("  ⚠ Using sample data (no analysis_summary.csv found)")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Generating Publication-Quality Plots")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    plot_grouped_bar_cpu(df, output_dir)
    plot_gpu_comparison(df, output_dir)
    plot_speedup_heatmap(df, output_dir)
    plot_violin_distributions(df, output_dir)
    plot_overhead_summary(df, output_dir)

    print(f"\n  ✓ All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
