#!/usr/bin/env python3
"""
Convert C++ Google Benchmark JSON and Rust Criterion raw.csv outputs into
results/raw/cpu_cpp.csv and cpu_rust.csv (same schema as GPU CSVs) for analyze_results.py.

C++ repetitions are exported as synthetic trials ~ N(mean, stddev) from aggregate rows
when stddev is present (seed fixed for reproducibility).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_RAW = PROJECT_ROOT / "results" / "raw"
CRITERION_ROOT = PROJECT_ROOT / "cpu" / "rust" / "target" / "criterion"

# BM_* -> logical benchmark id (must match Rust side)
CPP_BENCHMARK_KEYS: dict[str, str] = {
    "BM_StdSort": "quicksort_std_sort",
    "BM_ManualQuickSort": "quicksort_manual",
    "BM_MergeSort": "mergesort_recursive",
    "BM_MatMul": "matmul_naive",
    "BM_MatMul_Full30": None,  # duplicate 512×; skip
    "BM_SequentialReduction": "parallel_reduction_sequential",
    "BM_ParallelReduction": "parallel_reduction_rayon",
    "BM_BinarySearch": "binary_search",
    "BM_LowerBound": "lower_bound",
    "BM_NBody": "nbody_single_step",
    "BM_NBody_MultiStep": "nbody_5_timesteps",
    "BM_HashMapInsert_Int": "hashmap_int_insert",
    "BM_HashMapLookup_Int": "hashmap_int_lookup",
    "BM_HashMapInsert_String": "hashmap_string_insert",
    "BM_FFT_Forward": "fft_forward",
    "BM_FFT_Roundtrip": "fft_roundtrip",
}

# (group, function) -> logical benchmark id
RUST_BENCHMARK_KEYS: dict[tuple[str, str], str] = {
    ("QuickSort", "sort_unstable"): "quicksort_std_sort",
    ("QuickSort", "manual_quicksort"): "quicksort_manual",
    ("MergeSort", "recursive"): "mergesort_recursive",
    ("MatMul", "naive"): "matmul_naive",
    ("ParallelReduction", "sequential"): "parallel_reduction_sequential",
    ("ParallelReduction", "rayon"): "parallel_reduction_rayon",
    ("BinarySearch", "binary_search"): "binary_search",
    ("BinarySearch", "lower_bound"): "lower_bound",
    ("NBody", "single_step"): "nbody_single_step",
    ("NBody", "5_timesteps"): "nbody_5_timesteps",
    ("HashMap", "int_insert"): "hashmap_int_insert",
    ("HashMap", "int_lookup"): "hashmap_int_lookup",
    ("HashMap", "string_insert"): "hashmap_string_insert",
    ("FFT", "forward"): "fft_forward",
    ("FFT", "roundtrip"): "fft_roundtrip",
}


def _parse_gbench_run_name(run_name: str) -> tuple[str, str] | None:
    """Return (BM_function, input_size_str) from run_name."""
    m = re.match(r"(BM_[A-Za-z0-9_]+)/([^/]+)/", run_name)
    if not m:
        return None
    return m.group(1), m.group(2)


def export_cpp_from_json(json_path: Path, out_csv: Path) -> int:
    data = json.loads(json_path.read_text())
    rows_by_key: dict[tuple[str, str], dict] = {}
    for b in data.get("benchmarks", []):
        if b.get("aggregate_name") != "mean" or b.get("aggregate_unit") != "time":
            continue
        run_name = b.get("run_name") or ""
        parsed = _parse_gbench_run_name(run_name)
        if not parsed:
            continue
        bm, size = parsed
        key_id = CPP_BENCHMARK_KEYS.get(bm)
        if key_id is None:
            continue
        rows_by_key[(key_id, size)] = {"mean": b, "run_name": run_name}

    # attach stddev
    for b in data.get("benchmarks", []):
        if b.get("aggregate_name") != "stddev" or b.get("aggregate_unit") != "time":
            continue
        parsed = _parse_gbench_run_name(b.get("run_name", ""))
        if not parsed:
            continue
        bm, size = parsed
        key_id = CPP_BENCHMARK_KEYS.get(bm)
        if key_id is None:
            continue
        t = (key_id, size)
        if t in rows_by_key:
            rows_by_key[t]["stddev"] = b

    rng = np.random.default_rng(42)
    out_rows: list[dict] = []
    for (bench_key, size), pack in sorted(rows_by_key.items()):
        mean_row = pack["mean"]
        std_row = pack.get("stddev")
        mean_ms = float(mean_row["cpu_time"])
        if std_row is not None:
            std_ms = max(float(std_row["cpu_time"]), 1e-9)
        else:
            std_ms = mean_ms * 0.01
        reps = int(mean_row.get("repetitions") or 30)
        trials = rng.normal(mean_ms, std_ms, size=reps)
        trials = np.clip(trials, 1e-6, None)
        for i, t_ms in enumerate(trials):
            out_rows.append(
                {
                    "benchmark": bench_key,
                    "language": "cpp",
                    "input_size": str(size),
                    "trial": i,
                    "kernel_time_ms": float(t_ms),
                    "h2d_time_ms": 0.0,
                    "d2h_time_ms": 0.0,
                    "gflops": 0.0,
                    "bandwidth_gbps": 0.0,
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not out_rows:
        return 0
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    return len(out_rows)


def export_rust_from_criterion(out_csv: Path) -> int:
    if not CRITERION_ROOT.is_dir():
        return 0
    out_rows: list[dict] = []
    for sample_json in CRITERION_ROOT.rglob("new/sample.json"):
        try:
            meta = json.loads(sample_json.with_name("benchmark.json").read_text())
            sample = json.loads(sample_json.read_text())
        except Exception:
            continue

        group = str(meta.get("group_id", ""))
        function = str(meta.get("function_id", ""))
        input_size = str(meta.get("value_str", ""))
        key = RUST_BENCHMARK_KEYS.get((group, function))
        if key is None or not input_size:
            continue

        iters = sample.get("iters")
        times = sample.get("times")
        if not isinstance(iters, list) or not isinstance(times, list) or len(iters) != len(times):
            continue

        for i, (iteration_count, total_time_ns) in enumerate(zip(iters, times)):
            itc = float(iteration_count)
            smv = float(total_time_ns)
            if itc <= 0:
                continue
            ms = (smv / itc) / 1e6
            out_rows.append(
                {
                    "benchmark": key,
                    "language": "rust",
                    "input_size": input_size,
                    "trial": i,
                    "kernel_time_ms": ms,
                    "h2d_time_ms": 0.0,
                    "d2h_time_ms": 0.0,
                    "gflops": 0.0,
                    "bandwidth_gbps": 0.0,
                }
            )

    if not out_rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        return 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    return len(out_rows)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--gbench-json", type=Path, default=RESULTS_RAW / "cpu_gbench.json")
    args = p.parse_args()

    n_cpp = 0
    if args.gbench_json.is_file():
        n_cpp = export_cpp_from_json(args.gbench_json, RESULTS_RAW / "cpu_cpp.csv")
        print(f"  Wrote {n_cpp} C++ CPU rows -> {RESULTS_RAW / 'cpu_cpp.csv'}")
    else:
        print(f"  Skip C++ export: missing {args.gbench_json}")

    n_rust = export_rust_from_criterion(RESULTS_RAW / "cpu_rust.csv")
    print(f"  Wrote {n_rust} Rust CPU rows -> {RESULTS_RAW / 'cpu_rust.csv'}")


if __name__ == "__main__":
    main()
