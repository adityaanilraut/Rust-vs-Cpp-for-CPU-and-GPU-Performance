#!/usr/bin/env python3
"""
Safety & Developer Metrics Analyzer
=====================================
Measures H3 (safety-performance trade-off):
- Lines of code (LOC) per language per benchmark
- `unsafe` block count in Rust code
- Compile time comparison
- Outputs comparison table (CSV + LaTeX)
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def count_loc(directory: Path, extensions: list) -> dict:
    """Count non-blank, non-comment lines of code."""
    results = {}
    for ext in extensions:
        for path in directory.rglob(f"*.{ext}"):
            if "target" in str(path) or "build" in str(path):
                continue
            with open(path, 'r', errors='ignore') as f:
                lines = f.readlines()
            code_lines = sum(1 for line in lines
                           if line.strip()
                           and not line.strip().startswith("//")
                           and not line.strip().startswith("/*")
                           and not line.strip().startswith("*"))
            results[str(path.relative_to(directory.parent.parent))] = code_lines
    return results


def count_unsafe_blocks(directory: Path) -> list:
    """Count `unsafe` blocks in Rust source files."""
    unsafe_locations = []
    for path in directory.rglob("*.rs"):
        if "target" in str(path):
            continue
        with open(path, 'r') as f:
            for i, line in enumerate(f, 1):
                if re.search(r'\bunsafe\b', line):
                    unsafe_locations.append({
                        "file": str(path.relative_to(directory.parent.parent)),
                        "line": i,
                        "content": line.strip()
                    })
    return unsafe_locations


def measure_compile_time(command: list, cwd: str, clean_cmd: list | None = None,
                         runs: int = 3) -> float:
    """Measure average compile time over multiple runs (cross-platform)."""
    times = []
    for _ in range(runs):
        if clean_cmd:
            subprocess.run(clean_cmd, cwd=cwd, capture_output=True)
        start = time.time()
        result = subprocess.run(command, cwd=cwd, capture_output=True)
        elapsed = time.time() - start
        if result.returncode == 0:
            times.append(elapsed)
    return sum(times) / len(times) if times else -1


def main():
    project_root = Path(__file__).resolve().parent.parent

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Safety & Developer Metrics Analysis")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ── Lines of Code ──
    print("\n  ┌─ Lines of Code")
    cpp_cpu = count_loc(project_root / "cpu" / "cpp", ["cpp", "h", "hpp"])
    cpp_gpu = count_loc(project_root / "gpu" / "cpp", ["cu", "cuh", "h"])
    rust_cpu = count_loc(project_root / "cpu" / "rust", ["rs"])
    rust_gpu = count_loc(project_root / "gpu" / "rust", ["rs"])

    cpp_total = sum(cpp_cpu.values()) + sum(cpp_gpu.values())
    rust_total = sum(rust_cpu.values()) + sum(rust_gpu.values())

    print(f"  │  C++ CPU:  {sum(cpp_cpu.values()):>6} LOC across {len(cpp_cpu)} files")
    print(f"  │  C++ GPU:  {sum(cpp_gpu.values()):>6} LOC across {len(cpp_gpu)} files")
    print(f"  │  C++ Total: {cpp_total:>5} LOC")
    print(f"  │")
    print(f"  │  Rust CPU: {sum(rust_cpu.values()):>6} LOC across {len(rust_cpu)} files")
    print(f"  │  Rust GPU: {sum(rust_gpu.values()):>6} LOC across {len(rust_gpu)} files")
    print(f"  │  Rust Total: {rust_total:>4} LOC")
    print(f"  │")

    ratio = rust_total / cpp_total if cpp_total > 0 else 0
    print(f"  │  Rust/C++ LOC ratio: {ratio:.2f}x")
    print(f"  └─")

    # ── Unsafe Blocks ──
    print("\n  ┌─ Unsafe Block Analysis (Rust)")
    unsafe_cpu = count_unsafe_blocks(project_root / "cpu" / "rust")
    unsafe_gpu = count_unsafe_blocks(project_root / "gpu" / "rust")

    if not unsafe_cpu and not unsafe_gpu:
        print("  │  ✓ No `unsafe` blocks found — fully safe Rust code!")
    else:
        for u in unsafe_cpu + unsafe_gpu:
            print(f"  │  ⚠ {u['file']}:{u['line']}  {u['content']}")
        print(f"  │  Total unsafe blocks: {len(unsafe_cpu) + len(unsafe_gpu)}")
    print(f"  └─")

    # ── Compile Times ──
    print("\n  ┌─ Compile Times (average of 3 runs)")

    cpp_build_dir = project_root / "cpu" / "cpp" / "build"
    cpp_build_dir.mkdir(parents=True, exist_ok=True)
    # Configure once if needed, then time incremental builds
    subprocess.run(
        ["cmake", "-DCMAKE_BUILD_TYPE=Release", ".."],
        cwd=str(cpp_build_dir), capture_output=True
    )
    cpp_compile = measure_compile_time(
        ["cmake", "--build", ".", "--config", "Release"],
        cwd=str(cpp_build_dir),
        clean_cmd=None,  # avoid full clean between runs; measures incremental build
    )
    rust_compile = measure_compile_time(
        ["cargo", "build", "--release"],
        cwd=str(project_root / "cpu" / "rust"),
        clean_cmd=["cargo", "clean"],
    )

    if cpp_compile > 0:
        print(f"  │  C++ CPU:  {cpp_compile:.1f}s")
    else:
        print(f"  │  C++ CPU:  (build failed or not available)")
    if rust_compile > 0:
        print(f"  │  Rust CPU: {rust_compile:.1f}s")
    else:
        print(f"  │  Rust CPU: (build failed or not available)")

    if cpp_compile > 0 and rust_compile > 0:
        print(f"  │  Ratio: {rust_compile/cpp_compile:.2f}x (Rust/C++)")
    print(f"  └─")

    # ── Summary Output ──
    output_dir = project_root / "results" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "loc": {
            "cpp_cpu": sum(cpp_cpu.values()),
            "cpp_gpu": sum(cpp_gpu.values()),
            "rust_cpu": sum(rust_cpu.values()),
            "rust_gpu": sum(rust_gpu.values()),
            "ratio": round(ratio, 2),
        },
        "unsafe_blocks": {
            "cpu": len(unsafe_cpu),
            "gpu": len(unsafe_gpu),
            "total": len(unsafe_cpu) + len(unsafe_gpu),
            "locations": unsafe_cpu + unsafe_gpu,
        },
        "compile_time": {
            "cpp_cpu_seconds": round(cpp_compile, 1) if cpp_compile > 0 else None,
            "rust_cpu_seconds": round(rust_compile, 1) if rust_compile > 0 else None,
        },
    }

    with open(output_dir / "safety_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ✓ Metrics saved to {output_dir / 'safety_metrics.json'}")

    # ── H3 Evaluation ──
    print("\n━━ H3 Evaluation: Safety-Performance Trade-off ━━━")
    total_unsafe = len(unsafe_cpu) + len(unsafe_gpu)
    if total_unsafe == 0:
        print("  H3 SUPPORTED: Zero unsafe blocks, demonstrating that")
        print("  Rust's safety guarantees incur no explicit opt-out cost.")
    else:
        print(f"  {total_unsafe} unsafe block(s) found.")
        print("  Investigate whether these are performance-critical or merely FFI boundaries.")


if __name__ == "__main__":
    main()
