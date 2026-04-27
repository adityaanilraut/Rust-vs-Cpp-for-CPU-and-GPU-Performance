<#
.SYNOPSIS
    Rust vs. C++ Benchmark Runner (Windows / PowerShell)
    Replaces the Linux Makefile for Windows environments.

.DESCRIPTION
    Runs every phase of the benchmark study in sequence:
      0. Prerequisite check
      1. Python dependency install
      2. Dataset generation
      3. C++ CPU  — build (cmake) + run (Google Benchmark)
      4. Rust CPU — build (cargo) + run (Criterion)
      5. Export Criterion + GBench results to unified CSVs
      6. Statistical analysis  (analyze_results.py)
      7. GPU analysis table   (generate_gpu_table.py)   [if GPU CSVs present]
      8. Publication plots    (generate_plots.py)
      9. Safety metrics       (safety_analysis.py)

.PARAMETER SkipBuild
    Skip cmake / cargo build steps (use pre-built binaries).

.PARAMETER SkipDatasets
    Skip dataset generation (datasets already exist).

.PARAMETER CpuOnly
    Skip GPU table generation even if CSV files are present.

.PARAMETER NTrials
    Number of Google Benchmark repetitions (default: 30).

.EXAMPLE
    .\run_benchmarks.ps1
    .\run_benchmarks.ps1 -SkipBuild -SkipDatasets
    .\run_benchmarks.ps1 -NTrials 10
#>

param(
    [switch]$SkipBuild,
    [switch]$SkipDatasets,
    [switch]$CpuOnly,
    [int]$NTrials = 10
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Force UTF-8 output so Python's Unicode box-drawing characters render correctly
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding             = [System.Text.Encoding]::UTF8

$Root       = $PSScriptRoot
$CpuCppDir  = Join-Path $Root "cpu\cpp"
$CpuRustDir = Join-Path $Root "cpu\rust"
$GpuCppDir  = Join-Path $Root "gpu\cpp"
$GpuRustDir = Join-Path $Root "gpu\rust"
$ScriptsDir = Join-Path $Root "scripts"
$ResultsRaw = Join-Path $Root "results\raw"
$ResultsFig = Join-Path $Root "results\figures"

$CppBuildDir    = Join-Path $CpuCppDir  "build"
$CppExe         = Join-Path $CppBuildDir "Release\cpu_benchmarks.exe"
$GpuCppBuildDir = Join-Path $GpuCppDir  "build"
$GpuCppExe      = Join-Path $GpuCppBuildDir "Release\gpu_benchmarks.exe"

$Divider = "=" * 60

function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host $Divider -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host $Divider -ForegroundColor Cyan
}

function Write-Ok([string]$msg)   { Write-Host "  OK  $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "  WARN $msg" -ForegroundColor Yellow }
function Write-Fail([string]$msg) { Write-Host "  FAIL $msg" -ForegroundColor Red }

# ── 0. Prerequisite Check ────────────────────────────────────────────────────
Write-Step "0 / 9  Prerequisite Check"

$missing = @()

foreach ($tool in @("python", "cmake", "cargo", "rustc")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        $missing += $tool
        Write-Fail "$tool not found in PATH"
    } else {
        $ver = & $tool --version 2>&1 | Select-Object -First 1
        Write-Ok "$tool  →  $ver"
    }
}

$nvccAvailable = $null -ne (Get-Command "nvcc" -ErrorAction SilentlyContinue)
if ($nvccAvailable) {
    $ver = & nvcc --version 2>&1 | Select-String "release"
    Write-Ok "nvcc  →  $ver"
} else {
    Write-Warn "nvcc not found — GPU benchmarks will be SKIPPED"
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Fail "Missing required tools: $($missing -join ', ')"
    Write-Host "  Install them and re-run." -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path $ResultsRaw | Out-Null
New-Item -ItemType Directory -Force -Path $ResultsFig | Out-Null

# ── 1. Python Dependencies ───────────────────────────────────────────────────
Write-Step "1 / 9  Install Python Dependencies"
& python -m pip install -q -r (Join-Path $ScriptsDir "requirements.txt")
Write-Ok "Python packages ready"

# ── 2. Dataset Generation ────────────────────────────────────────────────────
Write-Step "2 / 9  Generate Datasets"
if ($SkipDatasets) {
    Write-Warn "Skipping dataset generation (-SkipDatasets)"
} else {
    & python (Join-Path $ScriptsDir "generate_datasets.py")
    Write-Ok "Datasets written to datasets\"
}

# ── 3. C++ CPU  — Build ──────────────────────────────────────────────────────
Write-Step "3a / 9  Build C++ CPU Benchmarks"
if ($SkipBuild -and (Test-Path $CppExe)) {
    Write-Warn "Skipping C++ build (-SkipBuild) — using existing binary"
} else {
    New-Item -ItemType Directory -Force -Path $CppBuildDir | Out-Null
    Push-Location $CppBuildDir
    try {
        Write-Host "  cmake configure …"
        & cmake -DCMAKE_BUILD_TYPE=Release .. 2>&1 | Write-Host
        if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

        Write-Host "  cmake build …"
        & cmake --build . --config Release --parallel 2>&1 | Write-Host
        if ($LASTEXITCODE -ne 0) { throw "CMake build failed" }

        Write-Ok "cpu_benchmarks.exe built"
    } finally {
        Pop-Location
    }
}

# ── 3b. C++ CPU  — Run ───────────────────────────────────────────────────────
Write-Step "3b / 9  Run C++ CPU Benchmarks  ($NTrials repetitions)"
$GbenchJson = Join-Path $ResultsRaw "cpu_gbench.json"
if (-not (Test-Path $CppExe)) {
    Write-Fail "cpu_benchmarks.exe not found — cannot run C++ benchmarks"
    Write-Warn "Run without -SkipBuild to build first"
} else {
    & $CppExe `
        "--benchmark_repetitions=$NTrials" `
        "--benchmark_min_time=1.0s" `
        "--benchmark_out=$GbenchJson" `
        "--benchmark_out_format=json" `
        "--benchmark_report_aggregates_only=true" 2>&1 | Write-Host
    if ($LASTEXITCODE -ne 0) { Write-Fail "C++ benchmark run exited with code $LASTEXITCODE" }
    else { Write-Ok "Results → $GbenchJson" }
}

# ── 4. Rust CPU  — Build + Run ───────────────────────────────────────────────
Write-Step "4 / 9  Build + Run Rust CPU Benchmarks  ($NTrials samples)"
Push-Location $CpuRustDir
try {
    if (-not $SkipBuild) {
        Write-Host "  cargo build --release …"
        & cargo build --release 2>&1 | Write-Host
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
    }
    Write-Host "  cargo bench …"
    & cargo bench 2>&1 | Write-Host
    if ($LASTEXITCODE -ne 0) { Write-Fail "cargo bench exited with code $LASTEXITCODE" }
    else { Write-Ok "Criterion results → cpu\rust\target\criterion\" }
} finally {
    Pop-Location
}

# ── 5. GPU Benchmarks ────────────────────────────────────────────────────────
Write-Step "5 / 9  GPU Benchmarks"
if (-not $nvccAvailable -or $CpuOnly) {
    Write-Warn "Skipping GPU benchmarks (nvcc not available or -CpuOnly)"
} else {
    # Build GPU C++
    New-Item -ItemType Directory -Force -Path $GpuCppBuildDir | Out-Null
    Push-Location $GpuCppBuildDir
    try {
        & cmake .. 2>&1 | Write-Host
        & cmake --build . --config Release --parallel 2>&1 | Write-Host
    } finally { Pop-Location }

    # Compile PTX kernels for Rust GPU
    Push-Location $GpuRustDir
    try {
        & nvcc -ptx kernels/matmul.cu   -o kernels/matmul.ptx
        & nvcc -ptx kernels/softmax.cu  -o kernels/softmax.ptx
        & nvcc -ptx kernels/reduction.cu -o kernels/reduction.ptx
        & cargo build --release 2>&1 | Write-Host
    } finally { Pop-Location }

    # Run GPU C++
    if (Test-Path $GpuCppExe) {
        & $GpuCppExe 2>&1 | Write-Host
        Write-Ok "GPU C++ done"
    }
    # Run GPU Rust
    Push-Location $GpuRustDir
    try {
        & cargo run --release 2>&1 | Write-Host
        Write-Ok "GPU Rust done"
    } finally { Pop-Location }
}

# ── 6. Export CPU Results to CSV ─────────────────────────────────────────────
Write-Step "6 / 9  Export CPU Results to CSV"
& python (Join-Path $ScriptsDir "export_cpu_results_to_csv.py") `
    "--gbench-json" $GbenchJson 2>&1 | Write-Host
Write-Ok "cpu_cpp.csv + cpu_rust.csv ready"

# ── 7. Statistical Analysis ───────────────────────────────────────────────────
Write-Step "7 / 9  Statistical Analysis"
& python (Join-Path $ScriptsDir "analyze_results.py") 2>&1 | Write-Host
Write-Ok "analysis_summary.csv + analysis_table.tex written"

# ── 7b. GPU Analysis Table ────────────────────────────────────────────────────
$GpuCppCsv  = Join-Path $ResultsRaw "gpu_cpp.csv"
$GpuRustCsv = Join-Path $ResultsRaw "gpu_rust.csv"
if ((Test-Path $GpuCppCsv) -and (Test-Path $GpuRustCsv) -and -not $CpuOnly) {
    Write-Host ""
    Write-Host "  Running GPU table analysis …" -ForegroundColor Cyan
    & python (Join-Path $ScriptsDir "generate_gpu_table.py") 2>&1 | Write-Host
    Write-Ok "gpu_analysis_summary.csv + gpu_analysis_table.tex written"
} else {
    Write-Warn "Skipping GPU table (no GPU CSV files found)"
}

# ── 8. Generate Plots ─────────────────────────────────────────────────────────
Write-Step "8 / 9  Generate Publication Plots"
& python (Join-Path $ScriptsDir "generate_plots.py") 2>&1 | Write-Host
Write-Ok "Figures saved to results\figures\"

# ── 9. Safety Metrics ─────────────────────────────────────────────────────────
Write-Step "9 / 9  Safety & Developer Metrics"
& python (Join-Path $ScriptsDir "safety_analysis.py") 2>&1 | Write-Host
Write-Ok "safety_metrics.json updated"

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host $Divider -ForegroundColor Green
Write-Host "  ALL PHASES COMPLETE" -ForegroundColor Green
Write-Host $Divider -ForegroundColor Green
Write-Host ""
Write-Host "  Raw results : $ResultsRaw"
Write-Host "  Figures     : $ResultsFig"
Write-Host ""
