#!/bin/bash
# ============================================================================
# Assembly Diff Helper
# ============================================================================
# Compiles C++ and Rust to assembly and produces a side-by-side diff.
# Useful for verifying that both compilers generate equivalent optimizations.
#
# Usage:
#   ./asm_diff.sh quicksort    # Compare QuickSort assembly
#   ./asm_diff.sh matmul       # Compare MatMul assembly
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ASM_DIR="$PROJECT_ROOT/results/assembly"
mkdir -p "$ASM_DIR"

BENCHMARK="${1:-quicksort}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Assembly Diff: $BENCHMARK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── C++ Assembly ─────────────────────────────────────────────────────────────
CPP_SRC="$PROJECT_ROOT/cpu/cpp/src/${BENCHMARK}.cpp"
CPP_ASM="$ASM_DIR/${BENCHMARK}_cpp.s"

if [ -f "$CPP_SRC" ]; then
    echo "  ▸ Compiling C++ to assembly..."
    g++ -S -O3 -march=native -std=c++17 \
        -I"$PROJECT_ROOT/cpu/cpp/include" \
        -o "$CPP_ASM" "$CPP_SRC" 2>/dev/null || {
            echo "  ⚠ C++ assembly generation failed (likely missing Google Benchmark headers)"
            echo "    Trying with minimal flags..."
            g++ -S -O3 -march=native -std=c++17 \
                -o "$CPP_ASM" "$CPP_SRC" 2>/dev/null || true
        }
    
    if [ -f "$CPP_ASM" ]; then
        # Strip metadata, keep only code
        LINES=$(wc -l < "$CPP_ASM")
        echo "  ✓ C++ assembly: $CPP_ASM ($LINES lines)"
    fi
else
    echo "  ⚠ C++ source not found: $CPP_SRC"
fi

# ── Rust Assembly ────────────────────────────────────────────────────────────
RUST_ASM="$ASM_DIR/${BENCHMARK}_rust.s"

echo "  ▸ Compiling Rust to assembly..."
cd "$PROJECT_ROOT/cpu/rust"
RUSTFLAGS="--emit=asm" cargo build --release 2>/dev/null || true

# Find the generated assembly
ASM_FOUND=""
for f in target/release/deps/*.s; do
    if [ -f "$f" ]; then
        cp "$f" "$RUST_ASM"
        ASM_FOUND="1"
        break
    fi
done

if [ -n "$ASM_FOUND" ] && [ -f "$RUST_ASM" ]; then
    LINES=$(wc -l < "$RUST_ASM")
    echo "  ✓ Rust assembly: $RUST_ASM ($LINES lines)"
else
    echo "  ⚠ Rust assembly generation failed"
fi

# ── Diff ─────────────────────────────────────────────────────────────────────
if [ -f "$CPP_ASM" ] && [ -f "$RUST_ASM" ]; then
    DIFF_FILE="$ASM_DIR/${BENCHMARK}_diff.txt"
    
    echo ""
    echo "  ▸ Generating diff..."
    
    # Filter to just instructions (remove metadata/directives)
    grep -E '^\s+[a-z]' "$CPP_ASM" | head -100 > "$ASM_DIR/${BENCHMARK}_cpp_filtered.s" 2>/dev/null || true
    grep -E '^\s+[a-z]' "$RUST_ASM" | head -100 > "$ASM_DIR/${BENCHMARK}_rust_filtered.s" 2>/dev/null || true
    
    diff --side-by-side --width=120 \
        "$ASM_DIR/${BENCHMARK}_cpp_filtered.s" \
        "$ASM_DIR/${BENCHMARK}_rust_filtered.s" \
        > "$DIFF_FILE" 2>/dev/null || true
    
    if [ -f "$DIFF_FILE" ]; then
        DIFF_LINES=$(wc -l < "$DIFF_FILE")
        echo "  ✓ Diff: $DIFF_FILE ($DIFF_LINES lines)"
        echo ""
        echo "  ── First 30 lines of diff ──"
        head -30 "$DIFF_FILE" | sed 's/^/  │ /'
        echo "  └─"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Output directory: $ASM_DIR/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
