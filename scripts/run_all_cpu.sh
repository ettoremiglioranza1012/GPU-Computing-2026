#!/bin/bash
# run_all_cpu.sh — sweep all CPU kernels × all downloaded matrices
#
# Run from the repo root:  bash scripts/run_all_cpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

KERNELS=(spmv_csr_naive spmv_csr_opt)

# Collect all .mtx files under Data/
mapfile -t MATRICES < <(find "$REPO_DIR/Data" -name "*.mtx" 2>/dev/null | sort)

if [ ${#MATRICES[@]} -eq 0 ]; then
    echo "No matrices found in Data/. Run scripts/download_data.sh first."
    exit 1
fi

for kernel in "${KERNELS[@]}"; do
    bin="$REPO_DIR/bin/CPU/$kernel"
    if [ ! -f "$bin" ]; then
        echo "[skip] $kernel not built — run: make cpu"
        continue
    fi
    for mtx in "${MATRICES[@]}"; do
        bash "$SCRIPT_DIR/run_cpu.sh" "$kernel" "$mtx"
    done
done
