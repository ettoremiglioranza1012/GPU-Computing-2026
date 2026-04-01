#!/bin/bash
# run_cpu.sh — submit a CPU SpMV benchmark job to SLURM
#
# Usage:
#   bash scripts/run_cpu.sh <binary_name> <matrix_path>
#
# Example:
#   bash scripts/run_cpu.sh spmv_csr_naive Data/1138_bus/1138_bus.mtx
#   bash scripts/run_cpu.sh spmv_csr_opt   Data/bcsstk17/bcsstk17.mtx

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <binary_name> <matrix_path>"
    echo "  binary_name:  name inside bin/CPU/ (e.g. spmv_csr_naive)"
    echo "  matrix_path:  path to .mtx file    (e.g. Data/1138_bus/1138_bus.mtx)"
    exit 1
fi

BINARY="$1"
MATRIX="$2"
MAT_NAME="$(basename "$MATRIX" .mtx)"
JOB_NAME="${BINARY}_${MAT_NAME}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$REPO_DIR/outputs"

sbatch \
    --partition=edu-short \
    --account=gpu.computing26 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --gres=gpu:0 \
    --job-name="$JOB_NAME" \
    --output="$REPO_DIR/outputs/R-%x.%j.out" \
    --error="$REPO_DIR/outputs/R-%x.%j.err" \
    --wrap="cd $REPO_DIR && $REPO_DIR/bin/CPU/$BINARY $MATRIX"

echo "Submitted: $JOB_NAME"
echo "Output:    outputs/R-${JOB_NAME}.<jobid>.out"
