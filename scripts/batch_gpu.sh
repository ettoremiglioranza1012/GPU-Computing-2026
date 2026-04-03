#!/bin/bash
# batch_gpu.sh — launcher: submits one SLURM job per (kernel, matrix) pair.
#
# NOT a SLURM job itself — run directly from the repo root:
#   bash scripts/batch_gpu.sh
#
# Each job runs a single kernel on a single matrix, so wall time stays well
# within the edu-short 5-minute limit even for large matrices.
#
# After all jobs finish:
#   uv run scripts/parse_results.py outputs/R-spmv_gpu_*.txt --out results_tables/results.csv
#   uv run scripts/plot_results.py results_tables/results.csv

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KERNELS=(spmv_gpu_tpv spmv_gpu_tpr spmv_gpu_stride)

# ── Best config for spmv_gpu_stride (auto-loaded from sweep results) ──────────
# Run scripts/sweep_gpu.sh → parse_results.py → plot_sweep.py first.
# Then write scripts/best_gpu_config.sh with BEST_BLOCKS and BEST_THREADS.
# If the file does not exist, the stride kernel falls back to its built-in
# defaults (256 threads, auto grid).
BEST_BLOCKS=""
BEST_THREADS=""
CFG_FILE="$REPO_DIR/scripts/best_gpu_config.sh"
if [ -f "$CFG_FILE" ]; then
    # shellcheck source=/dev/null
    source "$CFG_FILE"
    echo "[config] loaded best sweep config: blocks=$BEST_BLOCKS threads=$BEST_THREADS"
else
    echo "[config] no best_gpu_config.sh found — stride kernel uses built-in defaults"
fi

# Explicit allowlist — matches the 10 intended matrices from download_data.sh.
# Do NOT use find here: bone010's tarball contains auxiliary files
# (bone010_B.mtx, bone010_C.mtx, bone010_M.mtx) that are not real matrices.
MATRIX_NAMES=(
    bone010
    ldoor
    Rucci1
    nlpkkt80
    ASIC_680ks
    rajat31
    boyd2
    eu-2005
    webbase-1M
    hollywood-2009
)

MATRICES=()
for name in "${MATRIX_NAMES[@]}"; do
    mtx="$REPO_DIR/Data/${name}/${name}.mtx"
    if [ -f "$mtx" ]; then
        MATRICES+=("$mtx")
    else
        echo "[warn] matrix not found, skipping: $mtx"
    fi
done

if [ ${#MATRICES[@]} -eq 0 ]; then
    echo "No matrices found in Data/. Run scripts/download_data.sh first."
    exit 1
fi

echo "Kernels:  ${KERNELS[*]}"
echo "Matrices: ${#MATRICES[@]}"
echo "Total jobs: $((${#KERNELS[@]} * ${#MATRICES[@]}))"
echo ""

mkdir -p "$REPO_DIR/outputs"

for kernel in "${KERNELS[@]}"; do
    bin="$REPO_DIR/bin/GPU/${kernel}.exec"
    if [ ! -f "$bin" ]; then
        echo "[skip] $kernel not built"
        continue
    fi

    for mtx in "${MATRICES[@]}"; do
        mtx_name=$(basename "$mtx")
        job_name="spmv_gpu_${kernel}_${mtx_name%.mtx}"

        if [ "$kernel" = "spmv_gpu_stride" ] && \
           [ -n "$BEST_THREADS" ] && [ -n "$BEST_BLOCKS" ]; then
            extra_args="$BEST_THREADS $BEST_BLOCKS"
        else
            extra_args=""
        fi

        JOB_SUBMISSION=$(sbatch \
            --partition=edu-short \
            --account=gpu.computing26 \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=1 \
            --gres=gpu:1 \
            --job-name="$job_name" \
            --output="$REPO_DIR/outputs/R-${job_name}.%j.txt" \
            --error="$REPO_DIR/outputs/R-${job_name}.%j.err" \
            --wrap="echo '========================================' && \
echo 'KERNEL: ${kernel}  MATRIX: ${mtx_name}' && \
echo '========================================' && \
\"${bin}\" \"${mtx}\" ${extra_args}")

        JOB_ID=$(echo "$JOB_SUBMISSION" | grep -oP 'Submitted batch job \K[0-9]+')
        echo "[submitted] $kernel × $mtx_name  job_id=$JOB_ID"
    done
done

echo ""
echo "All jobs submitted."
echo "When complete, run:"
echo "  uv run scripts/parse_results.py \"$REPO_DIR/outputs/R-spmv_gpu_*.txt\" --out results_tables/results.csv"
echo "  uv run scripts/plot_results.py results_tables/results.csv"
