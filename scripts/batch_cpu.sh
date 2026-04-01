#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --job-name=spmv_cpu
#SBATCH --output=outputs/R-%x.%j.txt
#SBATCH --error=outputs/R-%x.%j.err

# Run from repo root: sbatch scripts/batch_cpu.sh

# BASH_SOURCE[0] is unreliable inside sbatch (SLURM copies the script to a temp dir).
# Use REPO_DIR env var if set (passed via --export), otherwise fall back to the
# directory from which sbatch was invoked (SLURM sets SLURM_SUBMIT_DIR).
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
KERNELS=(spmv_coo_naive spmv_coo_opt)

mapfile -t MATRICES < <(find "$REPO_DIR/Data" -name "*.mtx" | sort)

if [ ${#MATRICES[@]} -eq 0 ]; then
    echo "No matrices found in Data/. Run scripts/download_data.sh first."
    exit 1
fi

for kernel in "${KERNELS[@]}"; do
    bin="$REPO_DIR/bin/CPU/$kernel"
    if [ ! -f "$bin" ]; then
        echo "[skip] $kernel not built"
        continue
    fi
    for mtx in "${MATRICES[@]}"; do
        echo "========================================"
        echo "KERNEL: $kernel  MATRIX: $(basename $mtx)"
        echo "========================================"
        "$bin" "$mtx"
    done
done
