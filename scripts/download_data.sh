#!/bin/bash
# download_data.sh — fetch SuiteSparse matrices used for SpMV benchmarking
#
# 10-matrix dataset selected to match Chu et al. HPDC '23 (required reference [2]).
# Covers structured FEM, circuit simulation, and power-law graphs for full contrast.
#
#   Structural / FEM (uniform rows, high regularity):
#     1138_bus      (1.1K rows,    1.4K NNZ)   — tiny debug matrix
#     bcsstk17     (11K rows,    429K NNZ)   — medium FEM
#     bone010      (987K rows,   36.3M NNZ)  — large FEM, very structured
#     ldoor        (952K rows,   42.5M NNZ)  — large FEM/LP structural
#     Rucci1       (1.98M rows,   7.8M NNZ)  — land survey, semi-structured
#
#   Circuit / mixed regularity:
#     rajat31      (4.69M rows,  20.3M NNZ)  — circuit simulation, irregular
#
#   Power-law graphs (irregular, skewed row lengths):
#     web-Google   (916K rows,    5.1M NNZ)  — web crawl, power-law
#     eu-2005      (863K rows,   16.1M NNZ)  — European web graph
#     webbase-1M   (1M rows,      3.1M NNZ)  — web graph (Williams)
#     hollywood-2009 (1.1M rows, 112M NNZ)   — actor co-appearance, dense

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../Data"
BASE_URL="https://suitesparse-collection-website.herokuapp.com/MM"

mkdir -p "$DATA_DIR"

# pick downloader
if command -v wget &>/dev/null; then
    download() { wget -q --show-progress -P "$DATA_DIR" "$1"; }
elif command -v curl &>/dev/null; then
    download() { curl -L --progress-bar -o "$DATA_DIR/$(basename "$1")" "$1"; }
else
    echo "Error: neither wget nor curl found." >&2
    exit 1
fi

# Array of "Group/Name" entries from SuiteSparse
# Verify URLs at https://sparse.tamu.edu if a download returns 404
MATRICES=(
    "HB/1138_bus"
    "HB/bcsstk17"
    "SNAP/web-Google"
    "Wissgott/bone010"
    "GHS_psdef/ldoor"
    "Rucci/Rucci1"
    "Rajat/rajat31"
    "LAW/eu-2005"
    "Williams/webbase-1M"
    "LAW/hollywood-2009"
)

for entry in "${MATRICES[@]}"; do
    group="${entry%%/*}"
    name="${entry##*/}"
    tarball="${name}.tar.gz"
    target_dir="$DATA_DIR/$name"

    if [ -f "$target_dir/${name}.mtx" ]; then
        echo "[skip] $name already present"
        continue
    fi

    echo "[download] $name ..."
    download "${BASE_URL}/${group}/${tarball}"

    echo "[extract]  $name ..."
    tar -xzf "$DATA_DIR/$tarball" -C "$DATA_DIR"
    rm "$DATA_DIR/$tarball"

    echo "[done]     $name -> $target_dir/${name}.mtx"
done

echo ""
echo "All matrices ready in $DATA_DIR"
