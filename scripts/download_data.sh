#!/bin/bash
# download_data.sh — fetch SuiteSparse matrices used for SpMV benchmarking
#
# Matrices chosen to cover:
#   - Small/debug:        1138_bus      (1138 rows,     1416 NNZ)
#   - Medium structural:  bcsstk17     (10974 rows,  428650 NNZ)
#   - Medium power-law:   web-Google   (916428 rows, 5105039 NNZ)  [SNAP group]
#   - Large power-law:    soc-LiveJournal1 (4847571 rows, 68993773 NNZ) [SNAP]
#
# Adjust the list below to add/remove matrices.

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
MATRICES=(
    "HB/1138_bus"
    "HB/bcsstk17"
    "SNAP/web-Google"
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
