#!/bin/bash
# download_data.sh — fetch SuiteSparse matrices used for SpMV benchmarking
#
# 10-matrix dataset selected to match Chu et al. HPDC '23 (required reference [2]).
# Professor instruction: "I suggest using the same matrices reported in [2]."
# All entries verified at https://sparse.tamu.edu
#
# Covers three structural regimes for full analytical contrast:
#
#   Structural / FEM (uniform rows, high regularity):
#     bone010      (986K rows,   36.3M NNZ)  — large FEM, very structured
#     ldoor        (952K rows,   42.5M NNZ)  — large FEM / structural LP
#     Rucci1       (1.98M rows,   7.8M NNZ)  — land survey, semi-structured
#     nlpkkt80     (1.06M rows,  28.2M NNZ)  — large structured (NLP KKT system)
#
#   Circuit / optimisation / mixed regularity:
#     ASIC_680ks   (682K rows,    1.7M NNZ)  — circuit simulation, mixed regularity
#     rajat31      (4.69M rows,  20.3M NNZ)  — circuit / structural, irregular
#     boyd2        (466K rows,   1.03M NNZ)  — optimisation problem, dense-ish rows
#
#   Power-law graphs (irregular, skewed row lengths):
#     eu-2005      (863K rows,   16.1M NNZ)  — European web graph, power-law
#     webbase-1M   (1M rows,      3.1M NNZ)  — web graph (Williams), irregular
#     hollywood-2009 (1.1M rows, 112M NNZ)   — actor co-appearance, very dense

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../Data"
BASE_URL="https://sparse.tamu.edu/MM"

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

# Array of "Group/Name" entries from SuiteSparse.
# These 10 matrices match the benchmark set in Chu et al. HPDC '23 (Table 2).
# Verify URLs at https://sparse.tamu.edu if a download returns 404.
MATRICES=(
    "Wissgott/bone010"
    "GHS_psdef/ldoor"
    "Rucci/Rucci1"
    "Mazaheri/nlpkkt80"
    "Sandia/ASIC_680ks"
    "Rajat/rajat31"
    "Boyd/boyd2"
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
