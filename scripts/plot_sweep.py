#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.9",
#     "pandas>=2.2",
#     "numpy>=1.26",
# ]
# ///
"""plot_sweep.py — heatmap of GPU block/thread configuration sweep.

Reads the CSV produced by parse_results.py (which already handles the
sweep output since each configuration is encoded in the KERNEL: banner
as  spmv_gpu_stride_bN_tM). Produces an annotated heatmap:
  - rows    = num_blocks  [1, 4, 8, 16, 32, 64, 128]
  - columns = threads_per_block  [32, 128, 256, 512, 1024]
  - cell colour + annotation = effective bandwidth (GB/s)

Usage:
    uv run scripts/plot_sweep.py results.csv
    uv run scripts/plot_sweep.py results.csv --out assets/
    uv run scripts/plot_sweep.py results.csv --metric gflops

NOTE (from_development_to_large_scale_matrices.md):
    The title label is read from the 'matrix' column of the first matching
    row. After the large-matrix upgrade, rerun the sweep on the new large
    structured matrix and this script requires no changes.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── regex to parse config out of kernel name ──────────────────────────────────
# matches: spmv_gpu_stride_b<blocks>_t<threads>
RE_CONFIG = re.compile(r"spmv_gpu_stride_b(\d+)_t(\d+)$")

plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def parse_config(kernel: str):
    """Return (blocks, threads) from kernel name, or None if not a sweep row."""
    m = RE_CONFIG.match(kernel)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def build_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Extract sweep rows and pivot into blocks × threads table."""
    rows = []
    for _, row in df.iterrows():
        cfg = parse_config(str(row["kernel"]))
        if cfg is None:
            continue
        blocks, threads = cfg
        rows.append({"blocks": blocks, "threads": threads, "value": row[metric]})

    if not rows:
        print("[error] no sweep rows found in CSV (expected kernels named "
              "spmv_gpu_stride_bN_tM)", file=sys.stderr)
        sys.exit(1)

    sweep = pd.DataFrame(rows)
    pivot = sweep.pivot_table(index="blocks", columns="threads", values="value")
    # sort axes so heatmap reads low→high left and top→bottom
    pivot = pivot.sort_index(ascending=True)           # blocks: small at top
    pivot = pivot[sorted(pivot.columns, reverse=False)] # threads: small at left
    return pivot


def plot_heatmap(pivot: pd.DataFrame, metric: str, matrix_name: str,
                 out_dir: Path):
    """Annotated heatmap: rows=blocks, cols=threads_per_block, colour=metric."""
    data = pivot.values.astype(float)

    ylabel = "Effective Bandwidth (GB/s)" if metric == "bandwidth_GBs" else "GFLOPS"
    unit   = "GB/s"                        if metric == "bandwidth_GBs" else "GFLOP/s"
    fname  = "gpu_config_sweep.png"        if metric == "bandwidth_GBs" \
             else "gpu_config_sweep_gflops.png"

    n_blocks  = len(pivot.index)
    n_threads = len(pivot.columns)

    fig, ax = plt.subplots(figsize=(max(7, n_threads * 1.6),
                                    max(4, n_blocks  * 0.9)))

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd",
                   vmin=np.nanmin(data) * 0.85,
                   vmax=np.nanmax(data) * 1.05)

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(f"{ylabel} ({unit})", fontsize=10)

    # Axis labels
    ax.set_xticks(range(n_threads))
    ax.set_xticklabels(pivot.columns.astype(int))
    ax.set_yticks(range(n_blocks))
    ax.set_yticklabels(pivot.index.astype(int))
    ax.set_xlabel("Threads per Block")
    ax.set_ylabel("Number of Blocks")
    ax.set_title(
        f"Grid-Stride SpMV — {ylabel} vs Launch Config\n"
        f"matrix: {matrix_name}",
        pad=10,
    )

    # Annotate each cell with its value
    text_threshold = (np.nanmax(data) + np.nanmin(data)) / 2
    for i in range(n_blocks):
        for j in range(n_threads):
            v = data[i, j]
            if np.isnan(v):
                continue
            colour = "white" if v > text_threshold else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color=colour, fontweight="bold")

    # Mark the best configuration with a border
    best_idx = np.unravel_index(np.nanargmax(data), data.shape)
    rect = plt.Rectangle(
        (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
        linewidth=2.5, edgecolor="blue", facecolor="none",
    )
    ax.add_patch(rect)
    best_blocks  = pivot.index[best_idx[0]]
    best_threads = pivot.columns[best_idx[1]]
    ax.set_title(
        f"Grid-Stride SpMV — {ylabel} vs Launch Config\n"
        f"matrix: {matrix_name}   "
        f"[best: {best_blocks} blocks × {best_threads} threads = "
        f"{data[best_idx]:.2f} {unit}]",
        pad=10,
    )

    fig.tight_layout()
    path = out_dir / fname
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")
    print(f"Best config: {best_blocks} blocks × {best_threads} threads/block "
          f"→ {data[best_idx]:.2f} {unit}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", help="CSV produced by parse_results.py")
    parser.add_argument("--out", default="assets",
                        help="output directory for PNG (default: assets/)")
    parser.add_argument("--metric", default="bandwidth_GBs",
                        choices=["bandwidth_GBs", "gflops"],
                        help="metric to plot (default: bandwidth_GBs)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[error] file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required = {"kernel", "matrix", args.metric}
    missing  = required - set(df.columns)
    if missing:
        print(f"[error] missing columns in CSV: {missing}", file=sys.stderr)
        sys.exit(1)

    # Derive a display name for the matrix from the first sweep row
    sweep_rows = df[[bool(RE_CONFIG.match(str(k))) for k in df["kernel"]]]
    if sweep_rows.empty:
        print("[error] no sweep rows found", file=sys.stderr)
        sys.exit(1)
    matrix_name = Path(sweep_rows.iloc[0]["matrix"]).stem

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pivot = build_pivot(df, args.metric)
    plot_heatmap(pivot, args.metric, matrix_name, out_dir)


if __name__ == "__main__":
    main()
