#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.9",
#     "pandas>=2.2",
# ]
# ///
"""plot_results.py — generate CPU and GPU benchmark plots from parsed CSV.

Usage:
    uv run scripts/plot_results.py results.csv
    uv run scripts/plot_results.py results.csv --out assets/

Produces (in --out directory):
    cpu_bandwidth.png   — grouped bar: bandwidth (GB/s) per CPU kernel × matrix
    cpu_gflops.png      — grouped bar: GFLOPS per CPU kernel × matrix
    cpu_speedup.png     — speedup of opt over naive per matrix (CPU only)
    gpu_bandwidth.png   — grouped bar: bandwidth (GB/s) per GPU kernel × matrix
    gpu_gflops.png      — grouped bar: GFLOPS per GPU kernel × matrix
    gpu_vs_cpu.png      — speedup of each GPU kernel over CPU naive per matrix
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── constants ─────────────────────────────────────────────────────────────────

CPU_KERNELS = ["spmv_coo_naive", "spmv_coo_opt"]
GPU_KERNELS = ["spmv_gpu_tpv", "spmv_gpu_tpr", "spmv_gpu_stride"]

KERNEL_LABELS = {
    "spmv_coo_naive":   "Naive COO",
    "spmv_coo_opt":     "Optimised COO (4× unroll)",
    "spmv_gpu_tpv":     "GPU Thread-Per-Value",
    "spmv_gpu_tpr":     "GPU Thread-Per-Row",
    "spmv_gpu_stride":  "GPU Grid-Stride",
}
COLORS = {
    "spmv_coo_naive":  "#4878d0",
    "spmv_coo_opt":    "#ee854a",
    "spmv_gpu_tpv":    "#6acc65",
    "spmv_gpu_tpr":    "#d65f5f",
    "spmv_gpu_stride": "#b47cc7",
}

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "axes.axisbelow": True,
})


# ── helpers ───────────────────────────────────────────────────────────────────
def short_name(matrix: str) -> str:
    return Path(matrix).stem


def value_label(ax, bar, fmt="{:.2f}", rotation=0):
    h = bar.get_height()
    if h > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + ax.get_ylim()[1] * 0.01,
            fmt.format(h),
            ha="center", va="bottom", fontsize=9, rotation=rotation,
        )


def grouped_bar(ax, df_pivot, ylabel, title):
    n_matrices = len(df_pivot)
    n_kernels  = len(df_pivot.columns)
    width      = 0.8 / n_kernels
    x          = list(range(n_matrices))

    for i, kernel in enumerate(df_pivot.columns):
        offset = (i - (n_kernels - 1) / 2) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            df_pivot[kernel],
            width=width * 0.9,
            label=KERNEL_LABELS.get(kernel, kernel),
            color=COLORS.get(kernel, f"C{i}"),
            edgecolor="white",
            linewidth=0.5,
        )
        for bar in bars:
            value_label(ax, bar)

    ax.set_xticks(x)
    ax.set_xticklabels(df_pivot.index, rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)


# ── CPU plots ─────────────────────────────────────────────────────────────────
def plot_bandwidth(df: pd.DataFrame, out_dir: Path):
    pivot = (
        df.pivot_table(index="matrix", columns="kernel", values="bandwidth_GBs")
          .rename(index=short_name)
    )
    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 2.2), 6))
    grouped_bar(ax, pivot, "Effective Bandwidth (GB/s)",
                "CPU SpMV — Effective Bandwidth")
    fig.tight_layout()
    path = out_dir / "cpu_bandwidth.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


def plot_gflops(df: pd.DataFrame, out_dir: Path):
    pivot = (
        df.pivot_table(index="matrix", columns="kernel", values="gflops")
          .rename(index=short_name)
    )
    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 2.2), 6))
    grouped_bar(ax, pivot, "Performance (GFLOP/s)",
                "CPU SpMV — Compute Throughput")
    fig.tight_layout()
    path = out_dir / "cpu_gflops.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


def plot_speedup(df: pd.DataFrame, out_dir: Path):
    naive = (
        df[df["kernel"] == "spmv_coo_naive"]
        .set_index("matrix")["bandwidth_GBs"]
        .rename("naive_bw")
    )
    opt = (
        df[df["kernel"] == "spmv_coo_opt"]
        .set_index("matrix")["bandwidth_GBs"]
        .rename("opt_bw")
    )
    merged = pd.concat([naive, opt], axis=1).dropna()
    sp = (merged["opt_bw"] / merged["naive_bw"]).rename("speedup")
    sp.index = sp.index.map(short_name)

    fig, ax = plt.subplots(figsize=(max(7, len(sp) * 2.0), 6))
    bars = ax.bar(
        sp.index, sp.values,
        color=COLORS["spmv_coo_opt"],
        edgecolor="white", linewidth=0.5,
    )
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
               label="baseline (naive = 1×)")
    ax.set_ylim(0, max(sp.values) * 1.2)
    for bar in bars:
        value_label(ax, bar, fmt="{:.2f}×", rotation=0)

    ax.set_ylabel("Speedup  $S_p = T_{naive} / T_{opt}$")
    ax.set_title("CPU SpMV — Speedup of 4× Unrolling over Naive")
    ax.set_xticks(range(len(sp)))
    ax.set_xticklabels(sp.index, rotation=0, ha="center")
    ax.legend()
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    path = out_dir / "cpu_speedup.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


# ── GPU plots ─────────────────────────────────────────────────────────────────
def plot_gpu_bandwidth(df: pd.DataFrame, out_dir: Path):
    pivot = (
        df.pivot_table(index="matrix", columns="kernel", values="bandwidth_GBs")
          .rename(index=short_name)
    )
    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 2.2), 6))
    grouped_bar(ax, pivot, "Effective Bandwidth (GB/s)",
                "GPU SpMV — Effective Bandwidth")
    fig.tight_layout()
    path = out_dir / "gpu_bandwidth.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


def plot_gpu_gflops(df: pd.DataFrame, out_dir: Path):
    pivot = (
        df.pivot_table(index="matrix", columns="kernel", values="gflops")
          .rename(index=short_name)
    )
    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 2.2), 6))
    grouped_bar(ax, pivot, "Performance (GFLOP/s)",
                "GPU SpMV — Compute Throughput")
    fig.tight_layout()
    path = out_dir / "gpu_gflops.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


def plot_gpu_vs_cpu(df: pd.DataFrame, out_dir: Path):
    """Speedup of each GPU kernel over CPU naive, one grouped bar per matrix."""
    cpu_naive = (
        df[df["kernel"] == "spmv_coo_naive"]
        .set_index("matrix")["bandwidth_GBs"]
        .rename("cpu_naive_bw")
    )
    if cpu_naive.empty:
        print("[warn] no spmv_coo_naive rows found — skipping gpu_vs_cpu plot",
              file=sys.stderr)
        return

    rows = []
    for gpu_k in GPU_KERNELS:
        gpu_bw = (
            df[df["kernel"] == gpu_k]
            .set_index("matrix")["bandwidth_GBs"]
        )
        merged = pd.concat([cpu_naive, gpu_bw.rename("gpu_bw")], axis=1).dropna()
        for mtx, row in merged.iterrows():
            rows.append({
                "matrix":  short_name(mtx),
                "kernel":  gpu_k,
                "speedup": row["gpu_bw"] / row["cpu_naive_bw"],
            })

    if not rows:
        print("[warn] no GPU/CPU overlap found — skipping gpu_vs_cpu plot",
              file=sys.stderr)
        return

    sp_df  = pd.DataFrame(rows)
    pivot  = sp_df.pivot_table(index="matrix", columns="kernel", values="speedup")
    # preserve GPU_KERNELS column order
    pivot  = pivot.reindex(columns=[k for k in GPU_KERNELS if k in pivot.columns])

    n_matrices = len(pivot)
    n_kernels  = len(pivot.columns)
    width      = 0.8 / n_kernels
    x          = list(range(n_matrices))

    fig, ax = plt.subplots(figsize=(max(8, n_matrices * 2.5), 6))
    for i, kernel in enumerate(pivot.columns):
        offset = (i - (n_kernels - 1) / 2) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            pivot[kernel],
            width=width * 0.9,
            label=KERNEL_LABELS.get(kernel, kernel),
            color=COLORS.get(kernel, f"C{i}"),
            edgecolor="white",
            linewidth=0.5,
        )
        for bar in bars:
            value_label(ax, bar, fmt="{:.1f}×")

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
               label="CPU naive baseline (1×)")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=0, ha="center")
    ax.set_ylabel("Speedup over CPU naive  (bandwidth ratio)")
    ax.set_title("GPU vs CPU Naive — Bandwidth Speedup")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.legend()
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    path = out_dir / "gpu_vs_cpu.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", help="CSV produced by parse_results.py")
    parser.add_argument("--out", default="assets",
                        help="output directory for PNG files (default: assets/)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[error] file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    required = {"kernel", "matrix", "bandwidth_GBs", "gflops"}
    missing = required - set(df.columns)
    if missing:
        print(f"[error] missing columns in CSV: {missing}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu_df = df[df["kernel"].isin(CPU_KERNELS)]
    gpu_df = df[df["kernel"].isin(GPU_KERNELS)]

    # CPU plots (only when CPU data present)
    if not cpu_df.empty:
        plot_bandwidth(cpu_df, out_dir)
        plot_gflops(cpu_df, out_dir)
        if "spmv_coo_naive" in cpu_df["kernel"].values and \
           "spmv_coo_opt"   in cpu_df["kernel"].values:
            plot_speedup(cpu_df, out_dir)
    else:
        print("[info] no CPU kernel rows found — skipping CPU plots", file=sys.stderr)

    # GPU plots (only when GPU data present)
    if not gpu_df.empty:
        plot_gpu_bandwidth(gpu_df, out_dir)
        plot_gpu_gflops(gpu_df, out_dir)
        plot_gpu_vs_cpu(df, out_dir)   # needs both CPU naive and GPU rows
    else:
        print("[info] no GPU kernel rows found — skipping GPU plots", file=sys.stderr)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
