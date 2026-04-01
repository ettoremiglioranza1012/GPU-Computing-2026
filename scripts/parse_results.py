#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""parse_results.py — extract benchmark results from batch_cpu/gpu output files.

Usage:
    uv run scripts/parse_results.py outputs/R-spmv_cpu.473927.txt
    uv run scripts/parse_results.py outputs/*.txt          # multiple files
    uv run scripts/parse_results.py outputs/*.txt --out results_tables/results.csv

Output: CSV with columns:
    kernel, matrix, rows, cols, nnz, arith_mean_s, geom_mean_s, bandwidth_GBs, gflops
"""

import re
import sys
import csv
import os
from pathlib import Path

# ── regex patterns matching the C printf format strings ──────────────────────
RE_BANNER  = re.compile(r"^KERNEL:\s+(\S+)\s+MATRIX:\s+(\S+)")
RE_MATRIX  = re.compile(r"^Matrix:\s+(\S+)")
RE_DIMS    = re.compile(r"^Rows:\s+(\d+)\s+Cols:\s+(\d+)\s+NNZ:\s+(\d+)")
RE_STATS   = re.compile(r"^\s*(\S+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|")
RE_BW      = re.compile(r"^Effective bandwidth:\s+([\d.]+)\s+GB/s")
RE_GFLOPS  = re.compile(r"^GFLOPS:\s+([\d.]+)")
RE_CORRECT = re.compile(r"^Correctness check:\s+(\w+)")


def parse_file(path: Path) -> list[dict]:
    records = []
    current = {}

    with open(path) as f:
        for raw in f:
            line = raw.strip()

            m = RE_BANNER.match(line)
            if m:
                current = {"kernel": m.group(1), "matrix": m.group(2)}
                continue

            m = RE_MATRIX.match(line)
            if m and current:
                current["matrix_path"] = m.group(1)
                continue

            m = RE_DIMS.match(line)
            if m and current:
                current["rows"] = int(m.group(1))
                current["cols"] = int(m.group(2))
                current["nnz"]  = int(m.group(3))
                continue

            m = RE_STATS.match(line)
            if m and current and m.group(1) not in ("kernel",):
                current["arith_mean_s"] = float(m.group(2))
                current["geom_mean_s"]  = float(m.group(3))
                continue

            m = RE_BW.match(line)
            if m and current:
                current["bandwidth_GBs"] = float(m.group(1))
                continue

            m = RE_GFLOPS.match(line)
            if m and current:
                current["gflops"] = float(m.group(1))
                # GFLOPS is always the last field printed — record is complete
                if "arith_mean_s" in current:
                    records.append(current)
                current = {}
                continue

            m = RE_CORRECT.match(line)
            if m and current:
                current["correctness"] = m.group(1)
                continue

    return records


FIELDS = [
    "kernel", "matrix", "rows", "cols", "nnz",
    "arith_mean_s", "geom_mean_s", "bandwidth_GBs", "gflops", "correctness",
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help=".out files to parse")
    parser.add_argument("--out", default="results_tables/results.csv",
                        help="output CSV path (default: results_tables/results.csv)")
    args = parser.parse_args()

    all_records = []
    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"[warn] not found: {f}", file=sys.stderr)
            continue
        records = parse_file(path)
        if not records:
            print(f"[warn] no results parsed from {f}", file=sys.stderr)
        all_records.extend(records)

    if not all_records:
        print("[error] no records found", file=sys.stderr)
        sys.exit(1)

    if args.out != "-":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dest = open(args.out, "w", newline="") if args.out != "-" else sys.stdout
    writer = csv.DictWriter(dest, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_records)
    if args.out != "-":
        dest.close()
        print(f"Wrote {len(all_records)} records to {args.out}")
    else:
        print(f"\n[{len(all_records)} records]", file=sys.stderr)


if __name__ == "__main__":
    main()
