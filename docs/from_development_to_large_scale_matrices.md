# From Development Matrices to Large-Scale Matrices

**Purpose of this document:** this is a transition brief for a future LLM coding session.
It explains the current state of the dataset, why it is intentionally limited, and provides
a complete, actionable specification of every change needed to upgrade the project to
production-scale matrices. Read this before touching any script, kernel, or plot file.

---

## 1. Current State — What Exists and Why

### 1.1 The Three Development Matrices

`scripts/download_data.sh` downloads exactly three matrices from SuiteSparse:

| Name | Rows | NNZ | Shape | Role |
|------|------|-----|-------|------|
| `HB/1138_bus` | 1,138 | 4,054 | Unstructured (power network) | Smoke-test: tiny, instant |
| `HB/bcsstk17` | 10,974 | 428,650 | Structured (stiffness, symmetric) | Medium: catches symmetric expansion bugs |
| `SNAP/web-Google` | 916,428 | 5,105,039 | Unstructured (power-law web graph) | Large unstructured: stress-tests random x access |

**Why these three were chosen:** they were selected at the start of the project to get the
full pipeline (compile → run → parse → plot) working end-to-end as fast as possible. They
are small enough to run in seconds on the cluster's `edu-short` partition and in minutes
on a laptop. They cover three regimes: tiny (1138_bus), medium-structured (bcsstk17),
and large-unstructured (web-Google). This was deliberate scaffolding, not a final dataset.

### 1.2 Why This is a Problem for the Report

The report's central GPU argument requires two things side by side:
1. A **large structured/diagonal matrix** — shows the GPU at peak bandwidth (~100+ GB/s),
   because diagonal access patterns are cache-friendly, row lengths are uniform,
   and load balancing across threads is even.
2. A **large unstructured matrix** — shows the worst case (random x access, atomicAdd
   contention, load imbalance), where GPU still outperforms CPU but bandwidth is lower.

With only `web-Google` as the large matrix (and it is unstructured), we can only show
the worst case. The contrast that drives last year's report narrative — *"diagonal matrices
achieve ~120 GB/s while unstructured matrices stall at ~30 GB/s"* — cannot be made.
Additionally, our largest matrix (5M NNZ) is 7× smaller than last year's largest (38M NNZ),
meaning we never saturate the GPU memory bus on a structured input.

---

## 2. What Needs to Change — Full Refactoring Specification

### 2.1 `scripts/download_data.sh` — Add large matrices

Add at least two new matrices: one large diagonal/structured, one large unstructured.

**Recommended additions (verified on SuiteSparse at sparse.tamu.edu):**

| SuiteSparse path | Rows | NNZ | Shape | Why |
|-----------------|------|-----|-------|-----|
| `VLSI/vas_stokes_2M` or `Janna/StocF-1465` | ~1–2M | ~20M | Diagonal (FEM/stiffness) | Large structured — shows GPU peak BW |
| `LAW/indochina-2004` or `SNAP/LiveJournal1` | ~1–7M | ~50–70M | Unstructured (web/social graph) | Large unstructured — contrast case |

**Important:** verify the download URL pattern. Current script uses:
```bash
wget https://suitesparse-collection-website.herokuapp.com/MM/${GROUP}/${NAME}.tar.gz
```
Check this URL format is still valid at download time; SuiteSparse occasionally changes
its hosting URL. The group/name come from the matrix's page on sparse.tamu.edu.

**After adding matrices**, re-run `bash scripts/download_data.sh` and verify:
- Symmetric matrices are correctly expanded by `mtx_io.h` (check NNZ doubles)
- `pattern` matrices (no explicit values) parse correctly (value defaults to 1.0)

### 2.2 `scripts/batch_cpu.sh` — Exclude large matrices from CPU benchmarks

CPU sequential SpMV on a 50M-NNZ matrix can take 30–60 seconds per iteration × 50 iterations
= 25–50 minutes. The `edu-short` partition has a 30-minute wall-clock limit.

**Change:** add a size filter so batch_cpu.sh only runs on matrices below a threshold NNZ.
One clean approach: maintain a separate list of CPU-only matrix names in the script, or
check file size before running:

```bash
# Skip matrices larger than ~10M NNZ for CPU benchmarks
CPU_MATRICES=(1138_bus bcsstk17 web-Google)  # explicit allowlist
```

Alternatively, add a `Data/cpu_matrices.txt` file listing which matrices the CPU script
should use, and have `batch_gpu.sh` use all of them.

### 2.3 `scripts/batch_gpu.sh` — May need longer wall time

With multiple large matrices (50M+ NNZ), 3 kernels × 50 iterations each: estimate
~3–5 minutes per matrix. With 5 matrices: ~15–25 minutes. Increase `--time` to
`00:30:00` and verify against the partition's limit.

### 2.4 `scripts/sweep_gpu.sh` — Choose a large matrix for the sweep

Currently the sweep defaults to `bcsstk17` (428K NNZ). For the final report, the
block/thread sweep should be run on a large matrix (ideally a diagonal one with
10M+ NNZ) so that configuration differences are measurable. With tiny matrices, all
configurations are equally fast because kernel launch overhead dominates.

**Change:** update `SWEEP_MATRIX` variable in `sweep_gpu.sh` to point to the new
large structured matrix once downloaded.

### 2.5 `scripts/plot_results.py` — Handle more matrices on x-axis

With 5–6 matrices the bar chart x-axis labels may overlap. Two changes needed:

1. **Rotate labels:** change `rotation=0` to `rotation=30` or `45` in `grouped_bar()`.
2. **Figure width:** the formula `max(8, len(pivot) * 2.2)` may need the multiplier
   increased to `3.0` for wider bars when there are more matrices.

These are 2-line changes in `grouped_bar()` inside `plot_results.py`.

### 2.6 `scripts/plot_sweep.py` — Update sweep matrix label

The heatmap title currently says which matrix was used. Update the `SWEEP_MATRIX_LABEL`
constant at the top of `plot_sweep.py` to match the new large matrix name.

### 2.7 `report_insights.md` — Update sections 1.7 and 2.4

Both sections contain the dataset table. After adding new matrices:
- Add new rows to the table in **§1.7** (CPU dataset — only small matrices)
- Add new rows to the table in **§2.4** (GPU dataset — all matrices including large ones)
- Fill in the "Expected GPU behaviour" column for the new structured matrix

### 2.8 GPU Memory — No changes needed

The A30 GPU on the Baldo cluster has 24 GB of VRAM. A 50M-NNZ matrix in COO format
with double precision values requires:
```
50M × (4 + 4 + 8) bytes = 50M × 16 = 800 MB
```
Plus x (8 bytes × cols) and y (8 bytes × rows) vectors — well within 24 GB.
No changes to the kernel memory allocation code are needed.

---

## 3. What Does NOT Need to Change

- `GPU/spmv_gpu_tpv.cu`, `spmv_gpu_tpr.cu`, `spmv_gpu_stride.cu` — kernels are
  format/size agnostic. `mtx_io.h` handles arbitrarily large matrices.
- `scripts/parse_results.py` — format-agnostic, no changes needed.
- `makefile` — no changes needed.
- `include/mtx_io.h` — already handles large matrices; buffer is 512 bytes.
- `TIMER_LIB/` — CPU timing, unchanged.

---

## 4. Recommended Implementation Order for the Refactoring Session

1. Identify and verify 2 new matrices on sparse.tamu.edu (check NNZ, shape, download URL)
2. Update `scripts/download_data.sh` — add wget calls for new matrices
3. Update `scripts/batch_cpu.sh` — add matrix allowlist so CPU skips large ones
4. Update `scripts/batch_gpu.sh` — increase wall time
5. Update `scripts/sweep_gpu.sh` — change `SWEEP_MATRIX` to the new large structured one
6. Update `scripts/plot_results.py` — rotate labels, widen figure
7. Update `scripts/plot_sweep.py` — update matrix label constant
8. Update `report_insights.md` §1.7 and §2.4 tables
9. Run `bash scripts/download_data.sh` to fetch new matrices
10. Rerun all benchmarks (CPU on small, GPU on all, sweep on large structured)
11. Repopulate `report_insights.md` §2.5 and §2.6 with actual GPU results

---

## 5. Context on the Report Requirement

Last year's student (reference: `Deliverable_1_last_year_example/Deliverable1.md`) used:
- **6 CPU matrices** — all small (max 80K NNZ), to keep CPU runtimes manageable
- **6 GPU matrices** — all large (up to 38M NNZ), including both diagonal and unstructured

Their GPU results section showed:
- Diagonal matrices (`CurlCurl_4`, `af_shell8`): ~120 GB/s peak bandwidth
- Unstructured matrices (`mawi_201512012345`): ~30 GB/s
- The contrast between these two groups IS the main experimental result of the report

Without this contrast, the GPU section reduces to: "here are three numbers, the GPU is
faster than the CPU". That satisfies the minimum requirement but misses the analytical
depth that justifies a high report grade (report = 70% of the deliverable grade).
