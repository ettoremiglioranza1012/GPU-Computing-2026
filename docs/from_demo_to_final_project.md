# From Demo to Final Project — Gap Analysis & Implementation Plan

**Purpose:** Complete status assessment of the project against the professor's
`Deliverable1_SpMV_Guidance_v2` requirements, followed by a week-by-week
implementation schedule for the 5 weeks remaining before the May 1–7 deadline.

---

## 1. What Is Done — Solid Foundation

### 1.1 CPU Kernels (correct and complete)

- `spmv_coo_naive.c`: sequential COO SpMV, performance floor.
- `spmv_coo_opt.c`: 4-way unrolled with ILP + single-write-per-row.
- Correctness check: opt vs naive with adaptive `1e-9 * y_max + 1e-14` tolerance.
- CPU results measured and documented (`report_insights.md §1.8`).

### 1.2 GPU Kernels (3 implemented)

- `spmv_gpu_tpv.cu`: Thread-Per-Value with `atomicAdd`.
- `spmv_gpu_tpr.cu`: Thread-Per-Row with binary search into sorted COO.
- `spmv_gpu_stride.cu`: Grid-Stride Loop, grid size configurable via CLI.
- All kernels: explicit `cudaMalloc`/`cudaMemcpy`, `cudaEventRecord` timing,
  `__ldg()` on x, correctness verified against CPU naive.

### 1.3 Infrastructure (production-quality)

- Shared `mtx_io.h`: symmetric expansion, `pattern` type handling, inline row-sort,
  512-byte line buffer.
- SLURM launcher pattern: one independent job per (kernel, matrix) pair — scales
  to any number of matrices without exceeding the `edu-short` 5-minute wall limit.
- Block/thread configuration sweep: 35 configs (7 block counts × 5 thread counts);
  sweep CSVs already exist.
- `parse_results.py`, `plot_results.py`, `plot_sweep.py`: full analysis pipeline.
- Makefile with auto-discovery of new kernel files.
- Statistics: arithmetic mean + geometric mean, bandwidth via unique-column formula,
  GFLOPS reported.

### 1.4 Planning Documents

- `from_development_to_large_scale_matrices.md`: complete refactoring spec for
  dataset expansion, with verified matrix suggestions.
- `scheduling_strategy.md`: SLURM architecture and rationale documented.
- `report_insights.md`: structured template; CPU analysis fully written.

---

## 2. What Is Missing — Critical Gaps

### 2.1 Dataset: 3 matrices instead of 10 [Guidance §3 — HIGH]

The professor explicitly requires **10 matrices** from SuiteSparse, covering
diversity in: matrix dimensions, NNZ, average NNZ/row, row-length variability,
and structural regularity. The guidance says:

> "I suggest using the same matrices reported in [2]." (Chu et al. HPDC '23)

**Current state:** `1138_bus` (1K rows), `bcsstk17` (11K rows), `web-Google`
(916K rows). That is 3 matrices — 7 short of the requirement. `soc-LiveJournal1`
is mentioned in `download_data.sh` comments but is **not in the `MATRICES` array**
and has never been downloaded.

The `from_development_to_large_scale_matrices.md` doc addresses this partially
(proposes adding 2 large matrices) but the dataset still only reaches 5, not 10.

### 2.2 Data type is `double`, professor requires `float32` [Guidance §3 — HIGH]

> "Use Float32 as a data type."

Every file — `mtx_io.h`, both CPU kernels, all 3 GPU kernels — uses `double`
(64-bit). Bandwidth formulas, `cudaMalloc` sizes, and all timing are calibrated
to `double`. This is a fundamental mismatch with the primary data-type requirement.

### 2.3 No shared memory kernel [Guidance §1 and §4 — HIGH]

> "the other could use techniques to reduce the cost of accessing the global memory
> **(using shared memory is fundamental)**"

Zero kernels use shared memory. The Warp-Per-Row kernel with `__shfl_xor_sync`
is documented in `report_insights.md §2.8` as deferred future work — but the
professor calls shared memory **fundamental**, not optional. This is likely
the kernel that separates passing from high-scoring submissions.

Pseudocode for reference (from `report_insights.md §2.8`):

```cuda
int row  = blockIdx.x;
int lane = threadIdx.x % 32;
double partial = 0.0;
for (int i = row_start + lane; i < row_end; i += 32)
    partial += Avals[i] * __ldg(&x[Acols[i]]);
for (int offset = 16; offset > 0; offset >>= 1)
    partial += __shfl_xor_sync(0xffffffff, partial, offset);
if (lane == 0) y[row] = partial;
```

### 2.4 No cuSPARSE comparison [Guidance §1 — MEDIUM]

> "Compare with CuSparse implementation (does it matter if your implementation
> is slower)"

Not implemented. cuSPARSE is available on Baldo; adding a wrapper binary is
approximately 30 lines of CUDA C.

### 2.5 x vector is all-ones, not random [Guidance §3 — MEDIUM]

> "use a randomly generated dense vector of compatible size. You may keep the
> random seed fixed for reproducibility."

All kernels initialize `x[i] = 1.0`. A fixed-seed random vector (`srand(42)`,
`x[i] = (float)rand() / RAND_MAX`) is required.

### 2.6 GPU results section is empty [Guidance §5 — BLOCKING FOR REPORT]

`report_insights.md §2.5` and `§2.6` are placeholder dashes. No GPU benchmark
results have been collected and parsed. The core experimental claim of the
deliverable has no data.

### 2.7 Dataset too small to support the required analytical contrast [Guidance §6]

The professor wants the report to explain **why** one kernel outperforms another
across matrix types. The required contrast (structured FEM matrix vs power-law
graph) cannot be made without a large structured matrix. `web-Google` is
unstructured; `bcsstk17` (428K NNZ) is too small to saturate GPU bandwidth.
Without this contrast the GPU section reduces to "here are three numbers".

### 2.8 No report written [Guidance §7 — FINAL DELIVERABLE]

| Section | Status |
|---------|--------|
| Introduction | Not written |
| Methodology | Not written (pseudocode required) |
| Dataset | Not written (separate section required by guidance) |
| Results | No GPU results to insert |
| Discussion | Not written (this is where the grade is won) |
| Conclusion | Not written |
| References | Not written |

Note: the guidance adds a **Dataset** section and a **Discussion** section that
the original assignment template does not explicitly name. Both are required.

---

## 3. What Is Missing — Lower-Priority Gaps

### 3.1 Required references not yet cited [Guidance §8]

The professor mandates citing both:

1. Gao et al. 2024 — systematic literature survey (required reference [1])
2. Chu et al. HPDC '23 — optimizing SpMV on GPU (required reference [2])

`paper_insights.md` is based on Yang et al. VLDB 2011, which appears as optional
reference [3] in the guidance. The two **required** references are different papers
and are not yet incorporated into our docs.

### 3.2 Variability not reported [Guidance §5]

> "report either the average together with variability, or a justified stable
> statistic such as median/best-of-N"

We report arithmetic + geometric mean. No standard deviation or percentile spread
is printed. This needs either adding std dev to the output format or explicitly
justifying geometric mean as the stable statistic in the report.

### 3.3 Kernel timing not explicitly separated from setup costs [Guidance §5]

> "clearly distinguish [kernel time] from one-time setup costs such as file
> parsing or format conversion"

Our timing is kernel-only (correct). However, if a format-conversion step is added
(e.g. COO→CSR for the warp-per-row kernel), that conversion time must be reported
separately. Currently not addressed.

---

## 4. Implementation Steps — 5-Week Schedule

**Start date:** 2026-04-02  
**Deadline:** 2026-05-01 (hard) to 2026-05-07 (extended window)

---

### Week 1 (Apr 2–8) — Data Foundation

**Goal:** fix the two most fundamental data-layer mismatches before any new
benchmarking is run on incorrect data.

| Step | Task | Files touched |
|------|------|---------------|
| 1.1 | Switch all floating-point types from `double` to `float` in `mtx_io.h`, both CPU kernels, all 3 GPU kernels, bandwidth formulas, and `cudaMalloc` sizes | `include/mtx_io.h`, `CPU/*.c`, `GPU/*.cu` |
| 1.2 | Replace `x[i] = 1.0` with a fixed-seed random vector (`srand(42)`, `x[i] = (float)rand() / RAND_MAX`) in all 5 kernels | `CPU/*.c`, `GPU/*.cu` |
| 1.3 | Select 10 matrices from SuiteSparse (see criteria in §2.1 above). Use Chu et al. HPDC '23 matrix set as guidance. Verify download URLs at sparse.tamu.edu | — |
| 1.4 | Update `scripts/download_data.sh` with the 7 new matrices | `scripts/download_data.sh` |
| 1.5 | Update `scripts/batch_cpu.sh` with a CPU-only allowlist (matrices below ~10M NNZ) | `scripts/batch_cpu.sh` |
| 1.6 | Download all 10 matrices on the cluster (`make data`) and verify symmetric expansion and `pattern` handling | cluster |

**Checkpoint:** 10 matrices downloaded, all kernels using `float` and random x.

---

### Week 2 (Apr 9–15) — GPU Benchmarks on Full Dataset

**Goal:** collect the GPU numbers that are currently missing from `report_insights.md`.

| Step | Task | Files touched |
|------|------|---------------|
| 2.1 | Recompile all kernels after float/vector change (`make all`) | cluster |
| 2.2 | Re-run the block/thread sweep on a large structured matrix (replace `bcsstk17` as the sweep target) | `scripts/sweep_gpu.sh` |
| 2.3 | Parse sweep results and write `assets/best_gpu_config.sh` | `scripts/plot_sweep.py`, `assets/` |
| 2.4 | Run CPU benchmarks on the small-matrix allowlist (`sbatch scripts/batch_cpu.sh`) | cluster |
| 2.5 | Run GPU benchmarks across all 10 matrices and 3 kernels (`bash scripts/batch_gpu.sh`) | cluster |
| 2.6 | Parse all results into CSVs; generate all plots | `scripts/parse_results.py`, `scripts/plot_results.py` |
| 2.7 | Populate `report_insights.md §2.5` (GPU results table) and `§2.6` (key observations) | `docs/report_insights.md` |
| 2.8 | Update `report_insights.md §1.7` and `§2.4` dataset tables with all 10 matrices | `docs/report_insights.md` |

**Checkpoint:** GPU results filled in; structured vs unstructured bandwidth contrast visible in plots.

---

### Week 3 (Apr 16–22) — Advanced Kernels

**Goal:** add the shared memory kernel and the cuSPARSE baseline that the professor
marks as fundamental and recommended respectively.

| Step | Task | Files touched |
|------|------|---------------|
| 3.1 | Implement `GPU/spmv_gpu_warp.cu`: warp-per-row with `__shfl_xor_sync`. Uses COO with a CSR-style row-pointer array built at load time, or binary search per warp. Inherit improvements A/B/C from existing kernels | `GPU/spmv_gpu_warp.cu` |
| 3.2 | Add cuSPARSE baseline: `GPU/spmv_cusparse.cu`. Wraps `cusparseSpMV` with CSR format built from COO at load time. Report conversion time separately from kernel time | `GPU/spmv_cusparse.cu` |
| 3.3 | Recompile (`make gpu`) — Makefile auto-discovers new `.cu` files | cluster |
| 3.4 | Run GPU benchmarks for the 2 new kernels across all 10 matrices | cluster |
| 3.5 | Update plots to include warp and cuSPARSE series | `scripts/plot_results.py` |
| 3.6 | Add variability metric (std dev or min/max) to output format and `parse_results.py` | `GPU/*.cu`, `CPU/*.c`, `scripts/parse_results.py` |
| 3.7 | Update `report_insights.md §2.5` and `§2.6` with new kernel results | `docs/report_insights.md` |

**Checkpoint:** 5 GPU kernels benchmarked (TPV, TPR, stride, warp, cuSPARSE) across 10 matrices.

---

### Week 4 (Apr 23–29) — Report Writing

**Goal:** produce the complete 4-page report using `report_insights.md` as the
single source of truth. Follow the LaTeX template in `Report_Template/`.

| Step | Task |
|------|------|
| 4.1 | **Introduction:** problem statement, why SpMV matters, the investigation question |
| 4.2 | **Methodology:** storage format (COO + row-pointer for warp kernel), all 5 GPU kernels with pseudocode, CPU baseline, validation method, measurement protocol (50 iterations, 2 warmup, float32, fixed-seed x), hardware/software environment |
| 4.3 | **Dataset:** summary table of all 10 matrices (rows, cols, NNZ, avg NNZ/row, domain, one-line structural description); justify selection — diverse regimes, matches Chu et al. set |
| 4.4 | **Results:** bandwidth plots and GFLOPS plots across all kernels and matrices; optional cache-miss indicator if profiling data is available |
| 4.5 | **Discussion:** explain *why* each kernel wins or loses on each matrix subgroup. Address: load imbalance (TPR on power-law), atomicAdd contention (TPV on dense rows), warp efficiency (warp-per-row on structured matrices), memory-bound evidence, comparison vs cuSPARSE |
| 4.6 | **Conclusion:** main lessons learned, limitations (COO overhead, float32-only), practical takeaways |
| 4.7 | **References:** add Gao 2024, Chu HPDC'23 as required [1] and [2]; add Bell & Garland SC'09, Greathouse & Daga SC'14, and Yang et al. as supporting references |

**Checkpoint:** draft report complete, LaTeX compiles, under 4 pages.

---

### Week 5 (Apr 30 – May 7) — Polish, Verification, Submission

**Goal:** catch any remaining issues before submission.

| Step | Task |
|------|------|
| 5.1 | Final re-run of all benchmarks if any kernel or data change was made in Week 4 review |
| 5.2 | Verify all correctness checks pass on every (kernel, matrix) pair |
| 5.3 | Check report: correct student name, student ID, email, git repository link |
| 5.4 | Check all plots are readable in PDF (font size, legend, axis labels) |
| 5.5 | Push final code and results to git; confirm repo is accessible |
| 5.6 | Export report to PDF; verify page count ≤ 4 |
| 5.7 | Submit before May 7 at 23:59 |

---

## 5. Priority Summary

| Priority | Action | Week |
|----------|--------|------|
| 1 | Switch `double` → `float` everywhere | 1 |
| 2 | Replace `x = 1.0` with fixed-seed random vector | 1 |
| 3 | Expand dataset from 3 to 10 SuiteSparse matrices | 1 |
| 4 | Run GPU benchmarks on full dataset and fill in empty results | 2 |
| 5 | Implement shared memory kernel (warp-per-row + warp shuffle) | 3 |
| 6 | Add cuSPARSE baseline | 3 |
| 7 | Write the report — Discussion section is where the grade is won | 4 |
| 8 | Add required references (Gao 2024, Chu HPDC'23) | 4 |
| 9 | Add variability metric to output and plots | 3 |
