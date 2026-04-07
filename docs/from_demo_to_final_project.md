# From Demo to Final Project — Gap Analysis & Implementation Plan

**Purpose:** Complete status assessment of the project against the professor's
`Deliverable1_SpMV_Guidance_v2` requirements, followed by a week-by-week
implementation schedule for the 5 weeks remaining before the May 1–7 deadline.

---

## 0. Current Status Snapshot

> **Last verified: 2026-04-07**
> **Active branch: `dev`**

| Week | Goal | Status |
|------|------|--------|
| Week 1 (Apr 2–8) | Data foundation | ✅ Complete |
| Week 2 (Apr 9–15) | GPU benchmarks on full dataset | ✅ Complete |
| Week 3 (Apr 16–22) | Advanced kernels (warp + cuSPARSE) | ❌ Not started |
| Week 4 (Apr 23–29) | Report writing | ❌ Not started |
| Week 5 (Apr 30–May 7) | Polish & submission | ❌ Not started |

### How to verify status at the start of a session

Run these checks before claiming anything is done or not done:

```bash
# 1. Float/random-x: confirm no `double` in kernel files (timer vars are ok)
grep -n "double" GPU-computing-2026/include/mtx_io.h GPU-computing-2026/CPU/*.c GPU-computing-2026/GPU/*.cu

# 2. Dataset: count matrices in download_data.sh
grep -c '"' GPU-computing-2026/scripts/download_data.sh   # should be ≥10

# 3. Benchmark results: check results.csv exists and has GPU rows
wc -l GPU-computing-2026/results_tables/results.csv       # >1 means data present

# 4. Advanced kernels: check if warp and cuSPARSE exist
ls GPU-computing-2026/GPU/spmv_gpu_warp.cu GPU-computing-2026/GPU/spmv_cusparse.cu 2>/dev/null || echo "not yet"

# 5. Report: check if LaTeX source exists
ls GPU-computing-2026/report/*.tex 2>/dev/null || echo "not yet"

# 6. report_insights.md GPU sections: grep for placeholder dashes
grep "^| \`" GPU-computing-2026/docs/report_insights.md | grep -E "\-\s*\|"
```

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

## 2. Critical Gaps — Status

### 2.1 ✅ Dataset: 10 matrices from SuiteSparse matching Chu et al. HPDC '23
Resolved in Week 1. `scripts/download_data.sh` has all 10 matrices. Results confirmed in `results_tables/results.csv`.

### 2.2 ✅ Data type: float32 throughout
Resolved in Week 1. All kernels and `mtx_io.h` use `float`. Verified by grepping source files.

### 2.3 ❌ No shared memory kernel [Guidance §1 and §4 — HIGH]

> "the other could use techniques to reduce the cost of accessing the global memory
> **(using shared memory is fundamental)**"

Zero kernels use shared memory. The Warp-Per-Row kernel with `__shfl_xor_sync`
is the target for Week 3. The professor calls shared memory **fundamental** — this
is the kernel that separates passing from high-scoring submissions.

Pseudocode for reference (from `report_insights.md §2.8`):

```cuda
int row  = blockIdx.x;
int lane = threadIdx.x % 32;
float partial = 0.0f;
for (int i = row_start + lane; i < row_end; i += 32)
    partial += Avals[i] * __ldg(&x[Acols[i]]);
for (int offset = 16; offset > 0; offset >>= 1)
    partial += __shfl_xor_sync(0xffffffff, partial, offset);
if (lane == 0) y[row] = partial;
```

### 2.4 ❌ No cuSPARSE comparison [Guidance §1 — MEDIUM]

> "Compare with CuSparse implementation (does it matter if your implementation
> is slower)"

Not implemented. Target for Week 3. cuSPARSE is available on Baldo; wrapper is ~30 lines of CUDA C.

### 2.5 ✅ x vector: fixed-seed random
Resolved in Week 1. All kernels use `srand(42); x[i] = (float)rand() / RAND_MAX`.

### 2.6 ✅ GPU results filled in
Resolved in Week 2. `report_insights.md §2.5` and `§2.6` updated with full 10-matrix results.

### 2.7 ✅ Structured vs unstructured contrast now visible
Resolved in Week 2 by expanding to 10 matrices. FEM matrices (`bone010`, `ldoor`) vs power-law graphs (`hollywood-2009`, `webbase-1M`) show clear kernel behaviour differences in `report_insights.md §2.6`.

### 2.8 ❌ No report written [Guidance §7 — FINAL DELIVERABLE]

| Section | Status |
|---------|--------|
| Introduction | Not written |
| Methodology | Not written (pseudocode required) |
| Dataset | Not written (separate section required by guidance) |
| Results | No GPU results to insert yet (waiting for warp + cuSPARSE from Week 3) |
| Discussion | Not written (this is where the grade is won) |
| Conclusion | Not written |
| References | Not written |

Target for Week 4.

---

## 3. Lower-Priority Gaps — Status

### 3.1 ❌ Required references not yet cited [Guidance §8]

The professor mandates citing both:

1. Gao et al. 2024 — systematic literature survey (required reference [1])
2. Chu et al. HPDC '23 — optimizing SpMV on GPU (required reference [2])

Target for Week 4 (report writing).

### 3.2 ❌ Variability not reported [Guidance §5]

> "report either the average together with variability, or a justified stable
> statistic such as median/best-of-N"

We report arithmetic + geometric mean. Need to either add std dev to output or
explicitly justify geometric mean as a stable statistic in the report. Target for Week 3.

### 3.3 ❌ COO→CSR conversion time not reported separately [Guidance §5]

> "clearly distinguish [kernel time] from one-time setup costs such as file
> parsing or format conversion"

Kernel-only timing is already correct. When the warp kernel (Week 3) builds a
row-pointer array from COO at load time, that conversion time must be printed
separately. Must be addressed in Week 3 alongside step 3.1.

---

## 4. Implementation Steps — 5-Week Schedule

**Start date:** 2026-04-02  
**Deadline:** 2026-05-01 (hard) to 2026-05-07 (extended window)

---

### Week 1 (Apr 2–8) — Data Foundation ✅ COMPLETE

**Goal:** fix the two most fundamental data-layer mismatches before any new
benchmarking is run on incorrect data.

| Step | Status | Task | Files touched |
|------|--------|------|---------------|
| 1.1 | ✅ | Switch all floating-point types from `double` to `float` in `mtx_io.h`, both CPU kernels, all 3 GPU kernels, bandwidth formulas, and `cudaMalloc` sizes | `include/mtx_io.h`, `CPU/*.c`, `GPU/*.cu` |
| 1.2 | ✅ | Replace `x[i] = 1.0` with a fixed-seed random vector (`srand(42)`, `x[i] = (float)rand() / RAND_MAX`) in all 5 kernels | `CPU/*.c`, `GPU/*.cu` |
| 1.3 | ✅ | Select 10 matrices from SuiteSparse. Use Chu et al. HPDC '23 matrix set as guidance. Verify download URLs at sparse.tamu.edu | — |
| 1.4 | ✅ | Update `scripts/download_data.sh` with the 7 new matrices | `scripts/download_data.sh` |
| 1.5 | ✅ | Update `scripts/batch_cpu.sh` with a CPU-only allowlist (matrices below ~10M NNZ) | `scripts/batch_cpu.sh` |
| 1.6 | ✅ | Download all 10 matrices on the cluster (`make data`) and verify symmetric expansion and `pattern` handling — confirmed via results in `results_tables/results.csv` covering all 10 matrices | cluster |

**Checkpoint:** 10 matrices downloaded, all kernels using `float` and random x. ✅ PASSED

---

### Week 2 (Apr 9–15) — GPU Benchmarks on Full Dataset ✅ COMPLETE

**Goal:** collect the GPU numbers that are currently missing from `report_insights.md`.

| Step | Status | Task | Files touched |
|------|--------|------|---------------|
| 2.1 | ✅ | Recompile all kernels after float/vector change (`make all`) | cluster |
| 2.2 | ✅ | Re-run the block/thread sweep on a large structured matrix — `assets/gpu_config_sweep.png` exists | `scripts/sweep_gpu.sh` |
| 2.3 | ✅ | Parse sweep results and write best GPU config — `scripts/best_gpu_config.sh` exists (note: lives in `scripts/`, not `assets/`) | `scripts/plot_sweep.py`, `scripts/best_gpu_config.sh` |
| 2.4 | ✅ | Run CPU benchmarks on the small-matrix allowlist — 4 matrices × 2 kernels in `results.csv` | cluster |
| 2.5 | ✅ | Run GPU benchmarks across all 10 matrices and 3 kernels — 30 rows in `results.csv` | cluster |
| 2.6 | ✅ | Parse all results into CSVs; generate all plots — `results.csv` + 7 PNGs in `assets/` | `scripts/parse_results.py`, `scripts/plot_results.py` |
| 2.7 | ✅ | Populate `report_insights.md §2.5` and `§2.6` with 10-matrix results | `docs/report_insights.md` |
| 2.8 | ✅ | Update `report_insights.md §2.4` dataset table with all 10 matrices | `docs/report_insights.md` |

**Checkpoint:** GPU results filled in; structured vs unstructured bandwidth contrast visible in plots. ✅ PASSED

---

### Week 3 (Apr 16–22) — Advanced Kernels ❌ NOT STARTED

**Goal:** add the shared memory kernel and the cuSPARSE baseline that the professor
marks as fundamental and recommended respectively.

| Step | Status | Task | Files touched |
|------|--------|------|---------------|
| 3.1 | ✅ | Implement `GPU/spmv_gpu_warp.cu`: warp-per-row with `__shfl_xor_sync`. COO data + host-built `row_ptr`. Improvements A/B/C/D. Setup time printed separately. | `GPU/spmv_gpu_warp.cu` |
| 3.2 | ✅ | Add cuSPARSE baseline: `GPU/spmv_cusparse.cu`. Wraps `cusparseSpMV` (CSR built from COO at load time). Makefile updated with `-lcusparse` specific rule. Setup time printed separately. | `GPU/spmv_cusparse.cu`, `makefile` |
| 3.3 | ❌ | Recompile (`make gpu`) and add both kernels to `batch_gpu.sh` — done for the script; compilation must happen on cluster | `scripts/batch_gpu.sh` updated ✅ |
| 3.4 | ❌ | Run GPU benchmarks for the 2 new kernels across all 10 matrices | cluster |
| 3.5 | ❌ | Update plots to include warp and cuSPARSE series | `scripts/plot_results.py` |
| 3.6 | ❌ | Add variability metric (std dev or min/max) to output format and `parse_results.py` | `GPU/*.cu`, `CPU/*.c`, `scripts/parse_results.py` |
| 3.7 | ❌ | Update `report_insights.md §2.5` and `§2.6` with new kernel results | `docs/report_insights.md` |

**Checkpoint:** 5 GPU kernels benchmarked (TPV, TPR, stride, warp, cuSPARSE) across 10 matrices.

---

### Week 4 (Apr 23–29) — Report Writing ❌ NOT STARTED

**Goal:** produce the complete 4-page report using `report_insights.md` as the
single source of truth. Follow the LaTeX template in `Report_Template/`.

| Step | Status | Task |
|------|--------|------|
| 4.1 | ❌ | **Introduction:** problem statement, why SpMV matters, the investigation question |
| 4.2 | ❌ | **Methodology:** storage format (COO + row-pointer for warp kernel), all 5 GPU kernels with pseudocode, CPU baseline, validation method, measurement protocol (50 iterations, 2 warmup, float32, fixed-seed x), hardware/software environment |
| 4.3 | ❌ | **Dataset:** summary table of all 10 matrices (rows, cols, NNZ, avg NNZ/row, domain, one-line structural description); justify selection — diverse regimes, matches Chu et al. set |
| 4.4 | ❌ | **Results:** bandwidth plots and GFLOPS plots across all kernels and matrices; optional cache-miss indicator if profiling data is available |
| 4.5 | ❌ | **Discussion:** explain *why* each kernel wins or loses on each matrix subgroup. Address: load imbalance (TPR on power-law), atomicAdd contention (TPV on dense rows), warp efficiency (warp-per-row on structured matrices), memory-bound evidence, comparison vs cuSPARSE |
| 4.6 | ❌ | **Conclusion:** main lessons learned, limitations (COO overhead, float32-only), practical takeaways |
| 4.7 | ❌ | **References:** add Gao 2024, Chu HPDC'23 as required [1] and [2]; add Bell & Garland SC'09, Greathouse & Daga SC'14, and Yang et al. as supporting references |

**Checkpoint:** draft report complete, LaTeX compiles, under 4 pages.

---

### Week 5 (Apr 30 – May 7) — Polish, Verification, Submission ❌ NOT STARTED

**Goal:** catch any remaining issues before submission.

| Step | Status | Task |
|------|--------|------|
| 5.1 | ❌ | Final re-run of all benchmarks if any kernel or data change was made in Week 4 review |
| 5.2 | ❌ | Verify all correctness checks pass on every (kernel, matrix) pair |
| 5.3 | ❌ | Check report: correct student name, student ID, email, git repository link |
| 5.4 | ❌ | Check all plots are readable in PDF (font size, legend, axis labels) |
| 5.5 | ❌ | Push final code and results to git; confirm repo is accessible |
| 5.6 | ❌ | Export report to PDF; verify page count ≤ 4 |
| 5.7 | ❌ | Submit before May 7 at 23:59 |

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
