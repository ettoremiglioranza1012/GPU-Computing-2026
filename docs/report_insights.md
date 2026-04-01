# Report Insights — Deliverable 1 (SpMV)

This document is the single source of truth for the downstream LLM writing the final report.
It collects all implementation decisions, measured results, analysis, and comparisons accumulated
during development. Update it after every new experiment.

---

## 1. CPU

### 1.1 Problem & Context

**Task:** Sparse Matrix–dense Vector Multiplication (SpMV): `y = A * x`
- `A` sparse matrix in COO format, read from Matrix Market (.mtx) files
- `x` dense vector initialised to 1.0
- `y` dense output vector

**Why memory-bound:** SpMV performs only 2 FLOPs per NNZ (1 multiply + 1 add) but requires
reading at minimum `Arows`, `Acols`, `Avals`, and random reads from `x`. The arithmetic intensity
is very low (~0.17 FLOP/byte for COO). Performance is therefore limited by memory bandwidth,
not compute throughput.

**Why random access to x hurts:** `x[Acols[i]]` is driven by column indices which follow no
regular pattern for unstructured matrices. This destroys spatial and temporal cache locality,
causing frequent cache misses on large vectors.

---

### 1.2 Storage Format: COO

We use **COO (Coordinate) format** exclusively, matching the course convention (Lab 5) and
last year's reference student.

```
Arows[nnz]  — row index of each non-zero (int, 0-indexed)
Acols[nnz]  — column index of each non-zero (int, 0-indexed)
Avals[nnz]  — value of each non-zero (double, 64-bit)
```

Space: `nnz × (4 + 4 + 8) = nnz × 16 bytes`

The COO arrays are **sorted row-major** (primary: row asc, secondary: col asc). This is required
by the optimised CPU kernel's row-tracking loop, and guarantees coalesced access patterns for
GPU kernels that stride over NNZ.

**Why not CSR:** COO is the course-standard format used in Lab 5 and last year's project.
CSR would eliminate the per-row boundary detection overhead but introduces a separate
`row_ptr` array and a preprocessing conversion step.

---

### 1.3 Matrix I/O (`include/mtx_io.h`)

Single shared header included by all kernel files. Key features vs last year's inline parsing:

| Feature | Last year | This year |
|---------|-----------|-----------|
| Code location | Copy-pasted in each file | Shared header `mtx_io.h` |
| `symmetric` matrix expansion | Not handled | Fully expanded (off-diagonal mirrored) |
| Row-major sorting | Separate `sort.c` utility + manual script | Inline `qsort` on every read |
| `pattern` matrix type | Not handled | Reads as implicit 1.0 values |
| Buffer size | 100 bytes (overflow risk) | 512 bytes |

The `symmetric` flag matters for `bcsstk17` (structural stiffness), which is stored as a
symmetric matrix in the SuiteSparse archive. Without expansion, NNZ would be approximately
half the real value.

---

### 1.4 CPU Implementations

#### 1.4.1 Naive (`spmv_coo_naive.c`)

Single sequential scan over all NNZ:

```c
for (int i = 0; i < nnz; i++)
    y[Arows[i]] += Avals[i] * x[Acols[i]];
```

**Properties:**
- No unrolling, no ILP
- Sequential access to COO arrays (cache-friendly)
- Random read from `x` (cache pressure)
- Random write to `y` (repeated writes to same row)
- Establishes the performance floor

#### 1.4.2 Optimised — 4-way loop unrolling (`spmv_coo_opt.c`)

Row-tracking scan with 4 independent accumulators:

```c
while (i < nnz) {
    int row = Arows[i];
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    while (i < nnz && Arows[i] == row) {
        if (i + 3 < nnz && Arows[i+3] == row) {
            sum0 += Avals[i]   * x[Acols[i]];
            sum1 += Avals[i+1] * x[Acols[i+1]];
            sum2 += Avals[i+2] * x[Acols[i+2]];
            sum3 += Avals[i+3] * x[Acols[i+3]];
            i += 4;
        } else {
            sum0 += Avals[i] * x[Acols[i]];
            i += 1;
        }
    }
    y[row] += sum0 + sum1 + sum2 + sum3;
}
```

**Optimisations over naive:**
1. **4 independent accumulators** break the single-accumulator dependency chain → ILP
2. **Single write per row** (`y[row] += sum`) instead of one random write per NNZ → fewer
   cache conflicts on `y`
3. **Row locality** — all NNZ in the same row are processed together before moving on

**When it helps vs hurts:**
- Helps on matrices with many NNZ/row (e.g. `bcsstk17` at avg 39 NNZ/row → 34% speedup)
- Hurts on very sparse matrices where most rows have <4 NNZ (e.g. `web-Google` at avg 5.6
  NNZ/row → 20% regression). The overhead of 4 accumulators and the row-boundary branch
  outweighs the ILP benefit when the unrolled loop body rarely executes.

---

### 1.5 Benchmarking Infrastructure

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `WARMUP` | 2 | Two discarded iterations to let CPU caches and branch predictors stabilise |
| `NITER` | 50 | 50 timed iterations — matches last year's 51-1 protocol for stable statistics |
| Timing | `gettimeofday` (µs resolution) | Same macros as Lab 5 (`TIMER_DEF`, `TIMER_START`, `TIMER_STOP`, `TIMER_ELAPSED`) |
| Statistics | Arithmetic mean + geometric mean | Arithmetic mean for bandwidth/GFLOPS; geometric mean more robust to outliers |

**Bandwidth formula (matches last year):**
```
bytes = nnz × (2×sizeof(int) + sizeof(double))   ← Arows + Acols + Avals
      + unique_cols × sizeof(double)               ← distinct x[] reads (not worst-case nnz)
      + rows × sizeof(double)                      ← y write
```
`unique_cols` is computed by scanning `Acols` and counting distinct values. This gives a
more accurate picture of actual memory traffic to `x` than assuming one access per NNZ.

---

### 1.6 Correctness Checking

**naive:** two identical runs of the same kernel are compared. Results must be bit-identical
so absolute tolerance `1e-9` is used.

**opt vs naive:** different FP accumulation order (4-way partial sums vs sequential) produces
mathematically equivalent but numerically distinct results. Tolerance is:

```c
double y_max = max(|y_ref[i]|) over all rows;
double tol   = 1e-9 * y_max + 1e-14;
```

Rationale: rows whose sum cancels to near-zero (e.g. admittance rows in `1138_bus` that
satisfy Kirchhoff's law) get the same absolute slack as the largest row in the vector.
A per-element relative tolerance degenerates to an unworkably tight floor for these rows.

---

### 1.7 Dataset

| Matrix | Rows | NNZ | Avg NNZ/row | Domain | Storage |
|--------|------|-----|-------------|--------|---------|
| `1138_bus` | 1,138 | 4,054 | 3.6 | Power network (bus admittance) | General |
| `bcsstk17` | 10,974 | 428,650 | 39.1 | Structural stiffness | Symmetric → expanded |
| `web-Google` | 916,428 | 5,105,039 | 5.6 | Web graph (directed links) | General |

**Note on `bcsstk17`:** stored as symmetric in the SuiteSparse archive. After expansion by
`mtx_read_coo`, NNZ approximately doubles from the stored count.

---

### 1.8 CPU Results (job 1475030, cluster Baldo, Intel Xeon Silver 4309Y)

#### Effective Bandwidth (GB/s)

| Matrix | naive | opt | opt vs naive |
|--------|-------|-----|--------------|
| `1138_bus` | 20.8 | 17.7 | −15% (regression — too few NNZ/row) |
| `bcsstk17` | 6.4 | 38.0 | **+495%** (strong gain — 39 NNZ/row) |
| `web-Google` | 8.1 | 6.8 | −16% (regression — too few NNZ/row) |

#### Arithmetic Mean Time (s)

| Matrix | naive | opt |
|--------|-------|-----|
| `1138_bus` | 0.000004 | 0.000005 |
| `bcsstk17` | 0.001106 | 0.000185 |
| `web-Google` | 0.011699 | 0.013900 |

#### GFLOPS

| Matrix | naive | opt |
|--------|-------|-----|
| `1138_bus` | 2.03 | 1.73 |
| `bcsstk17` | 0.78 | 4.63 |
| `web-Google` | 0.87 | 0.73 |

---

### 1.9 Key CPU Observations (for the report narrative)

1. **The opt kernel is not universally better.** It regresses on `1138_bus` (3.6 NNZ/row)
   and `web-Google` (5.6 NNZ/row) because the 4-way unrolled loop body almost never executes
   — most rows fall through to the scalar tail, paying overhead with no benefit.

2. **The opt kernel excels on dense rows.** `bcsstk17` (39 NNZ/row) sees a ~6× speedup in
   bandwidth. The 4-accumulator ILP and single-write-per-row both contribute.

3. **Matrix structure dominates performance.** `web-Google` has 126× more NNZ than `1138_bus`
   but only ~2× slower on naive — irregular graph structure (random column accesses) causes
   cache misses that far outweigh the extra arithmetic.

4. **`bcsstk17` naive is slow despite moderate NNZ.** The symmetric expansion approximately
   doubles the effective NNZ vs what is stored. The large stiffness values and dense row
   structure mean each random `x[Acols[i]]` access is a different cache line → high miss rate.

5. **`web-Google` opt regression is explained by sparsity pattern.** Average 5.6 NNZ/row means
   at best one 4-way unrolled iteration per row, often zero. Yet we pay for 4 accumulator
   initialisations and 3 extra additions per row on every row.

---

### 1.10 Differences from Last Year's Reference Student (CPU)

| Aspect | Last year | This year |
|--------|-----------|-----------|
| Format | COO | COO (same) |
| Algorithms | naive + 4-way opt | naive + 4-way opt (same) |
| Matrix reader | Inline per-file | Shared `mtx_io.h` |
| Symmetric expansion | Missing | Handled |
| Sorting | Separate manual script | Inline in reader |
| Correctness check | None | opt vs naive with adaptive tolerance |
| Statistics | Arithmetic mean only | Arithmetic + geometric mean |
| Iterations | 51 total, 1 warmup | 52 total, 2 warmup, 50 measured |
| Bandwidth formula | Unique column count | Unique column count (same) |

---

## 2. GPU

### 2.1 Baseline Kernels

All three kernels operate on COO format (inherited from `mtx_io.h`) and share the same
host-side structure. They correspond directly to last year's reference student's GPU
implementations, adapted to use our shared infrastructure.

#### 2.1.1 Thread-Per-Value (`spmv_gpu_tpv.cu`)

One GPU thread per non-zero element:
```cuda
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < nnz)
    atomicAdd(&y[Arows[tid]], Avals[tid] * x[Acols[tid]]);
```
Grid size: `ceil(nnz / THREADS_PER_BLOCK)` — one thread per NNZ, no striding needed.

**Properties:**
- Simplest mapping: maximum parallelism (one thread per unit of work)
- `atomicAdd` required: multiple threads can write to the same `y[row]` when a row has
  many NNZ → serialisation at the atomic unit; contention grows with NNZ/row
- Memory access to `Arows`, `Acols`, `Avals` is coalesced (threads access consecutive NNZ)
- Access to `x[Acols[tid]]` is random — threads in the same warp access different columns

#### 2.1.2 Thread-Per-Row (`spmv_gpu_tpr.cu`)

One GPU thread per matrix row:
```cuda
int row = blockIdx.x * blockDim.x + threadIdx.x;
// binary search for first NNZ of this row in sorted COO
double sum = 0.0;
for (int i = start; i < nnz && Arows[i] == row; i++)
    sum += Avals[i] * x[Acols[i]];
y[row] = sum;
```
Grid size: `ceil(rows / THREADS_PER_BLOCK)`.

**Properties:**
- No `atomicAdd`: each thread owns its output element `y[row]` exclusively
- Thread divergence within a warp: threads on the same warp process rows of different
  lengths (especially severe on power-law matrices like `web-Google`)
- Binary search overhead: O(log nnz) per thread before any arithmetic begins
- Load imbalance: threads on long rows (e.g. a hub in `web-Google`) stall the warp

#### 2.1.3 Grid-Stride Loop (`spmv_gpu_stride.cu`)

Every thread iterates over NNZ with a stride equal to the total thread count:
```cuda
int tid    = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x  * blockDim.x;
for (int i = tid; i < nnz; i += stride)
    atomicAdd(&y[Arows[i]], Avals[i] * x[Acols[i]]);
```
Grid size: `min(ceil(nnz / THREADS_PER_BLOCK), 65535)`.

**Properties:**
- Grid size decoupled from NNZ: still correct with fewer blocks than NNZ
- `atomicAdd` required (same as TPV)
- For large matrices (e.g. `web-Google`, 5M NNZ) with the capped grid, each thread
  processes multiple NNZ — reduces kernel launch overhead, improves cache reuse on `x`
  between consecutive accesses by the same thread
- Considered the most scalable pattern: adapts to any hardware SM count

#### 2.1.4 Common Properties of All Baseline Kernels

- **Format:** COO (same arrays as CPU side, copied to device)
- **Timing:** `cudaEventRecord` wrapping only the kernel launch — GPU-side time only,
  no host↔device transfer included (kernel-only convention, matching the paper and labs)
- **x access pattern:** random column indices → L1/L2 cache pressure (same root problem
  as CPU, amplified because GPU L1 is shared across many concurrent warps)
- **Bandwidth formula:** identical to CPU formula (kernel-only bytes):
  ```
  bytes = nnz × (2×sizeof(int) + sizeof(double))   ← Arows + Acols + Avals
        + unique_cols × sizeof(double)               ← distinct x[] reads
        + rows × sizeof(double)                      ← y write
  ```
- **Output format:** identical printf format to CPU kernels → `parse_results.py` works
  unchanged for both CPU and GPU output files

---

### 2.2 Improvements Included (over last year's GPU reference)

#### A. Explicit Device Memory (`cudaMalloc` + `cudaMemcpy`)

Last year used `cudaMallocManaged` (unified memory) for all arrays. We use explicit device
memory instead:

```c
cudaMalloc(&d_Arows, nnz * sizeof(int));
cudaMemcpy(d_Arows, h_Arows, nnz * sizeof(int), cudaMemcpyHostToDevice);
// ... kernel benchmark loop ...
cudaMemcpy(h_y, d_y, rows * sizeof(double), cudaMemcpyDeviceToHost);
cudaFree(d_Arows);
```

**Why:** unified memory uses on-demand page migration. On first kernel access, pages fault
from host to device — adding latency that contaminates the first iteration. Even with
warmup, the migration cost can linger across iterations for large matrices. Explicit
transfers front-load the cost once before the benchmark loop, giving clean kernel-only
timing. This is also standard practice in production HPC code.

#### B. Correctness Check Against CPU Naive Reference

Before the benchmark, we run the CPU naive kernel on host arrays to produce `h_y_ref`.
After benchmarking, the GPU result is copied back and compared element-wise:

```c
double y_max = max(|h_y_ref[i]|);
double tol   = 1e-9 * y_max + 1e-14;
int ok = all(|h_y[i] - h_y_ref[i]| <= tol);
```

The adaptive tolerance handles the non-deterministic summation order in `atomicAdd`-based
kernels (TPV, stride): floating-point associativity means results differ from the sequential
CPU sum by small rounding errors, not by algorithmic bugs.

TPR is deterministic per-row (sequential inner loop), so its differences from naive are
purely due to parallel row ordering — mathematically identical per row.

**Why:** last year had no verification. A silent wrong result would invalidate all
performance numbers. This check catches implementation bugs before wasting cluster time.

#### C. `__ldg()` Read-Only Cache for x Vector

In all three kernels, every access to `x[col]` is routed through the L1 read-only cache
(the GPU's texture cache path):

```cuda
double val = Avals[i] * __ldg(&x[Acols[i]]);
```

**Why:** Yang et al. identify random access to `x` as the primary bottleneck for GPU SpMV.
The L1 read-only cache (256 KB on Ampere) is separate from the regular L1 data cache and
is optimised for broadcast and streaming read patterns. Since `x` is read-only during the
kernel (we never write to it), `__ldg()` is semantically safe and explicitly opts into
this cache path. On matrices where the active columns of `x` fit in the cache (small
`unique_cols`), this can provide significant latency reduction.

---

### 2.3 Benchmarking Infrastructure (GPU)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `WARMUP` | 2 | Two discarded iterations to warm up GPU pipeline and L2 cache |
| `NITER` | 50 | 50 timed iterations — same as CPU for apples-to-apples comparison |
| Timing | `cudaEventRecord` + `cudaEventElapsedTime` | GPU-async safe; measures device execution time only |
| Statistics | Arithmetic mean + geometric mean | Same as CPU side |
| Grid/block | 256 threads/block; grid per-kernel (see §2.1) | Default 256 matches last year; sufficient for sm_80 occupancy |

**Why `cudaEventRecord` not `gettimeofday`:** GPU kernels are asynchronous from the host.
`gettimeofday` measures host-side time including kernel dispatch latency and driver overhead,
not actual execution time. `cudaEventRecord` inserts GPU-side timestamps into the command
stream, giving true device execution time.

**`cudaMemset` is outside the timed region:** the output vector `d_y` is zeroed before each
`cudaEventRecord(start)`. This matches the CPU protocol where `memset(y, 0, ...)` precedes
`TIMER_START`. Both measure only the arithmetic kernel.

---

### 2.4 Dataset (GPU)

Same three matrices as CPU side:

| Matrix | Rows | NNZ | Avg NNZ/row | Expected GPU behaviour |
|--------|------|-----|-------------|------------------------|
| `1138_bus` | 1,138 | 4,054 | 3.6 | Very small — launch overhead dominates; all kernels slow |
| `bcsstk17` | 10,974 | 428,650 | 39.1 | Moderate size, uniform rows — TPR benefits; TPV contention on long rows |
| `web-Google` | 916,428 | 5,105,039 | 5.6 | Large, power-law — TPR diverges badly; TPV/stride benefit from parallelism |

---

### 2.5 GPU Results

*(Populate after cluster runs — mirror §1.8 table structure)*

| Matrix | tpv BW | tpr BW | stride BW |
|--------|--------|--------|-----------|
| `1138_bus` | — | — | — |
| `bcsstk17` | — | — | — |
| `web-Google` | — | — | — |

---

### 2.6 Key GPU Observations

*(Populate after cluster runs)*

---

### 2.7 Differences from Last Year's Reference Student (GPU)

| Aspect | Last year | This year |
|--------|-----------|-----------|
| Memory model | `cudaMallocManaged` (unified memory) | Explicit `cudaMalloc` + `cudaMemcpy` |
| Matrix reader | Inline per-file (100-byte buffer, no symmetric handling) | Shared `mtx_io.h` |
| Correctness check | None | vs CPU naive with adaptive tolerance |
| x-vector cache | Plain `x[col]` load | `__ldg(&x[col])` via read-only cache |
| Output format | Custom (not parseable by our scripts) | Matches `parse_results.py` format |
| Timing warmup | 1 iteration discarded | 2 iterations discarded |
| Statistics | Arithmetic mean only | Arithmetic + geometric mean |

---

### 2.8 Future Improvements

#### D. Warp-Per-Row (`spmv_gpu_warp.cu`) — not implemented

Assign one 32-thread warp to each matrix row. Each thread handles `row_nnz / 32` elements,
accumulates a partial sum, then the warp reduces via `__shfl_xor_sync`. One write to
`y[row]` per warp — no `atomicAdd`.

```cuda
int row  = blockIdx.x;                      // one block per row (or chunk of rows)
int lane = threadIdx.x % 32;               // lane within warp
double partial = 0.0;
for (int i = row_start + lane; i < row_end; i += 32)
    partial += Avals[i] * __ldg(&x[Acols[i]]);
// warp reduce
for (int offset = 16; offset > 0; offset >>= 1)
    partial += __shfl_xor_sync(0xffffffff, partial, offset);
if (lane == 0) y[row] = partial;
```

**Why deferred:** implementing warp shuffles requires careful handling of partial warps
(rows shorter than 32 NNZ) and requires CSR format (or a row-pointer array) to efficiently
find `row_start`/`row_end`. This directly implements the paper's "CSR-vector" concept and
would be the strongest kernel for `bcsstk17` (39 NNZ/row ≈ 1.2 warps). Added to future
work to keep scope manageable for D1.
