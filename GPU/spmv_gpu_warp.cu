/* spmv_gpu_warp.cu — GPU SpMV: Warp-Per-Row with warp-shuffle reduction (COO + row_ptr)
 *
 * Algorithm: one warp (32 threads) per row.
 *   Each lane accumulates a partial dot-product over its strided slice of the row's NNZ.
 *   The 32 partial sums are reduced via __shfl_xor_sync — no atomicAdd, no bank conflicts.
 *
 * Storage: COO arrays (Avals, Acols) from mtx_io.h unchanged.
 *   A row-pointer array (row_ptr[rows+1]) is built from the sorted COO Arows on the host
 *   via a single O(NNZ + rows) prefix-sum. Build time is printed separately from kernel
 *   time to comply with the setup-cost separation requirement.
 *
 * Improvements over existing kernels:
 *   A. Explicit device memory — no unified-memory overhead; kernel timing is clean.
 *   B. Correctness check against CPU naive reference (adaptive tolerance).
 *   C. __ldg() for x[] reads — read-only cache for random x accesses.
 *   D. Warp shuffle reduction — eliminates atomicAdd contention (vs TPV) and per-thread
 *      serial row traversal (vs TPR). Each row's dot-product is parallelised across 32 lanes.
 *
 * Usage: ./bin/GPU/spmv_gpu_warp.exec path/to/matrix.mtx
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "mtx_io.h"

#define WARMUP            2
#define NITER             50
#define WARP_SIZE         32
#define THREADS_PER_BLOCK 256              /* 8 warps per block */
#define WARPS_PER_BLOCK   (THREADS_PER_BLOCK / WARP_SIZE)

/* ── Kernel ──────────────────────────────────────────────────────────────── */

__global__ void spmv_warp(const int   * __restrict__ Acols,
                           const float * __restrict__ Avals,
                           const int   * __restrict__ row_ptr,
                           const float * __restrict__ x,
                           float *y, int rows)
{
    /* One warp per row */
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;

    if (warp_id >= rows) return;

    int row_start = row_ptr[warp_id];
    int row_end   = row_ptr[warp_id + 1];

    /* Each lane strides through the row's NNZ, accumulating a partial sum */
    float partial = 0.0f;
    for (int i = row_start + lane; i < row_end; i += WARP_SIZE)
        partial += Avals[i] * __ldg(&x[Acols[i]]);  /* improvement C */

    /* Warp shuffle tree reduction: fold 32 partial sums into lane 0 */
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, offset);

    if (lane == 0) y[warp_id] = partial;
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static double arith_mean(const double *t, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += t[i];
    return s / n;
}

static double geom_mean(const double *t, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += log(t[i]);
    return exp(s / n);
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s path/to/matrix.mtx\n", argv[0]);
        return 1;
    }

    /* --- Load matrix into host COO arrays --- */
    int rows, cols, nnz;
    int   *h_Arows, *h_Acols;
    float *h_Avals;
    mtx_read_coo(argv[1], &rows, &cols, &nnz, &h_Arows, &h_Acols, &h_Avals);

    /* --- Host vectors: fixed-seed random (reproducible) --- */
    float *h_x     = (float *)malloc((size_t)cols * sizeof(float));
    float *h_y     = (float *)malloc((size_t)rows * sizeof(float));
    float *h_y_ref = (float *)calloc((size_t)rows,  sizeof(float));
    if (!h_x || !h_y || !h_y_ref) { fprintf(stderr, "malloc failed\n"); return 1; }
    srand(42);
    for (int i = 0; i < cols; i++) h_x[i] = (float)rand() / (float)RAND_MAX;

    /* --- CPU naive reference for correctness check (improvement B) --- */
    for (int i = 0; i < nnz; i++)
        h_y_ref[h_Arows[i]] += h_Avals[i] * h_x[h_Acols[i]];

    /* --- Unique column count for bandwidth formula --- */
    int *seen = (int *)calloc((size_t)cols, sizeof(int));
    int unique_cols = 0;
    for (int i = 0; i < nnz; i++)
        if (!seen[h_Acols[i]]) { seen[h_Acols[i]] = 1; unique_cols++; }
    free(seen);

    /* --- Build row-pointer array from sorted COO (one-time setup cost) ---
     * Single pass over Arows to count NNZ per row, then prefix-sum to get
     * row_ptr[r] = index of first NNZ in row r. Equivalent to CSR row_ptr.
     * Time this separately: it is a setup cost, not kernel execution time. */
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    int *h_row_ptr = (int *)calloc((size_t)(rows + 1), sizeof(int));
    if (!h_row_ptr) { fprintf(stderr, "malloc failed\n"); return 1; }
    for (int i = 0; i < nnz; i++)  h_row_ptr[h_Arows[i] + 1]++;
    for (int i = 0; i < rows; i++) h_row_ptr[i + 1] += h_row_ptr[i];

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double setup_s = (double)(ts1.tv_sec  - ts0.tv_sec)
                   + (double)(ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

    /* --- Allocate device memory (improvement A) ---
     * Note: d_Arows is not transferred — the warp kernel uses row_ptr instead. */
    int   *d_Acols, *d_row_ptr;
    float *d_Avals, *d_x, *d_y;
    cudaMalloc((void **)&d_Acols,   (size_t)nnz        * sizeof(int));
    cudaMalloc((void **)&d_Avals,   (size_t)nnz        * sizeof(float));
    cudaMalloc((void **)&d_row_ptr, (size_t)(rows + 1) * sizeof(int));
    cudaMalloc((void **)&d_x,       (size_t)cols       * sizeof(float));
    cudaMalloc((void **)&d_y,       (size_t)rows       * sizeof(float));

    cudaMemcpy(d_Acols,   h_Acols,   (size_t)nnz        * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_Avals,   h_Avals,   (size_t)nnz        * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, h_row_ptr, (size_t)(rows + 1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,       h_x,       (size_t)cols       * sizeof(float), cudaMemcpyHostToDevice);

    /* --- Kernel configuration: one warp per row --- */
    int grid = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    /* --- Benchmark loop --- */
    double timers[NITER];
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (int iter = -WARMUP; iter < NITER; iter++) {
        cudaMemset(d_y, 0, (size_t)rows * sizeof(float));

        cudaEventRecord(ev_start);
        spmv_warp<<<grid, THREADS_PER_BLOCK>>>(d_Acols, d_Avals, d_row_ptr, d_x, d_y, rows);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        if (iter >= 0) timers[iter] = (double)ms / 1000.0;
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    /* --- Copy result back and verify (improvement B) --- */
    cudaMemcpy(h_y, d_y, (size_t)rows * sizeof(float), cudaMemcpyDeviceToHost);

    float y_max = 0.0f;
    for (int i = 0; i < rows; i++)
        if (fabsf(h_y_ref[i]) > y_max) y_max = fabsf(h_y_ref[i]);
    float tol = 1e-3f * y_max + 1e-5f;
    int ok = 1;
    for (int i = 0; i < rows && ok; i++)
        if (fabsf(h_y[i] - h_y_ref[i]) > tol) ok = 0;

    /* --- Statistics --- */
    double a_mean = arith_mean(timers, NITER);
    double g_mean = geom_mean(timers, NITER);

    /* Bandwidth: Acols + Avals (NNZ reads) + row_ptr (rows+1 reads) +
     *            unique x entries (unique_cols reads) + y (rows writes) */
    double bytes = (double)nnz         * (sizeof(int) + sizeof(float))
                 + (double)(rows + 1)  * sizeof(int)
                 + (double)unique_cols * sizeof(float)
                 + (double)rows        * sizeof(float);
    double bandwidth = bytes / a_mean / 1.0e9;
    double gflops    = 2.0 * nnz / a_mean / 1.0e9;

    /* --- Output (matches parse_results.py format) --- */
    fprintf(stdout, "Correctness check: %s\n", ok ? "PASSED" : "FAILED");
    fprintf(stdout, "Setup (row_ptr build):  %.6f s\n", setup_s);
    fprintf(stdout, "\nMatrix:              %s\n", argv[1]);
    fprintf(stdout, "Rows: %d  Cols: %d  NNZ: %d\n", rows, cols, nnz);
    fprintf(stdout, " %20s | %15s | %15s |\n",
            "kernel", "arith mean (s)", "geom mean (s)");
    fprintf(stdout, " %20s | %15f | %15f |\n",
            "spmv_gpu_warp", a_mean, g_mean);
    fprintf(stdout, "Effective bandwidth: %.4f GB/s\n", bandwidth);
    fprintf(stdout, "GFLOPS:              %.4f\n",       gflops);

    if (!ok) fprintf(stderr, "Correctness check: FAILED\n");

    /* --- Cleanup --- */
    cudaFree(d_Acols); cudaFree(d_Avals); cudaFree(d_row_ptr);
    cudaFree(d_x);     cudaFree(d_y);
    free(h_Arows); free(h_Acols); free(h_Avals);
    free(h_row_ptr);
    free(h_x);     free(h_y);     free(h_y_ref);
    return 0;
}
