/* spmv_gpu_stride.cu — GPU SpMV: Grid-Stride Loop (COO format)
 *
 * Algorithm: every thread strides over all NNZ with step = total_threads.
 *   thread tid: for i = tid; i < nnz; i += total_threads:
 *                   y[Arows[i]] += Avals[i] * x[Acols[i]]
 * Grid size is decoupled from NNZ count — the kernel adapts to any launch
 * configuration and remains correct when total_threads < nnz.
 * Concurrent writes handled with atomicAdd.
 *
 * Improvements over last year's reference:
 *   A. Explicit device memory (cudaMalloc + cudaMemcpy).
 *   B. Correctness check against CPU naive reference (adaptive tolerance).
 *   C. __ldg() for x[] reads — read-only cache for random x accesses.
 *
 * Usage:
 *   ./bin/GPU/spmv_gpu_stride.exec path/to/matrix.mtx
 *   ./bin/GPU/spmv_gpu_stride.exec path/to/matrix.mtx <threads_per_block>
 *   ./bin/GPU/spmv_gpu_stride.exec path/to/matrix.mtx <threads_per_block> <num_blocks>
 *
 * When num_blocks is given explicitly the grid-stride cap is bypassed,
 * allowing the block/thread configuration sweep to test small block counts.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "mtx_io.h"

#define WARMUP            2
#define NITER             50
#define THREADS_PER_BLOCK 256

/* Grid: enough blocks to cover all NNZ; capped at 65535 (max grid dim x).
 * For large NNZ the cap kicks in and each thread strides over multiple elements. */
#define MAX_GRID_DIM 65535

/* ── Kernel ──────────────────────────────────────────────────────────────── */

__global__ void spmv_stride(const int    * __restrict__ Arows,
                             const int    * __restrict__ Acols,
                             const double * __restrict__ Avals,
                             const double * __restrict__ x,
                             double *y, int nnz)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;
    for (int i = tid; i < nnz; i += stride) {
        double val = Avals[i] * __ldg(&x[Acols[i]]);  /* improvement C */
        atomicAdd(&y[Arows[i]], val);
    }
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
    if (argc < 2 || argc > 4) {
        fprintf(stderr,
                "Usage: %s path/to/matrix.mtx [threads_per_block] [num_blocks]\n",
                argv[0]);
        return 1;
    }

    /* --- Load matrix into host COO arrays --- */
    int rows, cols, nnz;
    int    *h_Arows, *h_Acols;
    double *h_Avals;
    mtx_read_coo(argv[1], &rows, &cols, &nnz, &h_Arows, &h_Acols, &h_Avals);

    /* --- Host vectors --- */
    double *h_x     = (double *)malloc((size_t)cols * sizeof(double));
    double *h_y     = (double *)malloc((size_t)rows * sizeof(double));
    double *h_y_ref = (double *)calloc((size_t)rows,  sizeof(double));
    if (!h_x || !h_y || !h_y_ref) { fprintf(stderr, "malloc failed\n"); return 1; }
    for (int i = 0; i < cols; i++) h_x[i] = 1.0;

    /* --- CPU naive reference for correctness check (improvement B) --- */
    for (int i = 0; i < nnz; i++)
        h_y_ref[h_Arows[i]] += h_Avals[i] * h_x[h_Acols[i]];

    /* --- Unique column count for bandwidth formula --- */
    int *seen = (int *)calloc((size_t)cols, sizeof(int));
    int unique_cols = 0;
    for (int i = 0; i < nnz; i++)
        if (!seen[h_Acols[i]]) { seen[h_Acols[i]] = 1; unique_cols++; }
    free(seen);

    /* --- Allocate device memory (improvement A) --- */
    int    *d_Arows, *d_Acols;
    double *d_Avals, *d_x, *d_y;
    cudaMalloc((void **)&d_Arows, (size_t)nnz  * sizeof(int));
    cudaMalloc((void **)&d_Acols, (size_t)nnz  * sizeof(int));
    cudaMalloc((void **)&d_Avals, (size_t)nnz  * sizeof(double));
    cudaMalloc((void **)&d_x,     (size_t)cols * sizeof(double));
    cudaMalloc((void **)&d_y,     (size_t)rows * sizeof(double));

    cudaMemcpy(d_Arows, h_Arows, (size_t)nnz  * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_Acols, h_Acols, (size_t)nnz  * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_Avals, h_Avals, (size_t)nnz  * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,     h_x,     (size_t)cols * sizeof(double), cudaMemcpyHostToDevice);

    /* --- Kernel configuration (overridable via CLI for config sweep) --- */
    int tpb  = THREADS_PER_BLOCK;
    int grid = -1;   /* -1 = use automatic formula */

    if (argc >= 3) {
        int v = atoi(argv[2]);
        if (v > 0) tpb = v;
        else fprintf(stderr, "[warn] invalid threads_per_block, using %d\n", tpb);
    }
    if (argc == 4) {
        int v = atoi(argv[3]);
        if (v > 0) grid = v;   /* explicit override: bypass auto formula */
        else fprintf(stderr, "[warn] invalid num_blocks, using auto\n");
    }
    if (grid < 0) {
        /* default: enough blocks to cover all NNZ, capped at MAX_GRID_DIM */
        grid = (nnz + tpb - 1) / tpb;
        if (grid > MAX_GRID_DIM) grid = MAX_GRID_DIM;
    }

    /* --- Benchmark loop --- */
    double timers[NITER];
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (int iter = -WARMUP; iter < NITER; iter++) {
        cudaMemset(d_y, 0, (size_t)rows * sizeof(double));

        cudaEventRecord(ev_start);
        spmv_stride<<<grid, tpb>>>(d_Arows, d_Acols, d_Avals, d_x, d_y, nnz);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        if (iter >= 0) timers[iter] = (double)ms / 1000.0;
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    /* --- Copy result back and verify (improvement B) --- */
    cudaMemcpy(h_y, d_y, (size_t)rows * sizeof(double), cudaMemcpyDeviceToHost);

    double y_max = 0.0;
    for (int i = 0; i < rows; i++)
        if (fabs(h_y_ref[i]) > y_max) y_max = fabs(h_y_ref[i]);
    double tol = 1e-9 * y_max + 1e-14;
    int ok = 1;
    for (int i = 0; i < rows && ok; i++)
        if (fabs(h_y[i] - h_y_ref[i]) > tol) ok = 0;

    /* --- Statistics --- */
    double a_mean = arith_mean(timers, NITER);
    double g_mean = geom_mean(timers, NITER);

    double bytes = (double)nnz         * (2 * sizeof(int) + sizeof(double))
                 + (double)unique_cols * sizeof(double)
                 + (double)rows        * sizeof(double);
    double bandwidth = bytes / a_mean / 1.0e9;
    double gflops    = 2.0 * nnz / a_mean / 1.0e9;

    /* --- Output (matches parse_results.py format) --- */
    fprintf(stdout, "Correctness check: %s\n", ok ? "PASSED" : "FAILED");
    fprintf(stdout, "\nMatrix:              %s\n", argv[1]);
    fprintf(stdout, "Rows: %d  Cols: %d  NNZ: %d\n", rows, cols, nnz);
    fprintf(stdout, " %20s | %15s | %15s |\n",
            "kernel", "arith mean (s)", "geom mean (s)");
    fprintf(stdout, " %20s | %15f | %15f |\n",
            "spmv_gpu_stride", a_mean, g_mean);
    fprintf(stdout, "Effective bandwidth: %.4f GB/s\n", bandwidth);
    fprintf(stdout, "GFLOPS:              %.4f\n",       gflops);

    if (!ok) fprintf(stderr, "Correctness check: FAILED\n");

    /* --- Cleanup --- */
    cudaFree(d_Arows); cudaFree(d_Acols); cudaFree(d_Avals);
    cudaFree(d_x);     cudaFree(d_y);
    free(h_Arows); free(h_Acols); free(h_Avals);
    free(h_x);     free(h_y);     free(h_y_ref);
    return 0;
}
