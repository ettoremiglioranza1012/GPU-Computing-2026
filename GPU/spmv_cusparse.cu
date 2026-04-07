/* spmv_cusparse.cu — GPU SpMV: cuSPARSE baseline (CSR format built from COO)
 *
 * Algorithm: wraps cusparseSpMV() using CSR storage.
 *   COO Arows is converted to a CSR row_ptr via a host-side prefix-sum (one-time
 *   setup cost, reported separately). cusparseSpMV with CUSPARSE_SPMV_ALG_DEFAULT
 *   is used for the timed benchmark — this is the library's auto-tuned path.
 *
 * Purpose: industry reference baseline. The professor asks:
 *   "Compare with CuSparse implementation (does it matter if your implementation
 *    is slower)"
 *
 * Improvements over existing kernels:
 *   A. Explicit device memory — no unified-memory overhead; kernel timing is clean.
 *   B. Correctness check against CPU naive reference (adaptive tolerance).
 *
 * Usage: ./bin/GPU/spmv_cusparse.exec path/to/matrix.mtx
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include "mtx_io.h"

#define WARMUP 2
#define NITER  50

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

    /* --- CPU naive reference for correctness check --- */
    for (int i = 0; i < nnz; i++)
        h_y_ref[h_Arows[i]] += h_Avals[i] * h_x[h_Acols[i]];

    /* --- Unique column count for bandwidth formula --- */
    int *seen = (int *)calloc((size_t)cols, sizeof(int));
    int unique_cols = 0;
    for (int i = 0; i < nnz; i++)
        if (!seen[h_Acols[i]]) { seen[h_Acols[i]] = 1; unique_cols++; }
    free(seen);

    /* --- COO → CSR: build row_ptr via prefix-sum on Arows (one-time setup) ---
     * This is a format conversion cost; reported separately from kernel time. */
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    int *h_row_ptr = (int *)calloc((size_t)(rows + 1), sizeof(int));
    if (!h_row_ptr) { fprintf(stderr, "malloc failed\n"); return 1; }
    for (int i = 0; i < nnz; i++)  h_row_ptr[h_Arows[i] + 1]++;
    for (int i = 0; i < rows; i++) h_row_ptr[i + 1] += h_row_ptr[i];

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double setup_s = (double)(ts1.tv_sec  - ts0.tv_sec)
                   + (double)(ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

    /* --- Allocate device memory (improvement A) --- */
    int   *d_row_ptr, *d_Acols;
    float *d_Avals, *d_x, *d_y;
    cudaMalloc((void **)&d_row_ptr, (size_t)(rows + 1) * sizeof(int));
    cudaMalloc((void **)&d_Acols,   (size_t)nnz        * sizeof(int));
    cudaMalloc((void **)&d_Avals,   (size_t)nnz        * sizeof(float));
    cudaMalloc((void **)&d_x,       (size_t)cols       * sizeof(float));
    cudaMalloc((void **)&d_y,       (size_t)rows       * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr, (size_t)(rows + 1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_Acols,   h_Acols,   (size_t)nnz        * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_Avals,   h_Avals,   (size_t)nnz        * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,       h_x,       (size_t)cols       * sizeof(float), cudaMemcpyHostToDevice);

    /* --- cuSPARSE descriptors --- */
    cusparseHandle_t     handle;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    cusparseCreate(&handle);
    cusparseCreateCsr(&matA,
                      (int64_t)rows, (int64_t)cols, (int64_t)nnz,
                      d_row_ptr, d_Acols, d_Avals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecX, (int64_t)cols, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, (int64_t)rows, d_y, CUDA_R_32F);

    float alpha = 1.0f, beta = 0.0f;

    /* Determine workspace size and allocate */
    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY,
                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    void *d_buffer = NULL;
    if (bufferSize > 0) cudaMalloc(&d_buffer, bufferSize);

    /* --- Benchmark loop --- */
    double timers[NITER];
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (int iter = -WARMUP; iter < NITER; iter++) {
        /* beta=0 means y is fully overwritten; memset is belt-and-suspenders */
        cudaMemset(d_y, 0, (size_t)rows * sizeof(float));

        cudaEventRecord(ev_start);
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, vecX, &beta, vecY,
                     CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
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

    /* Bandwidth: same formula as warp kernel (CSR = COO without Arows, plus row_ptr) */
    double bytes = (double)nnz         * (sizeof(int) + sizeof(float))
                 + (double)(rows + 1)  * sizeof(int)
                 + (double)unique_cols * sizeof(float)
                 + (double)rows        * sizeof(float);
    double bandwidth = bytes / a_mean / 1.0e9;
    double gflops    = 2.0 * nnz / a_mean / 1.0e9;

    /* --- Output (matches parse_results.py format) --- */
    fprintf(stdout, "Correctness check: %s\n", ok ? "PASSED" : "FAILED");
    fprintf(stdout, "Setup (COO->CSR):        %.6f s\n", setup_s);
    fprintf(stdout, "\nMatrix:              %s\n", argv[1]);
    fprintf(stdout, "Rows: %d  Cols: %d  NNZ: %d\n", rows, cols, nnz);
    fprintf(stdout, " %20s | %15s | %15s |\n",
            "kernel", "arith mean (s)", "geom mean (s)");
    fprintf(stdout, " %20s | %15f | %15f |\n",
            "spmv_cusparse", a_mean, g_mean);
    fprintf(stdout, "Effective bandwidth: %.4f GB/s\n", bandwidth);
    fprintf(stdout, "GFLOPS:              %.4f\n",       gflops);

    if (!ok) fprintf(stderr, "Correctness check: FAILED\n");

    /* --- Cleanup --- */
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    if (d_buffer) cudaFree(d_buffer);
    cudaFree(d_row_ptr); cudaFree(d_Acols); cudaFree(d_Avals);
    cudaFree(d_x);       cudaFree(d_y);
    free(h_Arows); free(h_Acols); free(h_Avals);
    free(h_row_ptr);
    free(h_x);     free(h_y);     free(h_y_ref);
    return 0;
}
