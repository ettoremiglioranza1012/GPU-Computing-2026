/* spmv_coo_opt.c — CPU SpMV optimized with 4-way loop unrolling (COO format)
 *
 * Algorithm: row-tracking scan over sorted COO arrays.
 * For each row, accumulate into 4 independent partial sums (sum0–sum3)
 * to expose instruction-level parallelism, then write once to y[row].
 *
 * Requires: COO arrays sorted row-major (mtx_read_coo guarantees this).
 *
 * Usage: ./bin/CPU/spmv_coo_opt path/to/matrix.mtx
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "my_time_lib.h"
#include "mtx_io.h"

#define WARMUP 2
#define NITER  50

/* 4-way unrolled COO SpMV.
 * Scans NNZ in row-major order. For each row, fills 4 independent accumulators
 * (sum0–sum3) to break the dependency chain and enable ILP, then writes the
 * row result once (no repeated random writes to y).
 */
static void spmv_coo_opt(const int *Arows, const int *Acols,
                         const float *Avals, int nnz,
                         const float *x, float *y)
{
    int i = 0;
    while (i < nnz) {
        int row = Arows[i];
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

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
}

/* Naive reference kernel for correctness check */
static void spmv_coo_naive_ref(const int *Arows, const int *Acols,
                               const float *Avals, int nnz,
                               const float *x, float *y)
{
    for (int i = 0; i < nnz; i++)
        y[Arows[i]] += Avals[i] * x[Acols[i]];
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s path/to/matrix.mtx\n", argv[0]);
        return 1;
    }

    /* --- Load matrix --- */
    int rows, cols, nnz;
    int   *Arows, *Acols;
    float *Avals;
    mtx_read_coo(argv[1], &rows, &cols, &nnz, &Arows, &Acols, &Avals);

    /* --- Dense input vector: fixed-seed random (reproducible) --- */
    float *x     = (float *)malloc((size_t)cols * sizeof(float));
    float *y     = (float *)malloc((size_t)rows * sizeof(float));
    float *y_ref = (float *)malloc((size_t)rows * sizeof(float));
    if (!x || !y || !y_ref) { fprintf(stderr, "malloc failed\n"); return 1; }
    srand(42);
    for (int i = 0; i < cols; i++) x[i] = (float)rand() / (float)RAND_MAX;

    /* --- Reference result from naive kernel --- */
    memset(y_ref, 0, (size_t)rows * sizeof(float));
    spmv_coo_naive_ref(Arows, Acols, Avals, nnz, x, y_ref);

    /* --- Benchmark loop --- */
    double timers[NITER];
    TIMER_DEF(0);

    for (int iter = -WARMUP; iter < NITER; iter++) {
        memset(y, 0, (size_t)rows * sizeof(float));

        TIMER_START(0);
        spmv_coo_opt(Arows, Acols, Avals, nnz, x, y);
        TIMER_STOP(0);

        double t = TIMER_ELAPSED(0) / 1.e6;
        if (iter >= 0) timers[iter] = t;

        /* Correctness check on first warm-up pass.
         * Tolerance is relative to the largest |y_ref| entry so that rows
         * whose sum cancels to near-zero get the same slack as the largest row. */
        if (iter == -WARMUP) {
            float y_max = 0.0f;
            for (int i = 0; i < rows; i++)
                if (fabsf(y_ref[i]) > y_max) y_max = fabsf(y_ref[i]);
            float tol = 1e-5f * y_max + 1e-7f;
            int ok = 1;
            for (int i = 0; i < rows && ok; i++)
                if (fabsf(y[i] - y_ref[i]) > tol) ok = 0;
            fprintf(stdout, "Correctness check: %s\n", ok ? "PASSED" : "FAILED");
            if (!ok) fprintf(stderr, "Correctness check: FAILED\n");
        }

        fprintf(stdout, "Iteration %d: %f s\n", iter, t);
    }

    /* --- Statistics --- */
    double a_mean = arithmetic_mean(timers, NITER);
    double g_mean = geometric_mean(timers, NITER);

    /* Count unique column indices — approximates actual distinct x[] accesses */
    int *seen = (int *)calloc((size_t)cols, sizeof(int));
    int unique_cols = 0;
    for (int i = 0; i < nnz; i++)
        if (!seen[Acols[i]]) { seen[Acols[i]] = 1; unique_cols++; }
    free(seen);

    /* Bandwidth: COO arrays + unique x reads + y write */
    double bytes = (double)nnz         * (2 * sizeof(int) + sizeof(float))
                 + (double)unique_cols * sizeof(float)
                 + (double)rows        * sizeof(float);
    double bandwidth = bytes / a_mean / 1.e9;
    double gflops    = 2.0 * nnz / a_mean / 1.e9;

    fprintf(stdout, "\nMatrix:              %s\n", argv[1]);
    fprintf(stdout, "Rows: %d  Cols: %d  NNZ: %d\n", rows, cols, nnz);
    fprintf(stdout, " %20s | %15s | %15s |\n",
            "kernel", "arith mean (s)", "geom mean (s)");
    fprintf(stdout, " %20s | %15f | %15f |\n",
            "spmv_coo_opt", a_mean, g_mean);
    fprintf(stdout, "Effective bandwidth: %.4f GB/s\n", bandwidth);
    fprintf(stdout, "GFLOPS:              %.4f\n",       gflops);

    free(Arows); free(Acols); free(Avals);
    free(x); free(y); free(y_ref);
    return 0;
}
