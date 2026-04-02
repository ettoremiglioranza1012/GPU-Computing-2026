/* spmv_coo_naive.c — CPU SpMV baseline using COO format
 *
 * Algorithm: iterate over all NNZ, accumulate y[row] += val * x[col].
 * Single pass, no unrolling. Performance floor for all implementations.
 *
 * Usage: ./bin/CPU/spmv_coo_naive path/to/matrix.mtx
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "my_time_lib.h"
#include "mtx_io.h"

#define WARMUP 2
#define NITER  50

static void spmv_coo_naive(const int *Arows, const int *Acols,
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

    /* --- Benchmark loop --- */
    double timers[NITER];
    TIMER_DEF(0);

    for (int iter = -WARMUP; iter < NITER; iter++) {
        memset(y, 0, (size_t)rows * sizeof(float));

        TIMER_START(0);
        spmv_coo_naive(Arows, Acols, Avals, nnz, x, y);
        TIMER_STOP(0);

        double t = TIMER_ELAPSED(0) / 1.e6;
        if (iter >= 0) timers[iter] = t;

        if (iter == -WARMUP)
            memcpy(y_ref, y, (size_t)rows * sizeof(float));

        fprintf(stdout, "Iteration %d: %f s\n", iter, t);
    }

    /* --- Correctness: last result must match first warm-up (identical runs) --- */
    int ok = 1;
    for (int i = 0; i < rows && ok; i++)
        if (fabsf(y[i] - y_ref[i]) > 1e-6f) ok = 0;
    fprintf(stdout, "Correctness check: %s\n", ok ? "PASSED" : "FAILED");
    if (!ok) fprintf(stderr, "Correctness check: FAILED\n");

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
            "spmv_coo_naive", a_mean, g_mean);
    fprintf(stdout, "Effective bandwidth: %.4f GB/s\n", bandwidth);
    fprintf(stdout, "GFLOPS:              %.4f\n",       gflops);

    free(Arows); free(Acols); free(Avals);
    free(x); free(y); free(y_ref);
    return 0;
}
