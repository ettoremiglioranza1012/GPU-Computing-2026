/* mtx_io.h — Matrix Market (.mtx) file reader
 *
 * Provides one function:
 *   mtx_read_coo() — parse .mtx file into flat COO arrays (0-indexed, row-sorted)
 *
 * Handles:
 *   - real / integer / pattern value fields
 *   - general / symmetric storage (symmetric matrices are fully expanded)
 *   - arbitrary comment lines (lines starting with %)
 *
 * Output arrays are sorted row-major (primary: row asc, secondary: col asc),
 * which is required by the optimised COO kernel.
 */
#ifndef MTX_IO_H
#define MTX_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* Internal struct used only for sorting */
typedef struct { int r, c; float v; } MtxEntry_;

static int mtx_entry_cmp_(const void *a, const void *b) {
    const MtxEntry_ *ea = (const MtxEntry_ *)a;
    const MtxEntry_ *eb = (const MtxEntry_ *)b;
    if (ea->r != eb->r) return ea->r - eb->r;
    return ea->c - eb->c;
}

/* Read a Matrix Market file into flat COO arrays, sorted row-major.
 *
 * Outputs (all heap-allocated, caller must free):
 *   *out_rows, *out_cols  — matrix dimensions
 *   *out_nnz              — number of non-zeros (after symmetric expansion)
 *   *out_row_arr, *out_col_arr, *out_val_arr — COO triplets (0-indexed, sorted)
 */
static void mtx_read_coo(const char *path,
                         int *out_rows, int *out_cols, int *out_nnz,
                         int **out_row_arr, int **out_col_arr, float **out_val_arr)
{
    FILE *fp = fopen(path, "r");
    if (!fp) { perror("mtx_read_coo: fopen"); exit(1); }

    char line[512];
    int is_pattern   = 0;
    int is_symmetric = 0;

    /* Parse the %%MatrixMarket banner */
    if (fgets(line, sizeof(line), fp)) {
        for (char *p = line; *p; p++) *p = (char)tolower((unsigned char)*p);
        if (strstr(line, "pattern"))   is_pattern   = 1;
        if (strstr(line, "symmetric")) is_symmetric = 1;
    }

    /* Skip comment / blank lines */
    do {
        if (!fgets(line, sizeof(line), fp)) {
            fprintf(stderr, "mtx_read_coo: unexpected EOF before dimensions\n");
            exit(1);
        }
    } while (line[0] == '%');

    int nr, nc, nz_file;
    if (sscanf(line, "%d %d %d", &nr, &nc, &nz_file) != 3) {
        fprintf(stderr, "mtx_read_coo: failed to parse dimension line: %s\n", line);
        exit(1);
    }

    /* Worst-case: symmetric expands each off-diagonal entry */
    int capacity = is_symmetric ? 2 * nz_file : nz_file;
    MtxEntry_ *entries = (MtxEntry_ *)malloc((size_t)capacity * sizeof(MtxEntry_));
    if (!entries) { fprintf(stderr, "mtx_read_coo: malloc failed\n"); exit(1); }

    int count = 0;
    for (int i = 0; i < nz_file; i++) {
        if (!fgets(line, sizeof(line), fp)) break;

        int r, c;
        float v = 1.0f;

        if (is_pattern)
            sscanf(line, "%d %d", &r, &c);
        else
            sscanf(line, "%d %d %f", &r, &c, &v);

        r--; c--;  /* 1-indexed → 0-indexed */

        entries[count].r = r;
        entries[count].c = c;
        entries[count].v = v;
        count++;

        if (is_symmetric && r != c) {
            entries[count].r = c;
            entries[count].c = r;
            entries[count].v = v;
            count++;
        }
    }
    fclose(fp);

    /* Sort row-major — required by optimised COO kernel */
    qsort(entries, (size_t)count, sizeof(MtxEntry_), mtx_entry_cmp_);

    /* Unpack into flat arrays */
    int   *Arows = (int   *)malloc((size_t)count * sizeof(int));
    int   *Acols = (int   *)malloc((size_t)count * sizeof(int));
    float *Avals = (float *)malloc((size_t)count * sizeof(float));
    if (!Arows || !Acols || !Avals) {
        fprintf(stderr, "mtx_read_coo: malloc failed (count=%d)\n", count);
        exit(1);
    }
    for (int i = 0; i < count; i++) {
        Arows[i] = entries[i].r;
        Acols[i] = entries[i].c;
        Avals[i] = entries[i].v;
    }
    free(entries);

    *out_rows    = nr;
    *out_cols    = nc;
    *out_nnz     = count;
    *out_row_arr = Arows;
    *out_col_arr = Acols;
    *out_val_arr = Avals;
}

#endif /* MTX_IO_H */
