# Scheduling Strategy for Cluster Job Submission

## The cluster constraint

The Baldo cluster exposes a single queue for this course: `edu-short`.
Its hard wall-time limit is **5 minutes** (`00:05:00`).
Any `sbatch` submission that requests more than this is rejected at submission time with:

```
sbatch: error: Batch job submission failed: Requested time limit is invalid
(missing or exceeds some limit)
```

No job is queued, no output is produced. The error is silent about what the actual
limit is, which makes it easy to miss.

---

## What the reference student (2025) did

The 2025 student never used `#SBATCH --time` at all. By omitting it, SLURM uses
the partition default, which for `edu-short` is exactly 5 minutes — always within
the allowed range, so submissions never fail.

Their architecture was also inherently safe: `run.sh` is an **interactive menu**
that prompts for one kernel and one matrix, then fires a single `sbatch --wrap`
for that specific pair. Wall time per job = time to run one kernel on one matrix ≈
a few seconds. No job ever came close to the limit.

The trade-off is that running all kernels across all matrices required manual
repetition. There was no automated batch loop.

---

## Our initial implementation and its limits

We initially wrote two monolithic SLURM job scripts with explicit time requests:

| Script | Requested time | Partition limit | Result |
|---|---|---|---|
| `batch_gpu.sh` | `00:10:00` | `00:05:00` | rejected at submission |
| `sweep_gpu.sh` | `00:15:00` | `00:05:00` | rejected at submission |

Both scripts used `#SBATCH` headers at the top, making them SLURM jobs themselves.
They looped over all kernels and all matrices **sequentially inside a single job**:

```
job starts
  for each kernel:
    for each matrix:
      run binary   ← all of this happens inside one 5-min window
job ends
```

Wall time grew as **O(K × M × T)** where K = number of kernels, M = number of
matrices, T = time per run. With a small development dataset this was borderline;
with large matrices (10M+ NNZ) it would have failed even if the time limit were
raised, since a single SpMV timing run on a large matrix takes meaningfully longer.

The sweep had the same problem at a larger scale: 7 block counts × 5 thread counts
= **35 configurations** all running sequentially inside one job. With `--time=00:15:00`
it was already rejected before even attempting to run.

---

## The solution: one job per (config/kernel, matrix) pair

Both scripts were restructured as **launcher scripts**: plain bash, no `#SBATCH`
headers, not submitted to SLURM themselves. They loop over all combinations and
fire one independent `sbatch --wrap` per pair:

```
bash scripts/batch_gpu.sh   ← runs locally, returns immediately
  sbatch --wrap "... spmv_gpu_tpv   matrix_A ..."   → job A (5 min window)
  sbatch --wrap "... spmv_gpu_tpv   matrix_B ..."   → job B (5 min window)
  sbatch --wrap "... spmv_gpu_tpr   matrix_A ..."   → job C (5 min window)
  ...
```

Each job does exactly one thing — one kernel on one matrix — so wall time is
O(T) regardless of how many kernels or matrices exist in the repo. No `--time`
flag is passed; the partition default (5 min) applies and is always sufficient.

For the sweep:

```
bash scripts/sweep_gpu.sh
  sbatch --wrap "... spmv_gpu_stride_b1_t32   bcsstk17 ..."   → job 1
  sbatch --wrap "... spmv_gpu_stride_b1_t128  bcsstk17 ..."   → job 2
  ...                                                           → jobs 3–35
```

35 jobs run **in parallel** on the cluster rather than sequentially inside one job.
Total elapsed time ≈ time for one configuration, not 35×.

### Scaling properties

| Scenario | Monolithic job | Launcher pattern |
|---|---|---|
| 3 kernels × 5 matrices | ~minutes, risky | 15 jobs × ~seconds each |
| 3 kernels × 20 matrices (large) | exceeds 5 min limit | 60 jobs × ~seconds each |
| 35 sweep configs | exceeds 5 min limit | 35 jobs × ~seconds each |
| Add new matrix | wall time grows | one new job, no change elsewhere |

### Output file naming

Each job writes its own output file. The names are structured so that
`parse_results.py` can glob them cleanly:

| Script | Glob pattern |
|---|---|
| `batch_gpu.sh` | `outputs/R-spmv_gpu_<kernel>_<matrix>.<jobid>.txt` |
| `sweep_gpu.sh` | `outputs/R-spmv_sweep_b<N>_t<M>.<jobid>.txt` |

The `KERNEL: ... MATRIX: ...` banner inside each output file is emitted by the
`--wrap` command itself, preserving the format that `parse_results.py` expects.

---

## Post-sweep best-config step

The original monolithic `sweep_gpu.sh` tracked the best bandwidth inline during
its sequential loop and wrote `scripts/best_gpu_config.sh` at the end. With async
jobs this is no longer possible — results arrive at different times.

The best config is now determined after all sweep jobs complete, by running
`parse_results.py` + `plot_sweep.py`. `plot_sweep.py` prints the winning
(blocks, threads) pair; the user writes it to `scripts/best_gpu_config.sh`
manually. `batch_gpu.sh` sources this file automatically if it exists, and falls
back to the stride kernel's built-in defaults if it does not.
