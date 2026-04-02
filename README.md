# GPU Computing 2026 — Deliverable 1

Sparse Matrix-Vector Multiplication (SpMV) benchmarked on CPU and GPU using COO format.

---

## Make commands

| Command | What it does |
|---------|-------------|
| `make` / `make all` | Build all CPU and GPU kernels |
| `make cpu` | Build CPU kernels only (`bin/CPU/`) |
| `make gpu` | Build GPU kernels only (`bin/GPU/`) — requires CUDA module loaded |
| `make data` | Download all test matrices from SuiteSparse into `Data/` |
| `make clean_bin` | Delete compiled binaries and object files (`bin/`, `TIMER_LIB/obj/`) |
| `make clean_outputs` | Delete SLURM output files (`outputs/*.txt`, `*.err`) |
| `make clean_results` | Delete parsed CSVs (`results_tables/`) and plots (`assets/`) |
| `make clean` | Delete **everything**: binaries + outputs + results + `Data/` |
| `make help` | Print this command reference in the terminal |

---

## Full pipeline from scratch

Steps 1–8 run on the **Baldo cluster**. Steps 9–11 run on your **local machine**.

### 1. Clone and enter

```bash
git clone <your-repo-url>
cd GPU-computing-2026
```

### 2. Download matrices

```bash
make data
```

Downloads the test matrices from SuiteSparse into `Data/`. Also creates the
`outputs/`, `results_tables/`, and `assets/` directories.

### 3. Load CUDA and compile

```bash
module load CUDA/12.1.1
make all
```

Builds all CPU kernels into `bin/CPU/` and all GPU kernels into `bin/GPU/`.

### 4. Run the block/thread configuration sweep (GPU stride kernel)

```bash
bash scripts/sweep_gpu.sh
```

Submits 35 independent SLURM jobs (7 block counts × 5 thread counts) to `edu-short`,
one job per configuration. Each job runs `spmv_gpu_stride` on `bcsstk17`.
Monitor with:

```bash
squeue -u $USER
```

When all jobs finish, parse and plot the sweep results:

```bash
uv run scripts/parse_results.py outputs/R-spmv_sweep_*.txt --out results_tables/sweep.csv
uv run scripts/plot_sweep.py results_tables/sweep.csv
```

`plot_sweep.py` prints the best config (highest bandwidth). Write it to
`scripts/best_gpu_config.sh` so that `batch_gpu.sh` picks it up automatically:

```bash
cat > scripts/best_gpu_config.sh << EOF
# Auto-generated — do not edit by hand.
BEST_BLOCKS=<N>
BEST_THREADS=<M>
EOF
```

### 5. Run CPU benchmarks

```bash
sbatch scripts/batch_cpu.sh
```

Runs both CPU kernels across all matrices. Output: `outputs/R-spmv_cpu.<jobid>.txt`.

### 6. Run GPU benchmarks

```bash
bash scripts/batch_gpu.sh
```

Submits one SLURM job per (kernel, matrix) pair — 3 kernels × number of matrices.
The stride kernel automatically uses the best config from step 4 if
`scripts/best_gpu_config.sh` exists, otherwise falls back to built-in defaults.
Output files: `outputs/R-spmv_gpu_<kernel>_<matrix>.<jobid>.txt`.

### 7. Push results to remote

```bash
git add outputs/ scripts/best_gpu_config.sh
git commit -m "add benchmark results"
git push
```

### 8. Pull on your local machine

```bash
git pull
```

### 9. Parse all results into CSVs

```bash
# CPU + GPU kernels → one unified CSV
uv run scripts/parse_results.py outputs/R-spmv_cpu.*.txt outputs/R-spmv_gpu_*.txt

# Sweep results → separate CSV
uv run scripts/parse_results.py outputs/R-spmv_sweep_*.txt \
    --out results_tables/results_sweep.csv
```

Both commands default to `results_tables/results.csv`. Pass `--out` to override.

### 10. Generate plots

```bash
# CPU and GPU performance plots → assets/
uv run scripts/plot_results.py results_tables/results.csv --out assets/

# Block/thread sweep heatmap → assets/gpu_config_sweep.png
uv run scripts/plot_sweep.py results_tables/results_sweep.csv --out assets/
```

Produces in `assets/`:

| File | Content |
|------|---------|
| `cpu_bandwidth.png` | Bandwidth (GB/s) per CPU kernel × matrix |
| `cpu_gflops.png` | GFLOPS per CPU kernel × matrix |
| `cpu_speedup.png` | Speedup of optimised over naive per matrix |
| `gpu_bandwidth.png` | Bandwidth (GB/s) per GPU kernel × matrix |
| `gpu_gflops.png` | GFLOPS per GPU kernel × matrix |
| `gpu_vs_cpu.png` | GPU speedup over CPU naive per matrix |
| `gpu_config_sweep.png` | Heatmap: bandwidth vs blocks × threads/block |

### 11. Clean up

```bash
make clean_results   # wipe CSVs and plots only (keep binaries and data)
make clean           # wipe absolutely everything — start from scratch
```
