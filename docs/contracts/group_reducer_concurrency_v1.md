# Group Reducer Concurrency Contract v1

Status: Draft implementation contract for native `fmrimod.group` hot reducers.

## Scope

The first native acceleration path uses deterministic feature-axis chunking.
It is a conservative reference implementation, not a claim that the native
Python reducers now match compiled fmrigds OpenMP throughput.

## Public Options

Hot voxelwise reducers accept:

- `n_jobs`: number of feature chunks to run concurrently; default `1`.
- `chunk_size`: optional feature count per chunk; default divides features
  across `n_jobs`.
- `blas_threads`: optional per-worker BLAS thread limit when `threadpoolctl` is
  installed.

The initial covered reducers are:

- `perm_onesample`
- `perm_twosample`
- `ols_voxelwise`

All options preserve the existing float64 result contract. Parallel and serial
execution must match within numeric equality tolerances, including NaN
placement.

## Oversubscription Policy

When `n_jobs > 1`, callers should set `blas_threads=1` for matrix-heavy work
unless they have an external thread policy. If `threadpoolctl` is unavailable,
the reducer still runs but cannot enforce BLAS thread limits.

## Compiled fmrigds Benchmark Gate

The benchmark harness is
`benchmarks/performance/group_reducer_fmrigds.py`. It compares the native
Python `meta_fe` and `perm_onesample` paths against the compiled fmrigds
`meta_fe_cpp` and `perm_onesample_t_cpp` kernels when R/fmrigds are available.

Reference local run on 2026-05-13:

```sh
uv run python benchmarks/performance/group_reducer_fmrigds.py \
  --features 5000 --subjects 24 --permutations 512 \
  --repeat 3 --warmup 1 --r-inner 20 \
  --n-jobs 4 --chunk-size 512 --blas-threads 1
```

Observed minimum timings:

- `python_meta_fe_s`: `0.001314667`
- `r_meta_fe_s`: `0.000600000`
- `meta_fe_slowdown_vs_fmrigds`: `2.19`
- `python_perm_onesample_serial_s`: `0.359389833`
- `python_perm_onesample_parallel_s`: `0.656098250`
- `r_perm_onesample_s`: `0.130850000`
- `perm_serial_slowdown_vs_fmrigds`: `2.75`
- `perm_parallel_slowdown_vs_fmrigds`: `5.01`

Policy decision: the NumPy reference path is acceptable for this phase because
the measured serial hot reducer slowdown is under `3x` against compiled
fmrigds on this representative fixture, not the earlier feared `10-100x`.
Threaded chunking is retained as an explicit option, but this fixture shows
thread overhead can dominate; callers should benchmark their workload before
using `n_jobs > 1`.

Future production-hardening may still add Numba or C++ kernels, especially for
larger permutation grids, but that is no longer a blocker for the native
reference reducer surface.
