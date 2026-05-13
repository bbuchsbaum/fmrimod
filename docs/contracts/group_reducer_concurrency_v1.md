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

## Remaining Performance Gate

This contract does not close the compiled-kernel performance question by
itself. Before marking the acceleration milestone production-complete, add a
representative benchmark against fmrigds compiled reducers and document the
accepted slowdown bounds or the chosen native kernel strategy.
