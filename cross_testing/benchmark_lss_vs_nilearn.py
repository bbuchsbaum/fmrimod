#!/usr/bin/env python
"""Benchmark fmrimod LSS against nilearn-style per-trial LSS."""

from __future__ import annotations

import argparse
import time
from typing import Callable, Tuple

import numpy as np

from fmrimod.single._project import build_nuisance_projector
from fmrimod.single.lss import lss_single_trial


def _extract_betas_from_run_glm(labels, estimates, n_regressors: int) -> np.ndarray:
    """Extract voxelwise beta matrix from nilearn run_glm output."""
    n_voxels = labels.shape[0]
    betas = np.empty((n_regressors, n_voxels), dtype=np.float64)
    for label, res in estimates.items():
        mask = labels == label
        theta = np.asarray(res.theta, dtype=np.float64)
        theta = np.atleast_2d(theta)
        if theta.shape[0] != n_regressors and theta.shape[1] == n_regressors:
            theta = theta.T
        betas[:, mask] = theta
    return betas


def _nilearn_lss(Y: np.ndarray, X: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    """Reference LSS via per-trial nilearn run_glm fits."""
    from nilearn.glm.first_level import run_glm

    n_trials = X.shape[1]
    n_voxels = Y.shape[1]
    out = np.empty((n_trials, n_voxels), dtype=np.float64)
    total = X.sum(axis=1, keepdims=True)

    for j in range(n_trials):
        target = X[:, [j]]
        others = total - target
        design = np.hstack([target, others, confounds])
        labels, estimates = run_glm(Y, design, noise_model="ols")
        out[j] = _extract_betas_from_run_glm(labels, estimates, design.shape[1])[0]

    return out


def _canonical_hrf_kernel(length: int = 24) -> np.ndarray:
    """Simple canonical-like HRF kernel sampled at TR resolution."""
    t = np.arange(length, dtype=np.float64)
    g1 = (t ** 5) * np.exp(-t / 1.0)
    g2 = (t ** 15) * np.exp(-t / 1.0)
    h = g1 - 0.5 * g2
    h /= np.max(np.abs(h))
    return h


def _build_trial_design(
    n_tp: int,
    n_trials: int,
    rng: np.random.Generator,
    min_gap: int = 2,
) -> np.ndarray:
    kernel = _canonical_hrf_kernel()
    onsets: list[int] = []
    attempts = 0
    while len(onsets) < n_trials and attempts < 200000:
        attempts += 1
        cand = int(rng.integers(0, n_tp - len(kernel)))
        if all(abs(cand - prev) >= min_gap for prev in onsets):
            onsets.append(cand)
    if len(onsets) < n_trials:
        raise RuntimeError(
            f"Could not place {n_trials} events with min_gap={min_gap} in {n_tp} TRs."
        )

    onsets.sort()
    X = np.zeros((n_tp, n_trials), dtype=np.float64)
    for j, onset in enumerate(onsets):
        X[onset : onset + len(kernel), j] = kernel
    return X


def _build_synthetic_data(
    *,
    n_tp: int,
    n_trials: int,
    n_voxels: int,
    n_confounds: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = _build_trial_design(n_tp, n_trials, rng=rng)
    confounds = rng.normal(size=(n_tp, n_confounds)).astype(np.float64)

    beta_true = rng.normal(scale=0.2, size=(n_trials, n_voxels))
    conf_true = rng.normal(scale=0.1, size=(n_confounds, n_voxels))
    noise = rng.normal(scale=1.0, size=(n_tp, n_voxels))
    Y = X @ beta_true + confounds @ conf_true + noise
    return Y, X, confounds


def _timeit(
    fn: Callable[[], np.ndarray], repeats: int, warmup: int
) -> Tuple[float, np.ndarray]:
    out = None
    for _ in range(warmup):
        out = fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(np.asarray(times, dtype=np.float64))), out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.ravel()
    bb = b.ravel()
    mask = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[mask]
    bb = bb[mask]
    if aa.size < 2:
        return 1.0
    sa = float(np.std(aa))
    sb = float(np.std(bb))
    if sa == 0.0 and sb == 0.0:
        return 1.0 if np.allclose(aa, bb) else 0.0
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-tp", type=int, default=300)
    p.add_argument("--n-trials", type=int, default=80)
    p.add_argument("--n-voxels", type=int, default=2500)
    p.add_argument("--n-confounds", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=4000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Y, X, confounds = _build_synthetic_data(
        n_tp=args.n_tp,
        n_trials=args.n_trials,
        n_voxels=args.n_voxels,
        n_confounds=args.n_confounds,
        seed=args.seed,
    )
    projector = build_nuisance_projector(confounds)
    if projector is None:
        raise RuntimeError("Failed to build nuisance projector.")

    t_ours_conf, b_ours_conf = _timeit(
        lambda: lss_single_trial(
            Y,
            X,
            confounds=confounds,
            chunk_size=args.chunk_size,
        ).betas,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    t_ours_cached, b_ours_cached = _timeit(
        lambda: lss_single_trial(
            Y,
            X,
            nuisance_projector=projector,
            chunk_size=args.chunk_size,
        ).betas,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    t_ref, b_ref = _timeit(
        lambda: _nilearn_lss(Y, X, confounds),
        repeats=args.repeats,
        warmup=args.warmup,
    )

    print(
        f"shape n_tp={args.n_tp} n_trials={args.n_trials} "
        f"n_voxels={args.n_voxels} n_confounds={args.n_confounds}"
    )
    print(
        f"ours_confounds={t_ours_conf:.3f}s "
        f"speedup_vs_nilearn={t_ref / max(t_ours_conf, 1e-9):.1f}x"
    )
    print(
        f"ours_cached_projector={t_ours_cached:.3f}s "
        f"speedup_vs_nilearn={t_ref / max(t_ours_cached, 1e-9):.1f}x "
        f"speedup_vs_ours_confounds={t_ours_conf / max(t_ours_cached, 1e-9):.2f}x"
    )
    print(f"nilearn_style_lss={t_ref:.3f}s")
    print(
        f"parity_corr_confounds={_corr(b_ours_conf, b_ref):.6f} "
        f"parity_mae_confounds={np.mean(np.abs(b_ours_conf - b_ref)):.6e}"
    )
    print(
        f"parity_corr_cached={_corr(b_ours_cached, b_ref):.6f} "
        f"parity_mae_cached={np.mean(np.abs(b_ours_cached - b_ref)):.6e}"
    )


if __name__ == "__main__":
    main()
