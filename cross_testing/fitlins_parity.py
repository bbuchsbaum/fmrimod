"""Parity and performance harness for fmrimod vs fitlins-aligned GLM.

This module defines:
1. A reproducible synthetic GLM data generator.
2. Two matched OLS fitting paths:
   - `fmrimod` low-level solver.
   - nilearn GLM path used by fitlins' nistats estimator.
3. A parity metric contract and benchmark runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from fmrimod.glm.contrasts import contrast_t
from fmrimod.glm.solver import fast_lm_matrix, fast_preproject


Array = NDArray[np.float64]


@dataclass(frozen=True)
class ParityThresholds:
    """Pass/fail thresholds for basic first-level OLS parity."""

    min_beta_corr: float = 0.999
    max_beta_mae: float = 5e-4
    max_beta_abs: float = 5e-2
    min_sigma2_corr: float = 0.999
    max_sigma2_mae: float = 1e-3
    min_t_corr: float = 0.995
    max_t_mae: float = 5e-3
    max_t_abs: float = 2.5e-1
    max_p_mae: float = 1e-2
    max_sign_flip_rate: float = 1e-3
    sign_flip_floor: float = 1e-8


@dataclass(frozen=True)
class SpeedThresholds:
    """Speed gate for benchmark runs."""

    min_speedup_vs_reference: float = 1.0


@dataclass(frozen=True)
class ParityMetrics:
    """Measured parity statistics between candidate and reference outputs."""

    beta_corr: float
    beta_mae: float
    beta_max_abs: float
    sigma2_corr: float
    sigma2_mae: float
    t_corr: float
    t_mae: float
    t_max_abs: float
    p_mae: float
    sign_flip_rate: float

    def failures(self, thresholds: ParityThresholds) -> list[str]:
        """Return a list of threshold names that failed."""
        failed: list[str] = []
        if self.beta_corr < thresholds.min_beta_corr:
            failed.append("min_beta_corr")
        if self.beta_mae > thresholds.max_beta_mae:
            failed.append("max_beta_mae")
        if self.beta_max_abs > thresholds.max_beta_abs:
            failed.append("max_beta_abs")
        if self.sigma2_corr < thresholds.min_sigma2_corr:
            failed.append("min_sigma2_corr")
        if self.sigma2_mae > thresholds.max_sigma2_mae:
            failed.append("max_sigma2_mae")
        if self.t_corr < thresholds.min_t_corr:
            failed.append("min_t_corr")
        if self.t_mae > thresholds.max_t_mae:
            failed.append("max_t_mae")
        if self.t_max_abs > thresholds.max_t_abs:
            failed.append("max_t_abs")
        if self.p_mae > thresholds.max_p_mae:
            failed.append("max_p_mae")
        if self.sign_flip_rate > thresholds.max_sign_flip_rate:
            failed.append("max_sign_flip_rate")
        return failed

    def passes(self, thresholds: ParityThresholds) -> bool:
        """Return True when all thresholds pass."""
        return not self.failures(thresholds)


@dataclass(frozen=True)
class BenchmarkSummary:
    """Timing summary for matched candidate/reference runs."""

    fmrimod_runs_s: tuple[float, ...]
    reference_runs_s: tuple[float, ...]
    fmrimod_median_s: float
    reference_median_s: float
    speedup_vs_reference: float


DEFAULT_PARITY_THRESHOLDS = ParityThresholds()
DEFAULT_SPEED_THRESHOLDS = SpeedThresholds()


def make_synthetic_glm(
    *,
    n_timepoints: int = 240,
    n_regressors: int = 8,
    n_voxels: int = 2000,
    noise_sd: float = 1.0,
    seed: int = 1234,
) -> tuple[Array, Array, Array, Array]:
    """Create a reproducible full-rank synthetic GLM dataset.

    Returns
    -------
    X, Y, beta_true, contrast
    """
    if n_regressors < 2:
        raise ValueError("n_regressors must be >= 2")
    if n_timepoints <= n_regressors:
        raise ValueError("n_timepoints must be greater than n_regressors")
    if n_voxels < 1:
        raise ValueError("n_voxels must be >= 1")

    rng = np.random.default_rng(seed)
    X = np.empty((n_timepoints, n_regressors), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1:] = rng.standard_normal((n_timepoints, n_regressors - 1))

    # Standardize non-intercept columns for numerical stability.
    means = X[:, 1:].mean(axis=0)
    stds = X[:, 1:].std(axis=0)
    stds = np.where(stds > 0.0, stds, 1.0)
    X[:, 1:] = (X[:, 1:] - means) / stds

    # Add tiny jitter if needed to avoid accidental rank deficiency.
    while np.linalg.matrix_rank(X) < n_regressors:
        X[:, 1:] += rng.normal(scale=1e-6, size=X[:, 1:].shape)

    beta_true = rng.normal(loc=0.0, scale=0.2, size=(n_regressors, n_voxels))
    noise = rng.normal(loc=0.0, scale=noise_sd, size=(n_timepoints, n_voxels))
    Y = X @ beta_true + noise

    contrast = np.zeros(n_regressors, dtype=np.float64)
    contrast[1] = 1.0
    return X, Y, beta_true, contrast


def fit_fmrimod_ols(
    X: Array,
    Y: Array,
    contrast: Array,
    *,
    compute_dtype: object = np.float64,
) -> Dict[str, Array]:
    """Fit OLS with fmrimod's matrix solver and return comparable outputs."""
    proj = fast_preproject(X, compute_dtype=compute_dtype)
    fit = fast_lm_matrix(X, Y, proj, return_fitted=False, compute_dtype=compute_dtype)
    sigma = np.sqrt(np.maximum(fit.sigma2, 0.0))
    cres = contrast_t(contrast, fit.betas, proj.XtXinv, sigma, fit.dfres, name="main")
    return {
        "betas": fit.betas,
        "sigma2": fit.sigma2,
        "t": np.asarray(cres.stat, dtype=np.float64).reshape(-1),
        "p": np.asarray(cres.p_value, dtype=np.float64).reshape(-1),
    }


def fit_fitlins_reference_ols(X: Array, Y: Array, contrast: Array) -> Dict[str, Array]:
    """Fit OLS via nilearn's GLM path used by fitlins nistats estimator."""
    from nilearn.glm.contrasts import compute_contrast
    from nilearn.glm.first_level import run_glm

    labels, estimates = run_glm(Y, X, noise_model="ols")
    try:
        # Older nilearn/nistats APIs.
        cres = compute_contrast(labels, estimates, contrast, contrast_type="t")
    except TypeError:
        # Current nilearn API.
        cres = compute_contrast(labels, estimates, contrast, stat_type="t")

    n_regressors = X.shape[1]
    n_voxels = Y.shape[1]
    betas = np.empty((n_regressors, n_voxels), dtype=np.float64)
    sigma2 = np.empty(n_voxels, dtype=np.float64)

    for label, res in estimates.items():
        mask = labels == label
        theta = np.asarray(res.theta, dtype=np.float64)
        theta = np.atleast_2d(theta)
        if theta.shape[0] != n_regressors and theta.shape[1] == n_regressors:
            theta = theta.T
        betas[:, mask] = theta

        dispersion = np.asarray(res.dispersion, dtype=np.float64).reshape(-1)
        if dispersion.size == 1:
            sigma2[mask] = dispersion[0]
        else:
            sigma2[mask] = dispersion

    t_vals = np.asarray(cres.stat(), dtype=np.float64).reshape(-1)
    p_raw = np.asarray(cres.p_value(), dtype=np.float64).reshape(-1)

    # nilearn variants differ in whether p-values are one- or two-sided.
    # Align to two-sided p-values to match fmrimod contrast_t semantics.
    dfres = float(X.shape[0] - np.linalg.matrix_rank(X))
    p_expected_two_sided = 2.0 * sp_stats.t.sf(np.abs(t_vals), dfres)
    p_folded_two_sided = np.clip(2.0 * np.minimum(p_raw, 1.0 - p_raw), 0.0, 1.0)

    mae_raw = float(np.mean(np.abs(p_raw - p_expected_two_sided)))
    mae_folded = float(np.mean(np.abs(p_folded_two_sided - p_expected_two_sided)))
    p_vals = p_raw if mae_raw <= mae_folded else p_folded_two_sided

    return {
        "betas": betas,
        "sigma2": sigma2,
        "t": t_vals,
        "p": p_vals,
    }


def _flatten_finite(a: Array, b: Array) -> tuple[Array, Array]:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(aa) & np.isfinite(bb)
    return aa[mask], bb[mask]


def _safe_corr(a: Array, b: Array) -> float:
    aa, bb = _flatten_finite(a, b)
    if aa.size < 2:
        return 1.0
    a_std = float(np.std(aa))
    b_std = float(np.std(bb))
    if a_std == 0.0 and b_std == 0.0:
        return 1.0 if np.allclose(aa, bb) else 0.0
    if a_std == 0.0 or b_std == 0.0:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def compute_parity_metrics(
    candidate: Mapping[str, Array],
    reference: Mapping[str, Array],
    *,
    sign_flip_floor: float = DEFAULT_PARITY_THRESHOLDS.sign_flip_floor,
) -> ParityMetrics:
    """Compute parity metrics across matched GLM outputs."""
    cand_beta, ref_beta = _flatten_finite(candidate["betas"], reference["betas"])
    cand_sigma2, ref_sigma2 = _flatten_finite(candidate["sigma2"], reference["sigma2"])
    cand_t, ref_t = _flatten_finite(candidate["t"], reference["t"])
    cand_p, ref_p = _flatten_finite(candidate["p"], reference["p"])

    beta_abs = np.abs(cand_beta - ref_beta)
    sigma2_abs = np.abs(cand_sigma2 - ref_sigma2)
    t_abs = np.abs(cand_t - ref_t)
    p_abs = np.abs(cand_p - ref_p)

    informative = (np.abs(cand_t) > sign_flip_floor) | (np.abs(ref_t) > sign_flip_floor)
    if informative.any():
        sign_flip_rate = float(
            np.mean(np.sign(cand_t[informative]) != np.sign(ref_t[informative]))
        )
    else:
        sign_flip_rate = 0.0

    return ParityMetrics(
        beta_corr=_safe_corr(candidate["betas"], reference["betas"]),
        beta_mae=float(beta_abs.mean()) if beta_abs.size else 0.0,
        beta_max_abs=float(beta_abs.max()) if beta_abs.size else 0.0,
        sigma2_corr=_safe_corr(candidate["sigma2"], reference["sigma2"]),
        sigma2_mae=float(sigma2_abs.mean()) if sigma2_abs.size else 0.0,
        t_corr=_safe_corr(candidate["t"], reference["t"]),
        t_mae=float(t_abs.mean()) if t_abs.size else 0.0,
        t_max_abs=float(t_abs.max()) if t_abs.size else 0.0,
        p_mae=float(p_abs.mean()) if p_abs.size else 0.0,
        sign_flip_rate=sign_flip_rate,
    )


def benchmark_implementations(
    X: Array,
    Y: Array,
    contrast: Array,
    *,
    repeats: int = 5,
    warmup: int = 1,
    fmrimod_compute_dtype: object = np.float64,
) -> BenchmarkSummary:
    """Benchmark matched OLS runs for fmrimod and fitlins reference."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    X_candidate = X.astype(fmrimod_compute_dtype, copy=False)
    Y_candidate = Y.astype(fmrimod_compute_dtype, copy=False)

    for _ in range(warmup):
        fit_fmrimod_ols(
            X_candidate,
            Y_candidate,
            contrast,
            compute_dtype=fmrimod_compute_dtype,
        )
        fit_fitlins_reference_ols(X, Y, contrast)

    fmrimod_runs: list[float] = []
    reference_runs: list[float] = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        fit_fmrimod_ols(
            X_candidate,
            Y_candidate,
            contrast,
            compute_dtype=fmrimod_compute_dtype,
        )
        fmrimod_runs.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        fit_fitlins_reference_ols(X, Y, contrast)
        reference_runs.append(time.perf_counter() - t0)

    fmrimod_median = float(np.median(fmrimod_runs))
    reference_median = float(np.median(reference_runs))
    speedup = float(reference_median / fmrimod_median)

    return BenchmarkSummary(
        fmrimod_runs_s=tuple(fmrimod_runs),
        reference_runs_s=tuple(reference_runs),
        fmrimod_median_s=fmrimod_median,
        reference_median_s=reference_median,
        speedup_vs_reference=speedup,
    )


def run_parity_and_benchmark(
    *,
    n_timepoints: int = 240,
    n_regressors: int = 8,
    n_voxels: int = 2000,
    noise_sd: float = 1.0,
    seed: int = 1234,
    repeats: int = 5,
    warmup: int = 1,
    fmrimod_compute_dtype: object = np.float64,
    parity_thresholds: ParityThresholds = DEFAULT_PARITY_THRESHOLDS,
    speed_thresholds: SpeedThresholds = DEFAULT_SPEED_THRESHOLDS,
) -> Dict[str, Any]:
    """Run one full parity + speed evaluation and return a JSON-safe report."""
    X, Y, _beta_true, contrast = make_synthetic_glm(
        n_timepoints=n_timepoints,
        n_regressors=n_regressors,
        n_voxels=n_voxels,
        noise_sd=noise_sd,
        seed=seed,
    )

    X_candidate = X.astype(fmrimod_compute_dtype)
    Y_candidate = Y.astype(fmrimod_compute_dtype)
    candidate = fit_fmrimod_ols(
        X_candidate,
        Y_candidate,
        contrast,
        compute_dtype=fmrimod_compute_dtype,
    )
    reference = fit_fitlins_reference_ols(X, Y, contrast)

    parity = compute_parity_metrics(
        candidate,
        reference,
        sign_flip_floor=parity_thresholds.sign_flip_floor,
    )
    parity_failures = parity.failures(parity_thresholds)
    parity_ok = not parity_failures

    bench = benchmark_implementations(
        X,
        Y,
        contrast,
        repeats=repeats,
        warmup=warmup,
        fmrimod_compute_dtype=fmrimod_compute_dtype,
    )
    speed_ok = bench.speedup_vs_reference >= speed_thresholds.min_speedup_vs_reference

    return {
        "config": {
            "n_timepoints": n_timepoints,
            "n_regressors": n_regressors,
            "n_voxels": n_voxels,
            "noise_sd": noise_sd,
            "seed": seed,
            "repeats": repeats,
            "warmup": warmup,
            "fmrimod_compute_dtype": str(np.dtype(fmrimod_compute_dtype)),
        },
        "thresholds": {
            "parity": asdict(parity_thresholds),
            "speed": asdict(speed_thresholds),
        },
        "parity": {
            "ok": parity_ok,
            "failures": parity_failures,
            "metrics": asdict(parity),
        },
        "speed": {
            "ok": speed_ok,
            "summary": asdict(bench),
        },
    }


def write_json_report(report: Mapping[str, Any], output_path: str | Path) -> None:
    """Write report payload to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fobj:
        json.dump(report, fobj, indent=2, sort_keys=True)
