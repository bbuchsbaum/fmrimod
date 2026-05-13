"""Parity and performance harness for fmrimod vs fitlins-aligned AR(1) GLM.

This module mirrors the OLS parity harness but targets AR(1) noise
modeling against nilearn's ``run_glm(..., noise_model='ar1')`` path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy import special as sp_special
from scipy import stats as sp_stats

from cross_testing.fitlins_parity import (
    BenchmarkSummary,
    ParityMetrics,
    ParityThresholds,
    SpeedThresholds,
    compute_parity_metrics,
)
from fmrimod.ar.estimation import estimate_ar
from fmrimod.ar.whitening import ar_whiten_matrix
from fmrimod.glm.solver import fast_lm_matrix, fast_preproject


Array = NDArray[np.float64]


DEFAULT_AR1_PARITY_THRESHOLDS = ParityThresholds(
    min_beta_corr=0.995,
    max_beta_mae=1.5e-2,
    max_beta_abs=2.5e-1,
    min_sigma2_corr=0.995,
    max_sigma2_mae=2.0e-2,
    min_t_corr=0.995,
    max_t_mae=8.0e-2,
    max_t_abs=1.0,
    max_p_mae=3.5e-2,
    max_sign_flip_rate=1.0e-2,
    sign_flip_floor=1e-8,
)
DEFAULT_AR1_SPEED_THRESHOLDS = SpeedThresholds(min_speedup_vs_reference=1.0)


@dataclass(frozen=True)
class AR1CandidateConfig:
    """Generic AR(1) candidate fitting configuration for parity fixtures.

    ``coefficient_bin_width`` groups voxelwise AR(1) coefficients before
    whitening/refitting. This mirrors binned-label AR workflows without naming
    any reference implementation in fmrimod's library surface.
    """

    iter_gls: int = 1
    voxelwise: bool = False
    coefficient_bin_width: float | None = None
    exact_first_ar1: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.iter_gls, bool)
            or not isinstance(self.iter_gls, int)
            or self.iter_gls < 1
        ):
            raise ValueError("iter_gls must be an integer >= 1")
        if not isinstance(self.voxelwise, bool):
            raise ValueError("voxelwise must be a boolean")
        if not isinstance(self.exact_first_ar1, bool):
            raise ValueError("exact_first_ar1 must be a boolean")
        if self.coefficient_bin_width is not None:
            width = float(self.coefficient_bin_width)
            if not np.isfinite(width) or width <= 0.0:
                raise ValueError("coefficient_bin_width must be positive")
            if not self.voxelwise:
                raise ValueError(
                    "coefficient_bin_width requires voxelwise AR coefficients"
                )


def bin_ar1_coefficients(phi: Array, bin_width: float | None) -> Array:
    """Bin AR(1) coefficients toward zero with a fixed bin width."""

    phi_arr = np.asarray(phi, dtype=np.float64)
    if bin_width is None:
        return phi_arr.copy()
    width = float(bin_width)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("bin_width must be positive")
    return np.trunc(phi_arr / width) * width


def make_synthetic_glm_ar1(
    *,
    n_timepoints: int = 260,
    n_regressors: int = 10,
    n_voxels: int = 3000,
    phi: float = 0.45,
    noise_sd: float = 1.0,
    seed: int = 1234,
) -> tuple[Array, Array, Array, Array]:
    """Create a reproducible synthetic AR(1) GLM dataset."""
    if n_regressors < 2:
        raise ValueError("n_regressors must be >= 2")
    if n_timepoints <= n_regressors:
        raise ValueError("n_timepoints must be greater than n_regressors")
    if n_voxels < 1:
        raise ValueError("n_voxels must be >= 1")
    if not np.isfinite(phi) or abs(phi) >= 1.0:
        raise ValueError("phi must be finite with abs(phi) < 1")

    rng = np.random.default_rng(seed)
    X = np.empty((n_timepoints, n_regressors), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1:] = rng.standard_normal((n_timepoints, n_regressors - 1))
    X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(axis=0)) / np.where(
        X[:, 1:].std(axis=0) > 0.0, X[:, 1:].std(axis=0), 1.0
    )
    while np.linalg.matrix_rank(X) < n_regressors:
        X[:, 1:] += rng.normal(scale=1e-6, size=X[:, 1:].shape)

    beta_true = rng.normal(loc=0.0, scale=0.2, size=(n_regressors, n_voxels))
    innovations = rng.normal(
        loc=0.0, scale=noise_sd, size=(n_timepoints, n_voxels)
    ).astype(np.float64)
    noise = np.zeros_like(innovations)
    noise[0] = innovations[0]
    for t in range(1, n_timepoints):
        noise[t] = phi * noise[t - 1] + innovations[t]

    Y = X @ beta_true + noise
    contrast = np.zeros(n_regressors, dtype=np.float64)
    contrast[1] = 1.0
    return X, Y, beta_true, contrast


def _extract_glm_outputs(
    labels: NDArray,
    estimates: Mapping[Any, Any],
    n_regressors: int,
) -> tuple[Array, Array]:
    """Extract betas and dispersion arrays from nilearn GLM outputs."""
    n_voxels = labels.shape[0]
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
        sigma2[mask] = dispersion[0] if dispersion.size == 1 else dispersion

    return betas, sigma2


def fit_fitlins_reference_ar1(X: Array, Y: Array, contrast: Array) -> Dict[str, Array]:
    """Fit AR(1) via nilearn path used by fitlins-style workflows."""
    from nilearn.glm.contrasts import compute_contrast
    from nilearn.glm.first_level import run_glm

    labels, estimates = run_glm(Y, X, noise_model="ar1")
    try:
        cres = compute_contrast(labels, estimates, contrast, contrast_type="t")
    except TypeError:
        cres = compute_contrast(labels, estimates, contrast, stat_type="t")

    betas, sigma2 = _extract_glm_outputs(labels, estimates, X.shape[1])
    t_vals = np.asarray(cres.stat(), dtype=np.float64).reshape(-1)
    p_raw = np.asarray(cres.p_value(), dtype=np.float64).reshape(-1)

    # Handle nilearn one-sided/two-sided p-value variant behavior.
    dfres_approx = float(X.shape[0] - np.linalg.matrix_rank(X))
    p_expected_two_sided = 2.0 * sp_stats.t.sf(np.abs(t_vals), dfres_approx)
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


def fit_fmrimod_ar1(
    X: Array,
    Y: Array,
    contrast: Array,
    *,
    iter_gls: int = 1,
    voxelwise: bool = False,
    coefficient_bin_width: float | None = None,
    exact_first_ar1: bool = False,
    config: AR1CandidateConfig | None = None,
) -> Dict[str, Array]:
    """Fit AR(1) with fmrimod's AR estimation + whitening path."""
    if config is None:
        config = AR1CandidateConfig(
            iter_gls=iter_gls,
            voxelwise=voxelwise,
            coefficient_bin_width=coefficient_bin_width,
            exact_first_ar1=exact_first_ar1,
        )
    else:
        iter_gls = config.iter_gls
        voxelwise = config.voxelwise
        coefficient_bin_width = config.coefficient_bin_width
        exact_first_ar1 = config.exact_first_ar1

    X_base = np.asarray(X, dtype=np.float64)
    Y_base = np.asarray(Y, dtype=np.float64)

    if iter_gls < 1:
        raise ValueError("iter_gls must be >= 1")

    con = np.asarray(contrast, dtype=np.float64).ravel()

    def finish_fit(X_w: Array, Y_w: Array) -> Dict[str, Array]:
        proj = fast_preproject(X_w, check_finite=False)
        fit = fast_lm_matrix(
            X_w,
            Y_w,
            proj,
            return_fitted=False,
            check_finite=False,
        )
        estimate = con @ np.asarray(fit.betas, dtype=np.float64)
        var_factor = float(con @ np.asarray(proj.XtXinv, dtype=np.float64) @ con)
        variance = np.maximum(var_factor, 0.0) * np.asarray(fit.sigma2, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_vals = np.where(variance > 1e-30, estimate / np.sqrt(variance), 0.0)
        p_vals = 2.0 * sp_special.stdtr(proj.dfres, -np.abs(t_vals))
        return {
            "betas": np.asarray(fit.betas, dtype=np.float64),
            "sigma2": np.asarray(fit.sigma2, dtype=np.float64),
            "effect": np.asarray(estimate, dtype=np.float64).reshape(-1),
            "variance": np.asarray(variance, dtype=np.float64).reshape(-1),
            "t": np.asarray(t_vals, dtype=np.float64).reshape(-1),
            "p": np.asarray(p_vals, dtype=np.float64).reshape(-1),
        }

    def finish_binned_fit(phi: Array) -> Dict[str, Array]:
        phi_bins = bin_ar1_coefficients(
            np.asarray(phi, dtype=np.float64).reshape(-1),
            coefficient_bin_width,
        )
        n_regressors, n_voxels = X_base.shape[1], Y_base.shape[1]
        betas = np.zeros((n_regressors, n_voxels), dtype=np.float64)
        sigma2 = np.zeros(n_voxels, dtype=np.float64)
        effect = np.zeros(n_voxels, dtype=np.float64)
        variance = np.zeros(n_voxels, dtype=np.float64)
        t_vals = np.zeros(n_voxels, dtype=np.float64)
        p_vals = np.zeros(n_voxels, dtype=np.float64)

        for ar1_bin in np.unique(phi_bins):
            cols_idx = phi_bins == ar1_bin
            X_w, Y_w = ar_whiten_matrix(
                X_base,
                Y_base[:, cols_idx],
                np.array([ar1_bin], dtype=np.float64),
                exact_first_ar1=exact_first_ar1,
            )
            outputs = finish_fit(X_w, Y_w)
            betas[:, cols_idx] = outputs["betas"]
            sigma2[cols_idx] = outputs["sigma2"]
            effect[cols_idx] = outputs["effect"]
            variance[cols_idx] = outputs["variance"]
            t_vals[cols_idx] = outputs["t"]
            p_vals[cols_idx] = outputs["p"]

        return {
            "betas": betas,
            "sigma2": sigma2,
            "effect": effect,
            "variance": variance,
            "t": t_vals,
            "p": p_vals,
        }

    outputs: Dict[str, Array] | None = None
    X_fit = X_base
    Y_fit = Y_base
    beta_for_residual: Array | None = None
    for _ in range(iter_gls):
        if beta_for_residual is None:
            proj_ols = fast_preproject(X_fit, check_finite=False)
            fit_ols = fast_lm_matrix(
                X_fit,
                Y_fit,
                proj_ols,
                return_fitted=True,
                check_finite=False,
            )
            beta_for_residual = np.asarray(fit_ols.betas, dtype=np.float64)
        # Re-estimate AR on residuals in the original (unwhitened) domain.
        residuals = Y_base - X_base @ beta_for_residual
        phi = estimate_ar(residuals, order=1, voxelwise=voxelwise)
        if coefficient_bin_width is None:
            X_fit, Y_fit = ar_whiten_matrix(
                X_base,
                Y_base,
                phi,
                exact_first_ar1=exact_first_ar1,
            )
            outputs = finish_fit(X_fit, Y_fit)
        else:
            outputs = finish_binned_fit(phi)
        beta_for_residual = outputs["betas"]

    if outputs is None:
        raise RuntimeError("AR(1) fit did not produce outputs")
    return outputs


def benchmark_ar1_implementations(
    X: Array,
    Y: Array,
    contrast: Array,
    *,
    repeats: int = 5,
    warmup: int = 1,
    iter_gls: int = 1,
    voxelwise: bool = False,
    coefficient_bin_width: float | None = None,
) -> BenchmarkSummary:
    """Benchmark matched AR(1) runs for fmrimod and nilearn reference."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    fmrimod_runs: list[float] = []
    reference_runs: list[float] = []

    for _ in range(warmup):
        fit_fmrimod_ar1(
            X,
            Y,
            contrast,
            iter_gls=iter_gls,
            voxelwise=voxelwise,
            coefficient_bin_width=coefficient_bin_width,
        )
        fit_fitlins_reference_ar1(X, Y, contrast)

    for _ in range(repeats):
        t0 = time.perf_counter()
        fit_fmrimod_ar1(
            X,
            Y,
            contrast,
            iter_gls=iter_gls,
            voxelwise=voxelwise,
            coefficient_bin_width=coefficient_bin_width,
        )
        fmrimod_runs.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        fit_fitlins_reference_ar1(X, Y, contrast)
        reference_runs.append(time.perf_counter() - t0)

    fmrimod_median = float(np.median(fmrimod_runs))
    reference_median = float(np.median(reference_runs))
    speedup = float(reference_median / max(fmrimod_median, 1e-12))

    return BenchmarkSummary(
        fmrimod_runs_s=tuple(fmrimod_runs),
        reference_runs_s=tuple(reference_runs),
        fmrimod_median_s=fmrimod_median,
        reference_median_s=reference_median,
        speedup_vs_reference=speedup,
    )


def run_ar1_parity_and_benchmark(
    *,
    n_timepoints: int = 260,
    n_regressors: int = 10,
    n_voxels: int = 3000,
    phi: float = 0.45,
    noise_sd: float = 1.0,
    seed: int = 1234,
    repeats: int = 5,
    warmup: int = 1,
    iter_gls: int = 1,
    voxelwise: bool = False,
    coefficient_bin_width: float | None = None,
    parity_thresholds: ParityThresholds = DEFAULT_AR1_PARITY_THRESHOLDS,
    speed_thresholds: SpeedThresholds = DEFAULT_AR1_SPEED_THRESHOLDS,
) -> Dict[str, Any]:
    """Run one full AR(1) parity + speed evaluation and return JSON-safe report."""
    X, Y, _beta_true, contrast = make_synthetic_glm_ar1(
        n_timepoints=n_timepoints,
        n_regressors=n_regressors,
        n_voxels=n_voxels,
        phi=phi,
        noise_sd=noise_sd,
        seed=seed,
    )

    candidate = fit_fmrimod_ar1(
        X,
        Y,
        contrast,
        iter_gls=iter_gls,
        voxelwise=voxelwise,
        coefficient_bin_width=coefficient_bin_width,
    )
    reference = fit_fitlins_reference_ar1(X, Y, contrast)

    parity = compute_parity_metrics(
        candidate,
        reference,
        sign_flip_floor=parity_thresholds.sign_flip_floor,
    )
    parity_failures = parity.failures(parity_thresholds)
    parity_ok = not parity_failures

    bench = benchmark_ar1_implementations(
        X,
        Y,
        contrast,
        repeats=repeats,
        warmup=warmup,
        iter_gls=iter_gls,
        voxelwise=voxelwise,
        coefficient_bin_width=coefficient_bin_width,
    )
    speed_ok = bench.speedup_vs_reference >= speed_thresholds.min_speedup_vs_reference

    return {
        "config": {
            "n_timepoints": int(n_timepoints),
            "n_regressors": int(n_regressors),
            "n_voxels": int(n_voxels),
            "phi": float(phi),
            "noise_sd": float(noise_sd),
            "seed": int(seed),
            "repeats": int(repeats),
            "warmup": int(warmup),
            "iter_gls": int(iter_gls),
            "voxelwise": bool(voxelwise),
            "coefficient_bin_width": coefficient_bin_width,
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
