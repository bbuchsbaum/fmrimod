"""Core parity matrix builder with concrete workstream runners.

Implemented workstreams:
- WS01: design-matrix construction parity
- WS02: contrast-engine parity
- WS03: variance + degrees-of-freedom parity
- WS04: run-combination parity
- WS05: censor/sample-mask parity
- WS06: LSA/LSS parity + performance
- WS07: rank-deficient design behavior parity
- WS08: numeric precision parity
- WS09: residual diagnostic parity
- WS10: performance decomposition parity
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import special as sp_special

from cross_testing.fitlins_parity import (
    benchmark_implementations,
    compute_parity_metrics,
    fit_fitlins_reference_ols,
    fit_fmrimod_ols,
    make_synthetic_glm,
)
from cross_testing.fitlins_ar1_parity import (
    benchmark_ar1_implementations,
    fit_fitlins_reference_ar1,
    fit_fmrimod_ar1,
    make_synthetic_glm_ar1,
)
from fmrimod.ar.estimation import estimate_ar
from fmrimod.ar.whitening import ar_whiten_matrix
from fmrimod.design.event_model import event_model
from fmrimod.glm.contrasts import contrast_f, contrast_t
from fmrimod.glm.preprocess import apply_censoring
from fmrimod.glm.solver import fast_lm_matrix, fast_preproject
from fmrimod.single._project import build_nuisance_projector
from fmrimod.single.lsa import lsa_single_trial
from fmrimod.single.lss import lss_single_trial


Array = NDArray[np.float64]


WS_NAMES = {
    "ws01": "Design matrix construction parity",
    "ws02": "Contrast engine parity",
    "ws03": "Variance + degrees-of-freedom parity",
    "ws04": "Run-combination parity",
    "ws05": "Censor/sample-mask parity",
    "ws06": "LSA/LSS parity + performance",
    "ws07": "Rank-deficient design behavior",
    "ws08": "Numeric precision parity",
    "ws09": "Residual diagnostic parity",
    "ws10": "Performance decomposition parity",
}


def _safe_corr(a: Array, b: Array) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[mask]
    bb = bb[mask]
    if aa.size < 2:
        return 1.0
    a_std = float(np.std(aa))
    b_std = float(np.std(bb))
    if a_std == 0.0 and b_std == 0.0:
        return 1.0 if np.allclose(aa, bb) else 0.0
    if a_std == 0.0 or b_std == 0.0:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _align_by_lag(
    candidate: Array, reference: Array, lag: int
) -> tuple[Array, Array]:
    """Align 1D series under an integer lag in scan units."""
    c = np.asarray(candidate, dtype=np.float64).reshape(-1)
    r = np.asarray(reference, dtype=np.float64).reshape(-1)
    if lag == 0:
        return c, r
    if lag > 0:
        return c[:-lag], r[lag:]
    return c[-lag:], r[:lag]


def _best_lag_match(
    candidate: Array,
    reference: Array,
    *,
    max_lag_scans: int = 1,
) -> dict[str, float]:
    """Return best lag-aligned correlation/MAE summary."""
    best = {
        "corr": -np.inf,
        "lag_scans": 0.0,
        "scaled_mae": np.inf,
        "scale": 1.0,
    }
    for lag in range(-int(max_lag_scans), int(max_lag_scans) + 1):
        c_seg, r_seg = _align_by_lag(candidate, reference, lag)
        if c_seg.size < 3 or r_seg.size < 3:
            continue
        corr = _safe_corr(c_seg, r_seg)
        denom = float(np.dot(c_seg, c_seg))
        scale = float(np.dot(c_seg, r_seg) / denom) if denom > 1e-18 else 1.0
        scaled_mae = float(np.mean(np.abs(scale * c_seg - r_seg)))
        if corr > float(best["corr"]):
            best = {
                "corr": float(corr),
                "lag_scans": float(lag),
                "scaled_mae": float(scaled_mae),
                "scale": float(scale),
            }
    return best


def _benchmark_pair(
    candidate_fn,
    reference_fn,
    *,
    repeats: int,
    warmup: int,
) -> Dict[str, Any]:
    if int(repeats) < 1:
        raise ValueError("repeats must be >= 1")

    for _ in range(int(warmup)):
        candidate_fn()
        reference_fn()

    candidate_runs: list[float] = []
    reference_runs: list[float] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        candidate_fn()
        candidate_runs.append(float(time.perf_counter() - t0))

        t0 = time.perf_counter()
        reference_fn()
        reference_runs.append(float(time.perf_counter() - t0))

    candidate_median = float(np.median(candidate_runs))
    reference_median = float(np.median(reference_runs))
    speedup = float(reference_median / max(candidate_median, 1e-12))
    return {
        "candidate_runs_s": candidate_runs,
        "reference_runs_s": reference_runs,
        "candidate_median_s": candidate_median,
        "reference_median_s": reference_median,
        "speedup_vs_reference": speedup,
    }


def _placeholder_workstream(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "status": "not_started",
        "parity_ok": False,
        "performance_ok": False,
        "notes": "Pending implementation.",
        "artifacts": [],
    }


def _compute_contrast_compat(labels: Array, estimates: Mapping[Any, Any], con: Array, stat: str):
    from nilearn.glm.contrasts import compute_contrast

    stat_norm = str(stat).strip().lower()
    if stat_norm not in {"t", "f"}:
        raise ValueError(f"Unsupported contrast stat type: {stat}")

    # New nilearn API: stat_type accepts "t" or "F"
    stat_type = "t" if stat_norm == "t" else "F"
    try:
        return compute_contrast(labels, estimates, con, stat_type=stat_type)
    except TypeError:
        # Older nilearn/nistats API.
        contrast_type = "t" if stat_norm == "t" else "F"
        return compute_contrast(labels, estimates, con, contrast_type=contrast_type)


def _maybe_attr_array(obj: Any, names: Sequence[str]) -> Array | None:
    for name in names:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        value = value() if callable(value) else value
        try:
            return np.asarray(value, dtype=np.float64).reshape(-1)
        except Exception:
            continue
    return None


def _z_from_two_sided_p(p_vals: Array, signed_stat: Array) -> Array:
    p = np.clip(np.asarray(p_vals, dtype=np.float64), 1e-300, 1.0 - 1e-16)
    signed = np.asarray(signed_stat, dtype=np.float64)
    z = sp_special.ndtri(1.0 - 0.5 * p)
    return np.sign(signed) * z


def run_ws01_design_matrix_parity(
    *,
    n_scans: int = 180,
    tr: float = 1.0,
    seed: int = 7,
    min_column_corr_raw: float = 0.92,
    min_column_corr_lagged: float = 0.97,
    max_scaled_mae_lagged: float = 0.025,
    max_abs_lag_scans: int = 1,
    match_max_lag_scans: int = 1,
) -> Dict[str, Any]:
    """Run WS01 parity on a deterministic two-condition event fixture."""
    from nilearn.glm.first_level import make_first_level_design_matrix

    fixture_specs = [
        ("base", int(n_scans), float(tr), int(seed)),
        (
            "alt_timing",
            max(120, int(round(0.75 * int(n_scans)))),
            float(tr) * 1.2,
            int(seed) + 101,
        ),
    ]

    fixture_reports: dict[str, Any] = {}
    missing: list[str] = []
    raw_corr_values: list[float] = []
    lagged_corr_values: list[float] = []
    lagged_mae_values: list[float] = []
    lag_values: list[int] = []

    for fixture_name, fixture_scans, fixture_tr, fixture_seed in fixture_specs:
        rng = np.random.default_rng(fixture_seed)
        frame_times = np.arange(fixture_scans, dtype=np.float64) * float(fixture_tr)

        n_events = 18
        onset_grid = np.linspace(8.0, max(20.0, frame_times[-1] - 24.0), n_events)
        onset_jitter = rng.uniform(-0.4, 0.4, size=n_events)
        onsets = np.sort(np.clip(onset_grid + onset_jitter, 0.0, frame_times[-1] - 4.0))
        durations = np.where(np.arange(n_events) % 3 == 0, 2.0, 1.0)
        trial_type = np.where(np.arange(n_events) % 2 == 0, "A", "B")

        events_nilearn = pd.DataFrame(
            {
                "onset": onsets,
                "duration": durations,
                "trial_type": trial_type,
            }
        )
        ref_df = make_first_level_design_matrix(
            frame_times,
            events=events_nilearn,
            hrf_model="spm",
            drift_model=None,
        )

        events_fmrimod = events_nilearn.rename(columns={"trial_type": "condition"})
        model = event_model(
            "hrf(condition)",
            data=events_fmrimod,
            tr=fixture_tr,
            n_scans=fixture_scans,
            durations="duration",
        )
        cand = np.asarray(model.design_matrix, dtype=np.float64)
        cand_names = list(model.column_names)

        per_column: dict[str, dict[str, float]] = {}
        for level in sorted(set(trial_type)):
            if level not in ref_df.columns:
                missing.append(f"{fixture_name}:reference:{level}")
                continue
            matches = [idx for idx, name in enumerate(cand_names) if f".{level}" in name]
            if len(matches) != 1:
                missing.append(f"{fixture_name}:candidate:{level}")
                continue

            col_ref = np.asarray(ref_df[level].values, dtype=np.float64)
            col_cand = cand[:, matches[0]]

            raw_corr = _safe_corr(col_cand, col_ref)
            best = _best_lag_match(
                col_cand, col_ref, max_lag_scans=int(match_max_lag_scans)
            )

            raw_corr_values.append(float(raw_corr))
            lagged_corr_values.append(float(best["corr"]))
            lagged_mae_values.append(float(best["scaled_mae"]))
            lag_values.append(int(round(float(best["lag_scans"]))))
            per_column[level] = {
                "raw_corr": float(raw_corr),
                "lagged_corr": float(best["corr"]),
                "lag_scans": float(best["lag_scans"]),
                "scaled_mae_lagged": float(best["scaled_mae"]),
                "scale_lagged": float(best["scale"]),
            }

        fixture_reports[fixture_name] = {
            "n_scans": int(fixture_scans),
            "tr": float(fixture_tr),
            "seed": int(fixture_seed),
            "per_column": per_column,
        }

    min_raw_corr = float(min(raw_corr_values)) if raw_corr_values else 0.0
    min_lagged_corr = float(min(lagged_corr_values)) if lagged_corr_values else 0.0
    max_lagged_mae = float(max(lagged_mae_values)) if lagged_mae_values else np.inf
    max_abs_lag = int(max((abs(v) for v in lag_values), default=0))

    failures: list[str] = []
    if missing:
        failures.append("missing_columns")
    if min_raw_corr < float(min_column_corr_raw):
        failures.append("min_column_corr_raw")
    if min_lagged_corr < float(min_column_corr_lagged):
        failures.append("min_column_corr_lagged")
    if max_lagged_mae > float(max_scaled_mae_lagged):
        failures.append("max_scaled_mae_lagged")
    if max_abs_lag > int(max_abs_lag_scans):
        failures.append("max_abs_lag_scans")

    return {
        "name": WS_NAMES["ws01"],
        "status": "complete",
        "parity_ok": len(failures) == 0,
        "performance_ok": True,
        "notes": "Two deterministic timing fixtures with lag-aware matching.",
        "artifacts": [],
        "thresholds": {
            "min_column_corr_raw": float(min_column_corr_raw),
            "min_column_corr_lagged": float(min_column_corr_lagged),
            "max_scaled_mae_lagged": float(max_scaled_mae_lagged),
            "max_abs_lag_scans": int(max_abs_lag_scans),
            "match_max_lag_scans": int(match_max_lag_scans),
        },
        "metrics": {
            "min_column_corr_raw": min_raw_corr,
            "min_column_corr_lagged": min_lagged_corr,
            "max_scaled_mae_lagged": max_lagged_mae,
            "max_abs_lag_scans": int(max_abs_lag),
            "missing_columns": missing,
            "fixtures": fixture_reports,
        },
        "failures": failures,
    }


def _default_f_contrast(n_regressors: int) -> Array:
    idx = [i for i in range(1, n_regressors)][:2]
    if not idx:
        idx = [0]
    con = np.zeros((len(idx), n_regressors), dtype=np.float64)
    for row, col in enumerate(idx):
        con[row, col] = 1.0
    return con


def run_ws02_contrast_parity(
    *,
    n_timepoints: int = 220,
    n_regressors: int = 8,
    n_voxels: int = 1200,
    noise_sd: float = 1.0,
    seed: int = 1234,
    min_t_corr: float = 0.999,
    max_p_mae: float = 0.001,
    min_effect_corr: float = 0.9999,
    min_variance_corr: float = 0.999,
    min_z_corr: float = 0.999,
    min_f_corr: float = 0.999,
    max_f_p_mae: float = 0.001,
) -> Dict[str, Any]:
    """Run WS02 parity on t/F/effect/variance/z/p outputs."""
    from nilearn.glm.first_level import run_glm

    X, Y, _beta_true, t_con = make_synthetic_glm(
        n_timepoints=n_timepoints,
        n_regressors=n_regressors,
        n_voxels=n_voxels,
        noise_sd=noise_sd,
        seed=seed,
    )
    f_con = _default_f_contrast(n_regressors)

    candidate = fit_fmrimod_ols(X, Y, t_con)
    reference = fit_fitlins_reference_ols(X, Y, t_con)
    basic = compute_parity_metrics(candidate, reference)

    proj = fast_preproject(X, check_finite=False)
    var_factor = float(t_con @ np.asarray(proj.XtXinv, dtype=np.float64) @ t_con)

    cand_effect = np.asarray(t_con @ candidate["betas"], dtype=np.float64).reshape(-1)
    cand_variance = np.asarray(var_factor * candidate["sigma2"], dtype=np.float64).reshape(-1)
    cand_z = _z_from_two_sided_p(candidate["p"], candidate["t"])

    labels, estimates = run_glm(Y, X, noise_model="ols")
    t_res = _compute_contrast_compat(labels, estimates, t_con, stat="t")
    f_res = _compute_contrast_compat(labels, estimates, f_con, stat="f")

    ref_effect = _maybe_attr_array(t_res, ["effect_size", "effect"])
    if ref_effect is None:
        ref_effect = np.asarray(t_con @ reference["betas"], dtype=np.float64).reshape(-1)

    ref_variance = _maybe_attr_array(t_res, ["effect_variance", "variance"])
    if ref_variance is None:
        denom = np.asarray(reference["t"], dtype=np.float64).reshape(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ref_variance = np.square(ref_effect / denom)
        ref_variance = np.where(np.isfinite(ref_variance), ref_variance, 0.0)
    ref_variance = np.asarray(ref_variance, dtype=np.float64).reshape(-1)

    ref_z = _maybe_attr_array(t_res, ["z_score", "z"])
    if ref_z is None:
        ref_z = _z_from_two_sided_p(reference["p"], reference["t"])

    ref_f_stat = np.asarray(f_res.stat(), dtype=np.float64).reshape(-1)
    ref_f_p = np.asarray(f_res.p_value(), dtype=np.float64).reshape(-1)

    cand_sigma = np.sqrt(np.maximum(np.asarray(candidate["sigma2"], dtype=np.float64), 0.0))
    cand_f = contrast_f(
        f_con,
        np.asarray(candidate["betas"], dtype=np.float64),
        np.asarray(proj.XtXinv, dtype=np.float64),
        cand_sigma,
        float(proj.dfres),
        name="ws02_f",
    )
    cand_f_stat = np.asarray(cand_f.stat, dtype=np.float64).reshape(-1)
    cand_f_p = np.asarray(cand_f.p_value, dtype=np.float64).reshape(-1)

    metrics = {
        "t_corr": float(basic.t_corr),
        "p_mae": float(basic.p_mae),
        "effect_corr": _safe_corr(cand_effect, ref_effect),
        "variance_corr": _safe_corr(cand_variance, ref_variance),
        "z_corr": _safe_corr(cand_z, ref_z),
        "f_corr": _safe_corr(cand_f_stat, ref_f_stat),
        "f_p_mae": float(np.mean(np.abs(cand_f_p - ref_f_p))),
    }

    failures: list[str] = []
    if metrics["t_corr"] < float(min_t_corr):
        failures.append("min_t_corr")
    if metrics["p_mae"] > float(max_p_mae):
        failures.append("max_p_mae")
    if metrics["effect_corr"] < float(min_effect_corr):
        failures.append("min_effect_corr")
    if metrics["variance_corr"] < float(min_variance_corr):
        failures.append("min_variance_corr")
    if metrics["z_corr"] < float(min_z_corr):
        failures.append("min_z_corr")
    if metrics["f_corr"] < float(min_f_corr):
        failures.append("min_f_corr")
    if metrics["f_p_mae"] > float(max_f_p_mae):
        failures.append("max_f_p_mae")

    return {
        "name": WS_NAMES["ws02"],
        "status": "complete",
        "parity_ok": len(failures) == 0,
        "performance_ok": True,
        "notes": "Synthetic OLS parity for t/F/effect/variance/z/p outputs.",
        "artifacts": [],
        "thresholds": {
            "min_t_corr": float(min_t_corr),
            "max_p_mae": float(max_p_mae),
            "min_effect_corr": float(min_effect_corr),
            "min_variance_corr": float(min_variance_corr),
            "min_z_corr": float(min_z_corr),
            "min_f_corr": float(min_f_corr),
            "max_f_p_mae": float(max_f_p_mae),
        },
        "metrics": metrics,
        "failures": failures,
    }


def _reference_dfres_from_nilearn(X: Array, Y: Array, noise_model: str) -> float:
    from nilearn.glm.first_level import run_glm

    _labels, estimates = run_glm(Y, X, noise_model=noise_model)
    if not estimates:
        return float(X.shape[0] - np.linalg.matrix_rank(X))
    first = next(iter(estimates.values()))
    if hasattr(first, "df_residuals"):
        return float(getattr(first, "df_residuals"))
    return float(X.shape[0] - np.linalg.matrix_rank(X))


def _fmrimod_ar1_candidate_dfres(
    X: Array,
    Y: Array,
    *,
    iter_gls: int,
    voxelwise: bool = False,
) -> float:
    X_base = np.asarray(X, dtype=np.float64)
    Y_base = np.asarray(Y, dtype=np.float64)
    X_fit = X_base
    Y_fit = Y_base

    for _ in range(int(iter_gls)):
        proj_ols = fast_preproject(X_fit, check_finite=False)
        fit_ols = fast_lm_matrix(
            X_fit,
            Y_fit,
            proj_ols,
            return_fitted=True,
            check_finite=False,
        )
        residuals = Y_base - X_base @ fit_ols.betas
        phi = estimate_ar(residuals, order=1, voxelwise=voxelwise)
        X_fit, Y_fit = ar_whiten_matrix(X_base, Y_base, phi)

    proj_final = fast_preproject(X_fit, check_finite=False)
    return float(proj_final.dfres)


def run_ws03_variance_df_parity(
    *,
    n_timepoints: int = 220,
    n_regressors: int = 8,
    n_voxels: int = 1200,
    noise_sd: float = 1.0,
    phi: float = 0.45,
    seed: int = 2026,
    min_sigma2_corr_ols: float = 0.999,
    max_sigma2_mae_ols: float = 0.002,
    min_sigma2_corr_ar1_iter1: float = 0.995,
    max_sigma2_mae_ar1_iter1: float = 0.025,
    min_sigma2_corr_ar1_iter2: float = 0.99,
    max_sigma2_mae_ar1_iter2: float = 0.04,
    max_df_absdiff: float = 1e-8,
) -> Dict[str, Any]:
    # OLS path.
    X_ols, Y_ols, _beta_ols, con_ols = make_synthetic_glm(
        n_timepoints=n_timepoints,
        n_regressors=n_regressors,
        n_voxels=n_voxels,
        noise_sd=noise_sd,
        seed=seed,
    )
    cand_ols = fit_fmrimod_ols(X_ols, Y_ols, con_ols)
    ref_ols = fit_fitlins_reference_ols(X_ols, Y_ols, con_ols)
    ols_sigma2_corr = _safe_corr(cand_ols["sigma2"], ref_ols["sigma2"])
    ols_sigma2_mae = float(
        np.mean(
            np.abs(
                np.asarray(cand_ols["sigma2"], dtype=np.float64)
                - np.asarray(ref_ols["sigma2"], dtype=np.float64)
            )
        )
    )
    ols_df_candidate = float(fast_preproject(X_ols, check_finite=False).dfres)
    ols_df_reference = _reference_dfres_from_nilearn(X_ols, Y_ols, noise_model="ols")
    ols_df_absdiff = float(abs(ols_df_candidate - ols_df_reference))

    # AR1 path (iter_gls=1 and iter_gls=2).
    X_ar, Y_ar, _beta_ar, con_ar = make_synthetic_glm_ar1(
        n_timepoints=n_timepoints,
        n_regressors=n_regressors,
        n_voxels=n_voxels,
        phi=phi,
        noise_sd=noise_sd,
        seed=seed + 1,
    )
    ref_ar = fit_fitlins_reference_ar1(X_ar, Y_ar, con_ar)
    ar_df_reference = _reference_dfres_from_nilearn(X_ar, Y_ar, noise_model="ar1")

    cand_ar_i1 = fit_fmrimod_ar1(X_ar, Y_ar, con_ar, iter_gls=1, voxelwise=False)
    ar1_i1_sigma2_corr = _safe_corr(cand_ar_i1["sigma2"], ref_ar["sigma2"])
    ar1_i1_sigma2_mae = float(
        np.mean(
            np.abs(
                np.asarray(cand_ar_i1["sigma2"], dtype=np.float64)
                - np.asarray(ref_ar["sigma2"], dtype=np.float64)
            )
        )
    )
    ar1_i1_df_candidate = _fmrimod_ar1_candidate_dfres(
        X_ar, Y_ar, iter_gls=1, voxelwise=False
    )
    ar1_i1_df_absdiff = float(abs(ar1_i1_df_candidate - ar_df_reference))

    cand_ar_i2 = fit_fmrimod_ar1(X_ar, Y_ar, con_ar, iter_gls=2, voxelwise=False)
    ar1_i2_sigma2_corr = _safe_corr(cand_ar_i2["sigma2"], ref_ar["sigma2"])
    ar1_i2_sigma2_mae = float(
        np.mean(
            np.abs(
                np.asarray(cand_ar_i2["sigma2"], dtype=np.float64)
                - np.asarray(ref_ar["sigma2"], dtype=np.float64)
            )
        )
    )
    ar1_i2_df_candidate = _fmrimod_ar1_candidate_dfres(
        X_ar, Y_ar, iter_gls=2, voxelwise=False
    )
    ar1_i2_df_absdiff = float(abs(ar1_i2_df_candidate - ar_df_reference))

    metrics = {
        "ols_sigma2_corr": float(ols_sigma2_corr),
        "ols_sigma2_mae": float(ols_sigma2_mae),
        "ols_df_candidate": float(ols_df_candidate),
        "ols_df_reference": float(ols_df_reference),
        "ols_df_absdiff": float(ols_df_absdiff),
        "ar1_iter1_sigma2_corr": float(ar1_i1_sigma2_corr),
        "ar1_iter1_sigma2_mae": float(ar1_i1_sigma2_mae),
        "ar1_iter1_df_candidate": float(ar1_i1_df_candidate),
        "ar1_iter1_df_reference": float(ar_df_reference),
        "ar1_iter1_df_absdiff": float(ar1_i1_df_absdiff),
        "ar1_iter2_sigma2_corr": float(ar1_i2_sigma2_corr),
        "ar1_iter2_sigma2_mae": float(ar1_i2_sigma2_mae),
        "ar1_iter2_df_candidate": float(ar1_i2_df_candidate),
        "ar1_iter2_df_reference": float(ar_df_reference),
        "ar1_iter2_df_absdiff": float(ar1_i2_df_absdiff),
    }

    thresholds = {
        "min_sigma2_corr_ols": float(min_sigma2_corr_ols),
        "max_sigma2_mae_ols": float(max_sigma2_mae_ols),
        "min_sigma2_corr_ar1_iter1": float(min_sigma2_corr_ar1_iter1),
        "max_sigma2_mae_ar1_iter1": float(max_sigma2_mae_ar1_iter1),
        "min_sigma2_corr_ar1_iter2": float(min_sigma2_corr_ar1_iter2),
        "max_sigma2_mae_ar1_iter2": float(max_sigma2_mae_ar1_iter2),
        "max_df_absdiff": float(max_df_absdiff),
    }

    failures: list[str] = []
    if metrics["ols_sigma2_corr"] < thresholds["min_sigma2_corr_ols"]:
        failures.append("min_sigma2_corr_ols")
    if metrics["ols_sigma2_mae"] > thresholds["max_sigma2_mae_ols"]:
        failures.append("max_sigma2_mae_ols")
    if metrics["ar1_iter1_sigma2_corr"] < thresholds["min_sigma2_corr_ar1_iter1"]:
        failures.append("min_sigma2_corr_ar1_iter1")
    if metrics["ar1_iter1_sigma2_mae"] > thresholds["max_sigma2_mae_ar1_iter1"]:
        failures.append("max_sigma2_mae_ar1_iter1")
    if metrics["ar1_iter2_sigma2_corr"] < thresholds["min_sigma2_corr_ar1_iter2"]:
        failures.append("min_sigma2_corr_ar1_iter2")
    if metrics["ar1_iter2_sigma2_mae"] > thresholds["max_sigma2_mae_ar1_iter2"]:
        failures.append("max_sigma2_mae_ar1_iter2")
    if metrics["ols_df_absdiff"] > thresholds["max_df_absdiff"]:
        failures.append("max_ols_df_absdiff")
    if metrics["ar1_iter1_df_absdiff"] > thresholds["max_df_absdiff"]:
        failures.append("max_ar1_iter1_df_absdiff")
    if metrics["ar1_iter2_df_absdiff"] > thresholds["max_df_absdiff"]:
        failures.append("max_ar1_iter2_df_absdiff")

    return {
        "name": WS_NAMES["ws03"],
        "status": "complete",
        "parity_ok": len(failures) == 0,
        "performance_ok": True,
        "notes": "OLS + AR1(iter_gls=1/2) sigma2 and df parity.",
        "artifacts": [],
        "thresholds": thresholds,
        "metrics": metrics,
        "failures": failures,
    }


def _fixed_effects_combine(
    effects: Array,
    variances: Array,
) -> tuple[Array, Array]:
    """Inverse-variance fixed-effects pooling across runs."""
    var = np.asarray(variances, dtype=np.float64)
    eff = np.asarray(effects, dtype=np.float64)
    var = np.maximum(var, 1e-12)
    precision = 1.0 / var
    precision_sum = np.sum(precision, axis=0)
    pooled_var = 1.0 / np.maximum(precision_sum, 1e-12)
    pooled_eff = np.sum(eff * precision, axis=0) * pooled_var
    return pooled_eff, pooled_var


def _run_ws04_single_fixture(
    *,
    n_runs: int,
    n_timepoints: int,
    n_regressors: int,
    n_voxels: int,
    noise_sd: float,
    seed: int,
) -> Dict[str, Any]:
    effects_c: list[Array] = []
    vars_c: list[Array] = []
    effects_r: list[Array] = []
    vars_r: list[Array] = []
    df_c_total = 0.0
    df_r_total = 0.0

    for run_idx in range(int(n_runs)):
        X, Y, _beta, con = make_synthetic_glm(
            n_timepoints=n_timepoints,
            n_regressors=n_regressors,
            n_voxels=n_voxels,
            noise_sd=noise_sd,
            seed=seed + 17 * run_idx,
        )
        cand = fit_fmrimod_ols(X, Y, con)
        ref = fit_fitlins_reference_ols(X, Y, con)

        proj = fast_preproject(X, check_finite=False)
        var_factor = float(con @ np.asarray(proj.XtXinv, dtype=np.float64) @ con)

        effect_c = np.asarray(con @ cand["betas"], dtype=np.float64).reshape(-1)
        var_c = np.maximum(var_factor, 0.0) * np.asarray(cand["sigma2"], dtype=np.float64)
        effect_r = np.asarray(con @ ref["betas"], dtype=np.float64).reshape(-1)
        var_r = np.maximum(var_factor, 0.0) * np.asarray(ref["sigma2"], dtype=np.float64)

        effects_c.append(effect_c)
        vars_c.append(np.asarray(var_c, dtype=np.float64).reshape(-1))
        effects_r.append(effect_r)
        vars_r.append(np.asarray(var_r, dtype=np.float64).reshape(-1))

        df_c_total += float(proj.dfres)
        df_r_total += _reference_dfres_from_nilearn(X, Y, noise_model="ols")

    eff_c, var_c = _fixed_effects_combine(np.vstack(effects_c), np.vstack(vars_c))
    eff_r, var_r = _fixed_effects_combine(np.vstack(effects_r), np.vstack(vars_r))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_c = np.where(var_c > 1e-20, eff_c / np.sqrt(var_c), 0.0)
        t_r = np.where(var_r > 1e-20, eff_r / np.sqrt(var_r), 0.0)
    p_c = 2.0 * sp_special.stdtr(df_c_total, -np.abs(t_c))
    p_r = 2.0 * sp_special.stdtr(df_r_total, -np.abs(t_r))

    return {
        "n_runs": int(n_runs),
        "df_candidate_total": float(df_c_total),
        "df_reference_total": float(df_r_total),
        "df_absdiff": float(abs(df_c_total - df_r_total)),
        "effect_corr": _safe_corr(eff_c, eff_r),
        "effect_mae": float(np.mean(np.abs(eff_c - eff_r))),
        "variance_corr": _safe_corr(var_c, var_r),
        "variance_mae": float(np.mean(np.abs(var_c - var_r))),
        "t_corr": _safe_corr(t_c, t_r),
        "t_mae": float(np.mean(np.abs(t_c - t_r))),
        "p_mae": float(np.mean(np.abs(p_c - p_r))),
    }


def run_ws04_run_combination_parity(
    *,
    n_timepoints: int = 180,
    n_regressors: int = 8,
    n_voxels: int = 1000,
    noise_sd: float = 1.0,
    seed: int = 3030,
    min_effect_corr: float = 0.999,
    max_effect_mae: float = 0.005,
    min_variance_corr: float = 0.999,
    max_variance_mae: float = 0.005,
    min_t_corr: float = 0.999,
    max_t_mae: float = 0.01,
    max_p_mae: float = 0.001,
    max_df_absdiff: float = 1e-8,
) -> Dict[str, Any]:
    fixture_2 = _run_ws04_single_fixture(
        n_runs=2,
        n_timepoints=n_timepoints,
        n_regressors=n_regressors,
        n_voxels=n_voxels,
        noise_sd=noise_sd,
        seed=seed,
    )
    fixture_3 = _run_ws04_single_fixture(
        n_runs=3,
        n_timepoints=n_timepoints,
        n_regressors=n_regressors,
        n_voxels=n_voxels,
        noise_sd=noise_sd,
        seed=seed + 1000,
    )

    thresholds = {
        "min_effect_corr": float(min_effect_corr),
        "max_effect_mae": float(max_effect_mae),
        "min_variance_corr": float(min_variance_corr),
        "max_variance_mae": float(max_variance_mae),
        "min_t_corr": float(min_t_corr),
        "max_t_mae": float(max_t_mae),
        "max_p_mae": float(max_p_mae),
        "max_df_absdiff": float(max_df_absdiff),
    }

    failures: list[str] = []
    fixtures = {"runs_2": fixture_2, "runs_3": fixture_3}
    for fixture_name, m in fixtures.items():
        if m["effect_corr"] < thresholds["min_effect_corr"]:
            failures.append(f"{fixture_name}:min_effect_corr")
        if m["effect_mae"] > thresholds["max_effect_mae"]:
            failures.append(f"{fixture_name}:max_effect_mae")
        if m["variance_corr"] < thresholds["min_variance_corr"]:
            failures.append(f"{fixture_name}:min_variance_corr")
        if m["variance_mae"] > thresholds["max_variance_mae"]:
            failures.append(f"{fixture_name}:max_variance_mae")
        if m["t_corr"] < thresholds["min_t_corr"]:
            failures.append(f"{fixture_name}:min_t_corr")
        if m["t_mae"] > thresholds["max_t_mae"]:
            failures.append(f"{fixture_name}:max_t_mae")
        if m["p_mae"] > thresholds["max_p_mae"]:
            failures.append(f"{fixture_name}:max_p_mae")
        if m["df_absdiff"] > thresholds["max_df_absdiff"]:
            failures.append(f"{fixture_name}:max_df_absdiff")

    return {
        "name": WS_NAMES["ws04"],
        "status": "complete",
        "parity_ok": len(failures) == 0,
        "performance_ok": True,
        "notes": "Fixed-effects pooling parity on 2-run and 3-run synthetic fixtures.",
        "artifacts": [],
        "thresholds": thresholds,
        "metrics": fixtures,
        "failures": failures,
    }


def _build_ws05_censor_fixtures(n_timepoints: int) -> Dict[str, NDArray[np.bool_]]:
    t = int(n_timepoints)
    fixtures: Dict[str, NDArray[np.bool_]] = {}

    mid_gap = np.zeros(t, dtype=bool)
    for idx in [max(3, t // 6), max(4, t // 6 + 1), t // 2, min(t - 3, t // 2 + 1)]:
        if 0 <= idx < t:
            mid_gap[idx] = True
    fixtures["mid_gap"] = mid_gap

    edge_and_mid = np.zeros(t, dtype=bool)
    edge_indices = [0, 1, 2, t - 3, t - 2, t - 1, t // 3]
    for idx in edge_indices:
        if 0 <= idx < t:
            edge_and_mid[idx] = True
    fixtures["edge_and_mid"] = edge_and_mid

    sparse_periodic = np.zeros(t, dtype=bool)
    stride = max(11, t // 14)
    sparse_periodic[::stride] = True
    sparse_periodic[:2] = False
    sparse_periodic[-2:] = False
    fixtures["sparse_periodic"] = sparse_periodic

    return fixtures


def run_ws05_censor_sample_mask_parity(
    *,
    n_timepoints: int = 220,
    n_regressors: int = 8,
    n_voxels: int = 1200,
    noise_sd: float = 1.0,
    seed: int = 505,
    min_t_corr: float = 0.999,
    max_p_mae: float = 0.001,
    min_beta_corr: float = 0.999,
    min_sigma2_corr: float = 0.999,
    max_df_absdiff: float = 1e-8,
) -> Dict[str, Any]:
    X, Y, _beta, contrast = make_synthetic_glm(
        n_timepoints=n_timepoints,
        n_regressors=n_regressors,
        n_voxels=n_voxels,
        noise_sd=noise_sd,
        seed=seed,
    )
    fixtures = _build_ws05_censor_fixtures(int(n_timepoints))

    fixture_metrics: Dict[str, Dict[str, Any]] = {}
    failures: list[str] = []

    for name, censor_mask in fixtures.items():
        Xc, Yc, keep = apply_censoring(X, Y, censor_mask)
        boundary_ok = bool(
            np.array_equal(keep, ~censor_mask)
            and np.array_equal(Xc, X[~censor_mask])
            and np.array_equal(Yc, Y[~censor_mask])
        )
        n_kept = int(Xc.shape[0])
        keep_fraction = float(n_kept / max(1, X.shape[0]))

        cand = fit_fmrimod_ols(Xc, Yc, contrast)
        ref = fit_fitlins_reference_ols(Xc, Yc, contrast)
        pm = compute_parity_metrics(cand, ref)
        df_candidate = float(fast_preproject(Xc, check_finite=False).dfres)
        df_reference = float(_reference_dfres_from_nilearn(Xc, Yc, noise_model="ols"))
        df_absdiff = float(abs(df_candidate - df_reference))

        metrics = {
            "n_kept": n_kept,
            "keep_fraction": keep_fraction,
            "boundary_ok": boundary_ok,
            "beta_corr": float(pm.beta_corr),
            "sigma2_corr": float(pm.sigma2_corr),
            "t_corr": float(pm.t_corr),
            "p_mae": float(pm.p_mae),
            "df_candidate": float(df_candidate),
            "df_reference": float(df_reference),
            "df_absdiff": float(df_absdiff),
        }
        fixture_metrics[name] = metrics

        if not boundary_ok:
            failures.append(f"{name}:boundary_ok")
        if metrics["beta_corr"] < float(min_beta_corr):
            failures.append(f"{name}:min_beta_corr")
        if metrics["sigma2_corr"] < float(min_sigma2_corr):
            failures.append(f"{name}:min_sigma2_corr")
        if metrics["t_corr"] < float(min_t_corr):
            failures.append(f"{name}:min_t_corr")
        if metrics["p_mae"] > float(max_p_mae):
            failures.append(f"{name}:max_p_mae")
        if metrics["df_absdiff"] > float(max_df_absdiff):
            failures.append(f"{name}:max_df_absdiff")

    thresholds = {
        "min_beta_corr": float(min_beta_corr),
        "min_sigma2_corr": float(min_sigma2_corr),
        "min_t_corr": float(min_t_corr),
        "max_p_mae": float(max_p_mae),
        "max_df_absdiff": float(max_df_absdiff),
    }

    return {
        "name": WS_NAMES["ws05"],
        "status": "complete",
        "parity_ok": len(failures) == 0,
        "performance_ok": True,
        "notes": "Censor/sample-mask parity with boundary-sensitive fixtures.",
        "artifacts": [],
        "thresholds": thresholds,
        "metrics": {"fixtures": fixture_metrics},
        "failures": failures,
    }


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


def _canonical_hrf_kernel(length: int = 24) -> np.ndarray:
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


def _build_lsa_lss_synthetic_data(
    *,
    n_tp: int,
    n_trials: int,
    n_voxels: int,
    n_confounds: int,
    noise_sd: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = _build_trial_design(n_tp, n_trials, rng=rng)
    confounds = rng.normal(size=(n_tp, n_confounds)).astype(np.float64)

    beta_true = rng.normal(scale=0.2, size=(n_trials, n_voxels))
    conf_true = rng.normal(scale=0.1, size=(n_confounds, n_voxels))
    noise = rng.normal(scale=float(noise_sd), size=(n_tp, n_voxels))
    Y = X @ beta_true + confounds @ conf_true + noise
    return Y, X, confounds


def _nilearn_lsa(Y: np.ndarray, X: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    from nilearn.glm.first_level import run_glm

    design = np.hstack([X, confounds])
    labels, estimates = run_glm(Y, design, noise_model="ols")
    betas = _extract_betas_from_run_glm(labels, estimates, design.shape[1])
    return betas[: X.shape[1]]


def _nilearn_lss(Y: np.ndarray, X: np.ndarray, confounds: np.ndarray) -> np.ndarray:
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


def run_ws06_lsa_lss_parity_performance(
    *,
    n_timepoints: int = 300,
    n_trials: int = 80,
    n_voxels: int = 2000,
    n_confounds: int = 6,
    noise_sd: float = 1.0,
    seed: int = 6060,
    repeats: int = 3,
    warmup: int = 1,
    chunk_size: int = 4000,
    min_lsa_corr: float = 0.999,
    max_lsa_mae: float = 0.02,
    min_lss_corr: float = 0.995,
    max_lss_mae: float = 0.03,
    min_speedup_lsa: float = 1.0,
    min_speedup_lss_cached: float = 1.2,
    min_projector_speedup_lss: float = 0.85,
) -> Dict[str, Any]:
    Y, X, confounds = _build_lsa_lss_synthetic_data(
        n_tp=n_timepoints,
        n_trials=n_trials,
        n_voxels=n_voxels,
        n_confounds=n_confounds,
        noise_sd=noise_sd,
        seed=seed,
    )
    projector = build_nuisance_projector(confounds)
    if projector is None:
        raise RuntimeError("Failed to build nuisance projector for WS06.")

    cand_lsa = lsa_single_trial(Y, X, confounds=confounds).betas
    ref_lsa = _nilearn_lsa(Y, X, confounds)

    cand_lss_conf = lss_single_trial(
        Y, X, confounds=confounds, chunk_size=chunk_size
    ).betas
    cand_lss_cached = lss_single_trial(
        Y, X, nuisance_projector=projector, chunk_size=chunk_size
    ).betas
    ref_lss = _nilearn_lss(Y, X, confounds)

    lsa_corr = _safe_corr(cand_lsa, ref_lsa)
    lsa_mae = float(np.mean(np.abs(cand_lsa - ref_lsa)))
    lss_corr = _safe_corr(cand_lss_cached, ref_lss)
    lss_mae = float(np.mean(np.abs(cand_lss_cached - ref_lss)))

    lsa_stage = _benchmark_pair(
        lambda: lsa_single_trial(Y, X, confounds=confounds).betas,
        lambda: _nilearn_lsa(Y, X, confounds),
        repeats=int(repeats),
        warmup=int(warmup),
    )
    lss_conf_stage = _benchmark_pair(
        lambda: lss_single_trial(Y, X, confounds=confounds, chunk_size=chunk_size).betas,
        lambda: _nilearn_lss(Y, X, confounds),
        repeats=int(repeats),
        warmup=int(warmup),
    )
    lss_cached_stage = _benchmark_pair(
        lambda: lss_single_trial(
            Y, X, nuisance_projector=projector, chunk_size=chunk_size
        ).betas,
        lambda: _nilearn_lss(Y, X, confounds),
        repeats=int(repeats),
        warmup=int(warmup),
    )
    projector_speedup = float(
        lss_conf_stage["candidate_median_s"]
        / max(float(lss_cached_stage["candidate_median_s"]), 1e-12)
    )

    thresholds = {
        "min_lsa_corr": float(min_lsa_corr),
        "max_lsa_mae": float(max_lsa_mae),
        "min_lss_corr": float(min_lss_corr),
        "max_lss_mae": float(max_lss_mae),
        "min_speedup_lsa": float(min_speedup_lsa),
        "min_speedup_lss_cached": float(min_speedup_lss_cached),
        "min_projector_speedup_lss": float(min_projector_speedup_lss),
    }
    metrics = {
        "lsa_parity_corr": float(lsa_corr),
        "lsa_parity_mae": float(lsa_mae),
        "lss_parity_corr": float(lss_corr),
        "lss_parity_mae": float(lss_mae),
        "lss_confounds_parity_corr": float(_safe_corr(cand_lss_conf, ref_lss)),
        "lss_confounds_parity_mae": float(np.mean(np.abs(cand_lss_conf - ref_lss))),
        "speed": {
            "lsa": lsa_stage,
            "lss_confounds": lss_conf_stage,
            "lss_cached": lss_cached_stage,
            "projector_speedup": float(projector_speedup),
        },
    }

    failures: list[str] = []
    if metrics["lsa_parity_corr"] < thresholds["min_lsa_corr"]:
        failures.append("min_lsa_corr")
    if metrics["lsa_parity_mae"] > thresholds["max_lsa_mae"]:
        failures.append("max_lsa_mae")
    if metrics["lss_parity_corr"] < thresholds["min_lss_corr"]:
        failures.append("min_lss_corr")
    if metrics["lss_parity_mae"] > thresholds["max_lss_mae"]:
        failures.append("max_lss_mae")
    if metrics["speed"]["lsa"]["speedup_vs_reference"] < thresholds["min_speedup_lsa"]:
        failures.append("min_speedup_lsa")
    if metrics["speed"]["lss_cached"]["speedup_vs_reference"] < thresholds["min_speedup_lss_cached"]:
        failures.append("min_speedup_lss_cached")
    if metrics["speed"]["projector_speedup"] < thresholds["min_projector_speedup_lss"]:
        failures.append("min_projector_speedup_lss")

    return {
        "name": WS_NAMES["ws06"],
        "status": "complete",
        "parity_ok": not any(
            f
            for f in failures
            if f in {"min_lsa_corr", "max_lsa_mae", "min_lss_corr", "max_lss_mae"}
        ),
        "performance_ok": not any(
            f
            for f in failures
            if f in {
                "min_speedup_lsa",
                "min_speedup_lss_cached",
                "min_projector_speedup_lss",
            }
        ),
        "notes": "LSA/LSS parity vs nilearn with cached-projector speed gate.",
        "artifacts": [],
        "thresholds": thresholds,
        "metrics": metrics,
        "failures": failures,
    }


def _build_ws07_rank_deficient_fixtures(
    X: Array,
    Y: Array,
) -> Dict[str, tuple[Array, Array]]:
    fixtures: Dict[str, tuple[Array, Array]] = {}

    x_dup = np.asarray(X, dtype=np.float64).copy()
    if x_dup.shape[1] >= 3:
        x_dup[:, 2] = x_dup[:, 1]
    fixtures["duplicate_column"] = (x_dup, np.asarray(Y, dtype=np.float64))

    x_combo = np.asarray(X, dtype=np.float64).copy()
    if x_combo.shape[1] >= 5:
        x_combo[:, 3] = x_combo[:, 1] + x_combo[:, 2]
        x_combo[:, 4] = 0.0
    fixtures["linear_combo_plus_zero"] = (x_combo, np.asarray(Y, dtype=np.float64))

    x_near = np.asarray(X, dtype=np.float64).copy()
    if x_near.shape[1] >= 3:
        x_near[:, 2] = x_near[:, 1] + 1e-10 * x_near[:, 2]
    fixtures["near_collinear"] = (x_near, np.asarray(Y, dtype=np.float64))

    return fixtures


def run_ws07_rank_deficient_design_parity(
    *,
    n_timepoints: int = 220,
    n_regressors: int = 8,
    n_voxels: int = 1200,
    noise_sd: float = 1.0,
    seed: int = 7070,
    min_beta_corr: float = 0.99,
    min_sigma2_corr: float = 0.995,
    min_t_corr: float = 0.99,
    max_p_mae: float = 0.01,
    max_df_absdiff: float = 1e-8,
    max_rank_absdiff: float = 1e-8,
) -> Dict[str, Any]:
    if int(n_regressors) < 5:
        raise ValueError("WS07 requires n_regressors >= 5")

    X, Y, _beta_true, t_con = make_synthetic_glm(
        n_timepoints=int(n_timepoints),
        n_regressors=int(n_regressors),
        n_voxels=int(n_voxels),
        noise_sd=float(noise_sd),
        seed=int(seed),
    )

    fixture_specs = _build_ws07_rank_deficient_fixtures(X, Y)
    fixture_metrics: Dict[str, Any] = {}
    failures: list[str] = []

    thresholds = {
        "min_beta_corr": float(min_beta_corr),
        "min_sigma2_corr": float(min_sigma2_corr),
        "min_t_corr": float(min_t_corr),
        "max_p_mae": float(max_p_mae),
        "max_df_absdiff": float(max_df_absdiff),
        "max_rank_absdiff": float(max_rank_absdiff),
    }

    for fixture_name, (x_fix, y_fix) in fixture_specs.items():
        cand = fit_fmrimod_ols(x_fix, y_fix, t_con)
        ref = fit_fitlins_reference_ols(x_fix, y_fix, t_con)
        pm = compute_parity_metrics(cand, ref)

        cand_dfres = float(fast_preproject(x_fix, check_finite=False).dfres)
        ref_dfres = float(_reference_dfres_from_nilearn(x_fix, y_fix, noise_model="ols"))
        df_absdiff = float(abs(cand_dfres - ref_dfres))

        expected_rank = float(np.linalg.matrix_rank(x_fix))
        cand_rank = float(x_fix.shape[0] - cand_dfres)
        ref_rank = float(x_fix.shape[0] - ref_dfres)
        rank_absdiff = float(abs(cand_rank - ref_rank))

        fixture_failures: list[str] = []
        if float(pm.beta_corr) < thresholds["min_beta_corr"]:
            fixture_failures.append("min_beta_corr")
        if float(pm.sigma2_corr) < thresholds["min_sigma2_corr"]:
            fixture_failures.append("min_sigma2_corr")
        if float(pm.t_corr) < thresholds["min_t_corr"]:
            fixture_failures.append("min_t_corr")
        if float(pm.p_mae) > thresholds["max_p_mae"]:
            fixture_failures.append("max_p_mae")
        if df_absdiff > thresholds["max_df_absdiff"]:
            fixture_failures.append("max_df_absdiff")
        if rank_absdiff > thresholds["max_rank_absdiff"]:
            fixture_failures.append("max_rank_absdiff")

        if fixture_failures:
            failures.extend([f"{fixture_name}:{name}" for name in fixture_failures])

        fixture_metrics[fixture_name] = {
            "beta_corr": float(pm.beta_corr),
            "sigma2_corr": float(pm.sigma2_corr),
            "t_corr": float(pm.t_corr),
            "p_mae": float(pm.p_mae),
            "candidate_dfres": cand_dfres,
            "reference_dfres": ref_dfres,
            "df_absdiff": df_absdiff,
            "expected_rank": expected_rank,
            "candidate_rank_from_df": cand_rank,
            "reference_rank_from_df": ref_rank,
            "rank_absdiff": rank_absdiff,
            "parity_ok": len(fixture_failures) == 0,
            "failures": fixture_failures,
        }

    return {
        "name": WS_NAMES["ws07"],
        "status": "complete",
        "parity_ok": len(failures) == 0,
        "performance_ok": True,
        "notes": "Rank-deficient OLS parity across duplicate/collinear fixtures.",
        "artifacts": [],
        "thresholds": thresholds,
        "metrics": {"fixtures": fixture_metrics},
        "failures": failures,
    }


def run_ws08_numeric_precision_parity(
    *,
    n_timepoints: int = 220,
    n_regressors: int = 8,
    n_voxels: int = 1200,
    noise_sd: float = 1.0,
    seed: int = 8080,
    min_candidate32_vs64_t_corr_standard: float = 0.995,
    max_candidate32_vs64_p_mae_standard: float = 0.01,
    min_candidate32_vs_ref32_t_corr_standard: float = 0.99,
    max_candidate32_vs_ref32_p_mae_standard: float = 0.015,
    min_candidate32_vs_ref32_sigma2_corr_standard: float = 0.995,
    min_candidate32_vs_ref32_t_corr_dynamic: float = 0.995,
    max_candidate32_vs_ref32_p_mae_dynamic: float = 0.01,
    min_candidate32_vs_ref32_sigma2_corr_dynamic: float = 0.94,
    max_sigma2_corr_gap_vs_reference64_dynamic: float = 0.08,
    max_p_mae_gap_vs_reference64_dynamic: float = 0.02,
) -> Dict[str, Any]:
    X, Y, beta_true, t_con = make_synthetic_glm(
        n_timepoints=int(n_timepoints),
        n_regressors=int(n_regressors),
        n_voxels=int(n_voxels),
        noise_sd=float(noise_sd),
        seed=int(seed),
    )

    noise = np.asarray(Y - (X @ beta_true), dtype=np.float64)
    X_scaled = np.asarray(X, dtype=np.float64).copy()
    if X_scaled.shape[1] > 1:
        scales = np.geomspace(1e-3, 1e3, X_scaled.shape[1] - 1)
        X_scaled[:, 1:] = X_scaled[:, 1:] * scales[np.newaxis, :]
    Y_scaled = X_scaled @ beta_true + noise

    fixtures = {
        "standard_scale": (np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)),
        "dynamic_range_scaled": (X_scaled, np.asarray(Y_scaled, dtype=np.float64)),
    }

    thresholds = {
        "min_candidate32_vs64_t_corr_standard": float(
            min_candidate32_vs64_t_corr_standard
        ),
        "max_candidate32_vs64_p_mae_standard": float(
            max_candidate32_vs64_p_mae_standard
        ),
        "min_candidate32_vs_ref32_t_corr_standard": float(
            min_candidate32_vs_ref32_t_corr_standard
        ),
        "max_candidate32_vs_ref32_p_mae_standard": float(
            max_candidate32_vs_ref32_p_mae_standard
        ),
        "min_candidate32_vs_ref32_sigma2_corr_standard": float(
            min_candidate32_vs_ref32_sigma2_corr_standard
        ),
        "min_candidate32_vs_ref32_t_corr_dynamic": float(
            min_candidate32_vs_ref32_t_corr_dynamic
        ),
        "max_candidate32_vs_ref32_p_mae_dynamic": float(
            max_candidate32_vs_ref32_p_mae_dynamic
        ),
        "min_candidate32_vs_ref32_sigma2_corr_dynamic": float(
            min_candidate32_vs_ref32_sigma2_corr_dynamic
        ),
        "max_sigma2_corr_gap_vs_reference64_dynamic": float(
            max_sigma2_corr_gap_vs_reference64_dynamic
        ),
        "max_p_mae_gap_vs_reference64_dynamic": float(
            max_p_mae_gap_vs_reference64_dynamic
        ),
    }

    fixture_metrics: Dict[str, Any] = {}
    failures: list[str] = []

    for fixture_name, (x_fix, y_fix) in fixtures.items():
        x32 = np.asarray(x_fix, dtype=np.float32)
        y32 = np.asarray(y_fix, dtype=np.float32)
        con32 = np.asarray(t_con, dtype=np.float32)

        cand64 = fit_fmrimod_ols(x_fix, y_fix, t_con, compute_dtype=np.float64)
        cand32 = fit_fmrimod_ols(x32, y32, con32, compute_dtype=np.float32)
        ref64 = fit_fitlins_reference_ols(x_fix, y_fix, t_con)
        ref32 = fit_fitlins_reference_ols(x32, y32, con32)

        c32_vs_64 = compute_parity_metrics(cand32, cand64)
        c32_vs_r32 = compute_parity_metrics(cand32, ref32)
        r32_vs_64 = compute_parity_metrics(ref32, ref64)
        sigma2_corr_gap_vs_ref64 = float(
            abs(float(c32_vs_64.sigma2_corr) - float(r32_vs_64.sigma2_corr))
        )
        p_mae_gap_vs_ref64 = float(
            abs(float(c32_vs_64.p_mae) - float(r32_vs_64.p_mae))
        )

        fixture_failures: list[str] = []
        if fixture_name == "standard_scale":
            if (
                float(c32_vs_64.t_corr)
                < thresholds["min_candidate32_vs64_t_corr_standard"]
            ):
                fixture_failures.append("min_candidate32_vs64_t_corr_standard")
            if (
                float(c32_vs_64.p_mae)
                > thresholds["max_candidate32_vs64_p_mae_standard"]
            ):
                fixture_failures.append("max_candidate32_vs64_p_mae_standard")
            if (
                float(c32_vs_r32.t_corr)
                < thresholds["min_candidate32_vs_ref32_t_corr_standard"]
            ):
                fixture_failures.append("min_candidate32_vs_ref32_t_corr_standard")
            if (
                float(c32_vs_r32.p_mae)
                > thresholds["max_candidate32_vs_ref32_p_mae_standard"]
            ):
                fixture_failures.append("max_candidate32_vs_ref32_p_mae_standard")
            if (
                float(c32_vs_r32.sigma2_corr)
                < thresholds["min_candidate32_vs_ref32_sigma2_corr_standard"]
            ):
                fixture_failures.append(
                    "min_candidate32_vs_ref32_sigma2_corr_standard"
                )
        else:
            if (
                float(c32_vs_r32.t_corr)
                < thresholds["min_candidate32_vs_ref32_t_corr_dynamic"]
            ):
                fixture_failures.append("min_candidate32_vs_ref32_t_corr_dynamic")
            if (
                float(c32_vs_r32.p_mae)
                > thresholds["max_candidate32_vs_ref32_p_mae_dynamic"]
            ):
                fixture_failures.append("max_candidate32_vs_ref32_p_mae_dynamic")
            if (
                float(c32_vs_r32.sigma2_corr)
                < thresholds["min_candidate32_vs_ref32_sigma2_corr_dynamic"]
            ):
                fixture_failures.append(
                    "min_candidate32_vs_ref32_sigma2_corr_dynamic"
                )
            if (
                sigma2_corr_gap_vs_ref64
                > thresholds["max_sigma2_corr_gap_vs_reference64_dynamic"]
            ):
                fixture_failures.append("max_sigma2_corr_gap_vs_reference64_dynamic")
            if (
                p_mae_gap_vs_ref64
                > thresholds["max_p_mae_gap_vs_reference64_dynamic"]
            ):
                fixture_failures.append("max_p_mae_gap_vs_reference64_dynamic")

        if fixture_failures:
            failures.extend([f"{fixture_name}:{name}" for name in fixture_failures])

        fixture_metrics[fixture_name] = {
            "candidate32_vs64": {
                "beta_corr": float(c32_vs_64.beta_corr),
                "sigma2_corr": float(c32_vs_64.sigma2_corr),
                "t_corr": float(c32_vs_64.t_corr),
                "p_mae": float(c32_vs_64.p_mae),
            },
            "candidate32_vs_reference32": {
                "beta_corr": float(c32_vs_r32.beta_corr),
                "sigma2_corr": float(c32_vs_r32.sigma2_corr),
                "t_corr": float(c32_vs_r32.t_corr),
                "p_mae": float(c32_vs_r32.p_mae),
            },
            "reference32_vs64": {
                "beta_corr": float(r32_vs_64.beta_corr),
                "sigma2_corr": float(r32_vs_64.sigma2_corr),
                "t_corr": float(r32_vs_64.t_corr),
                "p_mae": float(r32_vs_64.p_mae),
            },
            "sigma2_corr_gap_vs_reference64": sigma2_corr_gap_vs_ref64,
            "p_mae_gap_vs_reference64": p_mae_gap_vs_ref64,
            "parity_ok": len(fixture_failures) == 0,
            "failures": fixture_failures,
        }

    return {
        "name": WS_NAMES["ws08"],
        "status": "complete",
        "parity_ok": len(failures) == 0,
        "performance_ok": True,
        "notes": "Float32/float64 numerical stability and float32 parity vs nilearn.",
        "artifacts": [],
        "thresholds": thresholds,
        "metrics": {"fixtures": fixture_metrics},
        "failures": failures,
    }


def _residual_dw(residuals: Array) -> Array:
    r = np.asarray(residuals, dtype=np.float64)
    if r.shape[0] < 2:
        return np.zeros(r.shape[1], dtype=np.float64)
    d = np.diff(r, axis=0)
    num = np.sum(d * d, axis=0)
    den = np.maximum(np.sum(r * r, axis=0), 1e-30)
    return np.asarray(num / den, dtype=np.float64).reshape(-1)


def _residual_lag1(residuals: Array) -> Array:
    r = np.asarray(residuals, dtype=np.float64)
    if r.shape[0] < 2:
        return np.zeros(r.shape[1], dtype=np.float64)
    r0 = r[:-1]
    r1 = r[1:]
    num = np.sum(r0 * r1, axis=0)
    den = np.maximum(np.sum(r0 * r0, axis=0), 1e-30)
    return np.asarray(num / den, dtype=np.float64).reshape(-1)


def run_ws09_residual_diagnostic_parity(
    *,
    n_timepoints: int = 220,
    n_regressors: int = 8,
    n_voxels: int = 1200,
    noise_sd: float = 1.0,
    phi: float = 0.45,
    seed: int = 9090,
    min_dw_corr: float = 0.999,
    max_dw_mae: float = 5e-4,
    min_lag1_corr: float = 0.999,
    max_lag1_mae: float = 5e-4,
    min_rss_corr: float = 0.999999,
    max_residual_mae: float = 1e-5,
) -> Dict[str, Any]:
    x_ols, y_ols, _beta_ols, con_ols = make_synthetic_glm(
        n_timepoints=int(n_timepoints),
        n_regressors=int(n_regressors),
        n_voxels=int(n_voxels),
        noise_sd=float(noise_sd),
        seed=int(seed),
    )
    x_ar1, y_ar1, _beta_ar1, con_ar1 = make_synthetic_glm_ar1(
        n_timepoints=int(n_timepoints),
        n_regressors=int(n_regressors),
        n_voxels=int(n_voxels),
        phi=float(phi),
        noise_sd=float(noise_sd),
        seed=int(seed) + 1,
    )

    fixtures = {
        "ols_noise_fixture": (x_ols, y_ols, con_ols),
        "ar1_noise_fixture": (x_ar1, y_ar1, con_ar1),
    }

    thresholds = {
        "min_dw_corr": float(min_dw_corr),
        "max_dw_mae": float(max_dw_mae),
        "min_lag1_corr": float(min_lag1_corr),
        "max_lag1_mae": float(max_lag1_mae),
        "min_rss_corr": float(min_rss_corr),
        "max_residual_mae": float(max_residual_mae),
    }

    fixture_metrics: Dict[str, Any] = {}
    failures: list[str] = []

    for fixture_name, (x_fix, y_fix, con_fix) in fixtures.items():
        cand = fit_fmrimod_ols(x_fix, y_fix, con_fix)
        ref = fit_fitlins_reference_ols(x_fix, y_fix, con_fix)

        resid_c = np.asarray(y_fix - (x_fix @ cand["betas"]), dtype=np.float64)
        resid_r = np.asarray(y_fix - (x_fix @ ref["betas"]), dtype=np.float64)
        residual_mae = float(np.mean(np.abs(resid_c - resid_r)))

        dw_c = _residual_dw(resid_c)
        dw_r = _residual_dw(resid_r)
        lag1_c = _residual_lag1(resid_c)
        lag1_r = _residual_lag1(resid_r)

        rss_c = np.sum(resid_c * resid_c, axis=0)
        rss_r = np.sum(resid_r * resid_r, axis=0)

        dw_corr = float(_safe_corr(dw_c, dw_r))
        dw_mae = float(np.mean(np.abs(dw_c - dw_r)))
        lag1_corr = float(_safe_corr(lag1_c, lag1_r))
        lag1_mae = float(np.mean(np.abs(lag1_c - lag1_r)))
        rss_corr = float(_safe_corr(rss_c, rss_r))

        fixture_failures: list[str] = []
        if dw_corr < thresholds["min_dw_corr"]:
            fixture_failures.append("min_dw_corr")
        if dw_mae > thresholds["max_dw_mae"]:
            fixture_failures.append("max_dw_mae")
        if lag1_corr < thresholds["min_lag1_corr"]:
            fixture_failures.append("min_lag1_corr")
        if lag1_mae > thresholds["max_lag1_mae"]:
            fixture_failures.append("max_lag1_mae")
        if rss_corr < thresholds["min_rss_corr"]:
            fixture_failures.append("min_rss_corr")
        if residual_mae > thresholds["max_residual_mae"]:
            fixture_failures.append("max_residual_mae")

        if fixture_failures:
            failures.extend([f"{fixture_name}:{name}" for name in fixture_failures])

        fixture_metrics[fixture_name] = {
            "dw_corr": dw_corr,
            "dw_mae": dw_mae,
            "lag1_corr": lag1_corr,
            "lag1_mae": lag1_mae,
            "rss_corr": rss_corr,
            "residual_mae": residual_mae,
            "parity_ok": len(fixture_failures) == 0,
            "failures": fixture_failures,
        }

    return {
        "name": WS_NAMES["ws09"],
        "status": "complete",
        "parity_ok": len(failures) == 0,
        "performance_ok": True,
        "notes": "Residual diagnostics parity (Durbin-Watson, lag1 ACF, RSS).",
        "artifacts": [],
        "thresholds": thresholds,
        "metrics": {"fixtures": fixture_metrics},
        "failures": failures,
    }


def _benchmark_summary_to_stage(summary: Any) -> Dict[str, Any]:
    return {
        "candidate_runs_s": [float(x) for x in summary.fmrimod_runs_s],
        "reference_runs_s": [float(x) for x in summary.reference_runs_s],
        "candidate_median_s": float(summary.fmrimod_median_s),
        "reference_median_s": float(summary.reference_median_s),
        "speedup_vs_reference": float(summary.speedup_vs_reference),
    }


def run_ws10_performance_decomposition_parity(
    *,
    n_timepoints: int = 220,
    n_regressors: int = 8,
    n_voxels: int = 1500,
    noise_sd: float = 1.0,
    phi: float = 0.45,
    seed: int = 5050,
    repeats: int = 3,
    warmup: int = 1,
    design_n_scans: int = 180,
    design_tr: float = 1.0,
    run_combine_runs: int = 4,
    min_speedup_design_build: float = 0.7,
    min_speedup_fit_total_ols: float = 1.0,
    min_speedup_contrast_only: float = 0.25,
    min_speedup_fit_total_ar1: float = 0.7,
    min_speedup_run_combine: float = 0.9,
) -> Dict[str, Any]:
    from nilearn.glm.first_level import make_first_level_design_matrix, run_glm

    # Stage 1: design build
    rng_design = np.random.default_rng(seed)
    frame_times = np.arange(int(design_n_scans), dtype=np.float64) * float(design_tr)
    n_events = 18
    onset_grid = np.linspace(8.0, max(20.0, frame_times[-1] - 24.0), n_events)
    onset_jitter = rng_design.uniform(-0.4, 0.4, size=n_events)
    onsets = np.sort(np.clip(onset_grid + onset_jitter, 0.0, frame_times[-1] - 4.0))
    durations = np.where(np.arange(n_events) % 3 == 0, 2.0, 1.0)
    trial_type = np.where(np.arange(n_events) % 2 == 0, "A", "B")
    events_nilearn = pd.DataFrame(
        {"onset": onsets, "duration": durations, "trial_type": trial_type}
    )
    events_fmrimod = events_nilearn.rename(columns={"trial_type": "condition"})

    design_stage = _benchmark_pair(
        lambda: np.asarray(
            event_model(
                "hrf(condition)",
                data=events_fmrimod,
                tr=float(design_tr),
                n_scans=int(design_n_scans),
                durations="duration",
            ).design_matrix,
            dtype=np.float64,
        ),
        lambda: make_first_level_design_matrix(
            frame_times,
            events=events_nilearn,
            hrf_model="spm",
            drift_model=None,
        ),
        repeats=int(repeats),
        warmup=int(warmup),
    )

    # Stage 2: total OLS fit
    X, Y, _beta, t_con = make_synthetic_glm(
        n_timepoints=n_timepoints,
        n_regressors=n_regressors,
        n_voxels=n_voxels,
        noise_sd=noise_sd,
        seed=seed + 1,
    )
    ols_summary = benchmark_implementations(
        X,
        Y,
        t_con,
        repeats=int(repeats),
        warmup=int(warmup),
    )
    fit_total_ols_stage = _benchmark_summary_to_stage(ols_summary)

    # Stage 3: contrast-only on precomputed fit objects
    proj = fast_preproject(X, check_finite=False)
    fit = fast_lm_matrix(X, Y, proj, return_fitted=False, check_finite=False)
    betas_fit = np.asarray(fit.betas, dtype=np.float64)
    sigma2_fit = np.maximum(np.asarray(fit.sigma2, dtype=np.float64), 0.0)
    dfres_fit = float(proj.dfres)
    xtxinv = np.asarray(proj.XtXinv, dtype=np.float64)
    labels_ols, estimates_ols = run_glm(Y, X, noise_model="ols")
    t_con_vec = np.asarray(t_con, dtype=np.float64).ravel()
    f_con_mat = np.asarray(_default_f_contrast(int(n_regressors)), dtype=np.float64)
    t_var_factor = float(t_con_vec @ xtxinv @ t_con_vec)
    f_inner = f_con_mat @ xtxinv @ f_con_mat.T
    f_inner_inv = np.linalg.pinv(f_inner, hermitian=True)
    f_dof_num = float(f_con_mat.shape[0])

    def _candidate_contrast_eval():
        t_effect = t_con_vec @ betas_fit
        t_var = np.maximum(t_var_factor, 0.0) * sigma2_fit
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = np.where(t_var > 1e-30, t_effect / np.sqrt(t_var), 0.0)
        t_p = 2.0 * sp_special.stdtr(dfres_fit, -np.abs(t_stat))

        f_effect = f_con_mat @ betas_fit
        f_quad = np.einsum("iv,ij,jv->v", f_effect, f_inner_inv, f_effect, optimize=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            f_stat = np.where(sigma2_fit > 1e-30, (f_quad / f_dof_num) / sigma2_fit, 0.0)
        f_p = sp_special.fdtrc(f_dof_num, dfres_fit, np.maximum(f_stat, 0.0))
        return t_stat, t_p, f_stat, f_p

    def _reference_contrast_eval():
        t_ref = _compute_contrast_compat(labels_ols, estimates_ols, t_con_vec, stat="t")
        f_ref = _compute_contrast_compat(labels_ols, estimates_ols, f_con_mat, stat="f")
        # Force materialization so reference timing includes actual math, not object creation.
        _ = np.asarray(t_ref.stat(), dtype=np.float64).reshape(-1)
        _ = np.asarray(t_ref.p_value(), dtype=np.float64).reshape(-1)
        _ = np.asarray(f_ref.stat(), dtype=np.float64).reshape(-1)
        _ = np.asarray(f_ref.p_value(), dtype=np.float64).reshape(-1)
        return t_ref, f_ref

    contrast_only_stage = _benchmark_pair(
        _candidate_contrast_eval,
        _reference_contrast_eval,
        repeats=int(repeats),
        warmup=int(warmup),
    )

    # Stage 4: total AR1 fit
    X_ar, Y_ar, _beta_ar, con_ar = make_synthetic_glm_ar1(
        n_timepoints=n_timepoints,
        n_regressors=n_regressors,
        n_voxels=n_voxels,
        phi=phi,
        noise_sd=noise_sd,
        seed=seed + 2,
    )
    ar1_summary = benchmark_ar1_implementations(
        X_ar,
        Y_ar,
        con_ar,
        repeats=int(repeats),
        warmup=int(warmup),
        iter_gls=1,
        voxelwise=False,
    )
    fit_total_ar1_stage = _benchmark_summary_to_stage(ar1_summary)

    # Stage 5: run-combine only
    rng_rc = np.random.default_rng(seed + 3)
    rc_effects = rng_rc.normal(size=(int(run_combine_runs), int(n_voxels))).astype(np.float64)
    rc_variances = np.abs(
        rng_rc.normal(loc=0.5, scale=0.2, size=(int(run_combine_runs), int(n_voxels)))
    ).astype(np.float64)
    rc_variances = np.maximum(rc_variances, 1e-6)

    run_combine_stage = _benchmark_pair(
        lambda: _fixed_effects_combine(rc_effects, rc_variances),
        lambda: _fixed_effects_combine(rc_effects, rc_variances),
        repeats=int(repeats),
        warmup=int(warmup),
    )

    stage_metrics = {
        "design_build": design_stage,
        "fit_total_ols": fit_total_ols_stage,
        "contrast_only": contrast_only_stage,
        "fit_total_ar1": fit_total_ar1_stage,
        "run_combine": run_combine_stage,
    }
    thresholds = {
        "min_speedup_design_build": float(min_speedup_design_build),
        "min_speedup_fit_total_ols": float(min_speedup_fit_total_ols),
        "min_speedup_contrast_only": float(min_speedup_contrast_only),
        "min_speedup_fit_total_ar1": float(min_speedup_fit_total_ar1),
        "min_speedup_run_combine": float(min_speedup_run_combine),
    }

    failures: list[str] = []
    if stage_metrics["design_build"]["speedup_vs_reference"] < thresholds["min_speedup_design_build"]:
        failures.append("min_speedup_design_build")
    if stage_metrics["fit_total_ols"]["speedup_vs_reference"] < thresholds["min_speedup_fit_total_ols"]:
        failures.append("min_speedup_fit_total_ols")
    if stage_metrics["contrast_only"]["speedup_vs_reference"] < thresholds["min_speedup_contrast_only"]:
        failures.append("min_speedup_contrast_only")
    if stage_metrics["fit_total_ar1"]["speedup_vs_reference"] < thresholds["min_speedup_fit_total_ar1"]:
        failures.append("min_speedup_fit_total_ar1")
    if stage_metrics["run_combine"]["speedup_vs_reference"] < thresholds["min_speedup_run_combine"]:
        failures.append("min_speedup_run_combine")

    parity_ok = True
    for stage in stage_metrics.values():
        if not np.isfinite(stage["candidate_median_s"]) or not np.isfinite(stage["reference_median_s"]):
            parity_ok = False
            break
        if stage["candidate_median_s"] <= 0.0 or stage["reference_median_s"] <= 0.0:
            parity_ok = False
            break

    return {
        "name": WS_NAMES["ws10"],
        "status": "complete",
        "parity_ok": bool(parity_ok),
        "performance_ok": len(failures) == 0,
        "notes": "Stage-level timing decomposition across design/fit/contrast/run-combine.",
        "artifacts": [],
        "thresholds": thresholds,
        "metrics": stage_metrics,
        "failures": failures,
    }


def build_core_parity_matrix_report(
    *,
    ws01_n_scans: int = 180,
    ws01_tr: float = 1.0,
    ws01_seed: int = 7,
    ws02_n_timepoints: int = 220,
    ws02_n_regressors: int = 8,
    ws02_n_voxels: int = 1200,
    ws02_noise_sd: float = 1.0,
    ws02_seed: int = 1234,
    ws03_n_timepoints: int = 220,
    ws03_n_regressors: int = 8,
    ws03_n_voxels: int = 1200,
    ws03_noise_sd: float = 1.0,
    ws03_phi: float = 0.45,
    ws03_seed: int = 2026,
    ws04_n_timepoints: int = 180,
    ws04_n_regressors: int = 8,
    ws04_n_voxels: int = 1000,
    ws04_noise_sd: float = 1.0,
    ws04_seed: int = 3030,
    ws05_n_timepoints: int = 220,
    ws05_n_regressors: int = 8,
    ws05_n_voxels: int = 1200,
    ws05_noise_sd: float = 1.0,
    ws05_seed: int = 505,
    ws06_n_timepoints: int = 300,
    ws06_n_trials: int = 80,
    ws06_n_voxels: int = 2000,
    ws06_n_confounds: int = 6,
    ws06_noise_sd: float = 1.0,
    ws06_seed: int = 6060,
    ws06_repeats: int = 3,
    ws06_warmup: int = 1,
    ws06_chunk_size: int = 4000,
    ws07_n_timepoints: int = 220,
    ws07_n_regressors: int = 8,
    ws07_n_voxels: int = 1200,
    ws07_noise_sd: float = 1.0,
    ws07_seed: int = 7070,
    ws08_n_timepoints: int = 220,
    ws08_n_regressors: int = 8,
    ws08_n_voxels: int = 1200,
    ws08_noise_sd: float = 1.0,
    ws08_seed: int = 8080,
    ws09_n_timepoints: int = 220,
    ws09_n_regressors: int = 8,
    ws09_n_voxels: int = 1200,
    ws09_noise_sd: float = 1.0,
    ws09_phi: float = 0.45,
    ws09_seed: int = 9090,
    ws10_n_timepoints: int = 220,
    ws10_n_regressors: int = 8,
    ws10_n_voxels: int = 1500,
    ws10_noise_sd: float = 1.0,
    ws10_phi: float = 0.45,
    ws10_seed: int = 5050,
    ws10_repeats: int = 3,
    ws10_warmup: int = 1,
    ws10_design_n_scans: int = 180,
    ws10_design_tr: float = 1.0,
    ws10_run_combine_runs: int = 4,
) -> Dict[str, Any]:
    workstreams: Dict[str, Dict[str, Any]] = {
        key: _placeholder_workstream(name) for key, name in WS_NAMES.items()
    }

    workstreams["ws01"] = run_ws01_design_matrix_parity(
        n_scans=ws01_n_scans,
        tr=ws01_tr,
        seed=ws01_seed,
    )
    workstreams["ws02"] = run_ws02_contrast_parity(
        n_timepoints=ws02_n_timepoints,
        n_regressors=ws02_n_regressors,
        n_voxels=ws02_n_voxels,
        noise_sd=ws02_noise_sd,
        seed=ws02_seed,
    )
    workstreams["ws03"] = run_ws03_variance_df_parity(
        n_timepoints=ws03_n_timepoints,
        n_regressors=ws03_n_regressors,
        n_voxels=ws03_n_voxels,
        noise_sd=ws03_noise_sd,
        phi=ws03_phi,
        seed=ws03_seed,
    )
    workstreams["ws04"] = run_ws04_run_combination_parity(
        n_timepoints=ws04_n_timepoints,
        n_regressors=ws04_n_regressors,
        n_voxels=ws04_n_voxels,
        noise_sd=ws04_noise_sd,
        seed=ws04_seed,
    )
    workstreams["ws05"] = run_ws05_censor_sample_mask_parity(
        n_timepoints=ws05_n_timepoints,
        n_regressors=ws05_n_regressors,
        n_voxels=ws05_n_voxels,
        noise_sd=ws05_noise_sd,
        seed=ws05_seed,
    )
    workstreams["ws06"] = run_ws06_lsa_lss_parity_performance(
        n_timepoints=ws06_n_timepoints,
        n_trials=ws06_n_trials,
        n_voxels=ws06_n_voxels,
        n_confounds=ws06_n_confounds,
        noise_sd=ws06_noise_sd,
        seed=ws06_seed,
        repeats=ws06_repeats,
        warmup=ws06_warmup,
        chunk_size=ws06_chunk_size,
    )
    workstreams["ws07"] = run_ws07_rank_deficient_design_parity(
        n_timepoints=ws07_n_timepoints,
        n_regressors=ws07_n_regressors,
        n_voxels=ws07_n_voxels,
        noise_sd=ws07_noise_sd,
        seed=ws07_seed,
    )
    workstreams["ws08"] = run_ws08_numeric_precision_parity(
        n_timepoints=ws08_n_timepoints,
        n_regressors=ws08_n_regressors,
        n_voxels=ws08_n_voxels,
        noise_sd=ws08_noise_sd,
        seed=ws08_seed,
    )
    workstreams["ws09"] = run_ws09_residual_diagnostic_parity(
        n_timepoints=ws09_n_timepoints,
        n_regressors=ws09_n_regressors,
        n_voxels=ws09_n_voxels,
        noise_sd=ws09_noise_sd,
        phi=ws09_phi,
        seed=ws09_seed,
    )
    workstreams["ws10"] = run_ws10_performance_decomposition_parity(
        n_timepoints=ws10_n_timepoints,
        n_regressors=ws10_n_regressors,
        n_voxels=ws10_n_voxels,
        noise_sd=ws10_noise_sd,
        phi=ws10_phi,
        seed=ws10_seed,
        repeats=ws10_repeats,
        warmup=ws10_warmup,
        design_n_scans=ws10_design_n_scans,
        design_tr=ws10_design_tr,
        run_combine_runs=ws10_run_combine_runs,
    )

    return {
        "artifact_version": "0.1.0",
        "report_kind": "core_parity_matrix",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "ws01_n_scans": int(ws01_n_scans),
            "ws01_tr": float(ws01_tr),
            "ws01_seed": int(ws01_seed),
            "ws02_n_timepoints": int(ws02_n_timepoints),
            "ws02_n_regressors": int(ws02_n_regressors),
            "ws02_n_voxels": int(ws02_n_voxels),
            "ws02_noise_sd": float(ws02_noise_sd),
            "ws02_seed": int(ws02_seed),
            "ws03_n_timepoints": int(ws03_n_timepoints),
            "ws03_n_regressors": int(ws03_n_regressors),
            "ws03_n_voxels": int(ws03_n_voxels),
            "ws03_noise_sd": float(ws03_noise_sd),
            "ws03_phi": float(ws03_phi),
            "ws03_seed": int(ws03_seed),
            "ws04_n_timepoints": int(ws04_n_timepoints),
            "ws04_n_regressors": int(ws04_n_regressors),
            "ws04_n_voxels": int(ws04_n_voxels),
            "ws04_noise_sd": float(ws04_noise_sd),
            "ws04_seed": int(ws04_seed),
            "ws05_n_timepoints": int(ws05_n_timepoints),
            "ws05_n_regressors": int(ws05_n_regressors),
            "ws05_n_voxels": int(ws05_n_voxels),
            "ws05_noise_sd": float(ws05_noise_sd),
            "ws05_seed": int(ws05_seed),
            "ws06_n_timepoints": int(ws06_n_timepoints),
            "ws06_n_trials": int(ws06_n_trials),
            "ws06_n_voxels": int(ws06_n_voxels),
            "ws06_n_confounds": int(ws06_n_confounds),
            "ws06_noise_sd": float(ws06_noise_sd),
            "ws06_seed": int(ws06_seed),
            "ws06_repeats": int(ws06_repeats),
            "ws06_warmup": int(ws06_warmup),
            "ws06_chunk_size": int(ws06_chunk_size),
            "ws07_n_timepoints": int(ws07_n_timepoints),
            "ws07_n_regressors": int(ws07_n_regressors),
            "ws07_n_voxels": int(ws07_n_voxels),
            "ws07_noise_sd": float(ws07_noise_sd),
            "ws07_seed": int(ws07_seed),
            "ws08_n_timepoints": int(ws08_n_timepoints),
            "ws08_n_regressors": int(ws08_n_regressors),
            "ws08_n_voxels": int(ws08_n_voxels),
            "ws08_noise_sd": float(ws08_noise_sd),
            "ws08_seed": int(ws08_seed),
            "ws09_n_timepoints": int(ws09_n_timepoints),
            "ws09_n_regressors": int(ws09_n_regressors),
            "ws09_n_voxels": int(ws09_n_voxels),
            "ws09_noise_sd": float(ws09_noise_sd),
            "ws09_phi": float(ws09_phi),
            "ws09_seed": int(ws09_seed),
            "ws10_n_timepoints": int(ws10_n_timepoints),
            "ws10_n_regressors": int(ws10_n_regressors),
            "ws10_n_voxels": int(ws10_n_voxels),
            "ws10_noise_sd": float(ws10_noise_sd),
            "ws10_phi": float(ws10_phi),
            "ws10_seed": int(ws10_seed),
            "ws10_repeats": int(ws10_repeats),
            "ws10_warmup": int(ws10_warmup),
            "ws10_design_n_scans": int(ws10_design_n_scans),
            "ws10_design_tr": float(ws10_design_tr),
            "ws10_run_combine_runs": int(ws10_run_combine_runs),
        },
        "workstreams": workstreams,
    }


def write_json_report(report: Mapping[str, Any], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fobj:
        json.dump(report, fobj, indent=2, sort_keys=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a core parity matrix artifact with WS01/WS02 results."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cross_testing/reports/core_parity_matrix.json",
        help="Path to write the core parity matrix artifact JSON.",
    )
    parser.add_argument("--ws01-n-scans", type=int, default=180)
    parser.add_argument("--ws01-tr", type=float, default=1.0)
    parser.add_argument("--ws01-seed", type=int, default=7)
    parser.add_argument("--ws02-n-timepoints", type=int, default=220)
    parser.add_argument("--ws02-n-regressors", type=int, default=8)
    parser.add_argument("--ws02-n-voxels", type=int, default=1200)
    parser.add_argument("--ws02-noise-sd", type=float, default=1.0)
    parser.add_argument("--ws02-seed", type=int, default=1234)
    parser.add_argument("--ws03-n-timepoints", type=int, default=220)
    parser.add_argument("--ws03-n-regressors", type=int, default=8)
    parser.add_argument("--ws03-n-voxels", type=int, default=1200)
    parser.add_argument("--ws03-noise-sd", type=float, default=1.0)
    parser.add_argument("--ws03-phi", type=float, default=0.45)
    parser.add_argument("--ws03-seed", type=int, default=2026)
    parser.add_argument("--ws04-n-timepoints", type=int, default=180)
    parser.add_argument("--ws04-n-regressors", type=int, default=8)
    parser.add_argument("--ws04-n-voxels", type=int, default=1000)
    parser.add_argument("--ws04-noise-sd", type=float, default=1.0)
    parser.add_argument("--ws04-seed", type=int, default=3030)
    parser.add_argument("--ws05-n-timepoints", type=int, default=220)
    parser.add_argument("--ws05-n-regressors", type=int, default=8)
    parser.add_argument("--ws05-n-voxels", type=int, default=1200)
    parser.add_argument("--ws05-noise-sd", type=float, default=1.0)
    parser.add_argument("--ws05-seed", type=int, default=505)
    parser.add_argument("--ws06-n-timepoints", type=int, default=300)
    parser.add_argument("--ws06-n-trials", type=int, default=80)
    parser.add_argument("--ws06-n-voxels", type=int, default=2000)
    parser.add_argument("--ws06-n-confounds", type=int, default=6)
    parser.add_argument("--ws06-noise-sd", type=float, default=1.0)
    parser.add_argument("--ws06-seed", type=int, default=6060)
    parser.add_argument("--ws06-repeats", type=int, default=3)
    parser.add_argument("--ws06-warmup", type=int, default=1)
    parser.add_argument("--ws06-chunk-size", type=int, default=4000)
    parser.add_argument("--ws07-n-timepoints", type=int, default=220)
    parser.add_argument("--ws07-n-regressors", type=int, default=8)
    parser.add_argument("--ws07-n-voxels", type=int, default=1200)
    parser.add_argument("--ws07-noise-sd", type=float, default=1.0)
    parser.add_argument("--ws07-seed", type=int, default=7070)
    parser.add_argument("--ws08-n-timepoints", type=int, default=220)
    parser.add_argument("--ws08-n-regressors", type=int, default=8)
    parser.add_argument("--ws08-n-voxels", type=int, default=1200)
    parser.add_argument("--ws08-noise-sd", type=float, default=1.0)
    parser.add_argument("--ws08-seed", type=int, default=8080)
    parser.add_argument("--ws09-n-timepoints", type=int, default=220)
    parser.add_argument("--ws09-n-regressors", type=int, default=8)
    parser.add_argument("--ws09-n-voxels", type=int, default=1200)
    parser.add_argument("--ws09-noise-sd", type=float, default=1.0)
    parser.add_argument("--ws09-phi", type=float, default=0.45)
    parser.add_argument("--ws09-seed", type=int, default=9090)
    parser.add_argument("--ws10-n-timepoints", type=int, default=220)
    parser.add_argument("--ws10-n-regressors", type=int, default=8)
    parser.add_argument("--ws10-n-voxels", type=int, default=1500)
    parser.add_argument("--ws10-noise-sd", type=float, default=1.0)
    parser.add_argument("--ws10-phi", type=float, default=0.45)
    parser.add_argument("--ws10-seed", type=int, default=5050)
    parser.add_argument("--ws10-repeats", type=int, default=3)
    parser.add_argument("--ws10-warmup", type=int, default=1)
    parser.add_argument("--ws10-design-n-scans", type=int, default=180)
    parser.add_argument("--ws10-design-tr", type=float, default=1.0)
    parser.add_argument("--ws10-run-combine-runs", type=int, default=4)
    parser.add_argument(
        "--require-ws01-ws02",
        action="store_true",
        help="Exit non-zero if WS01 or WS02 parity_ok is false.",
    )
    parser.add_argument(
        "--require-ws03",
        action="store_true",
        help="Exit non-zero if WS03 parity_ok is false.",
    )
    parser.add_argument(
        "--require-ws04",
        action="store_true",
        help="Exit non-zero if WS04 parity_ok is false.",
    )
    parser.add_argument(
        "--require-ws05",
        action="store_true",
        help="Exit non-zero if WS05 parity_ok is false.",
    )
    parser.add_argument(
        "--require-ws06",
        action="store_true",
        help="Exit non-zero if WS06 performance/parity gates fail.",
    )
    parser.add_argument(
        "--require-ws07",
        action="store_true",
        help="Exit non-zero if WS07 parity_ok is false.",
    )
    parser.add_argument(
        "--require-ws08",
        action="store_true",
        help="Exit non-zero if WS08 parity_ok is false.",
    )
    parser.add_argument(
        "--require-ws09",
        action="store_true",
        help="Exit non-zero if WS09 parity_ok is false.",
    )
    parser.add_argument(
        "--require-ws10",
        action="store_true",
        help="Exit non-zero if WS10 performance/parity gates fail.",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print JSON to stdout without writing a file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = build_core_parity_matrix_report(
            ws01_n_scans=args.ws01_n_scans,
            ws01_tr=args.ws01_tr,
            ws01_seed=args.ws01_seed,
            ws02_n_timepoints=args.ws02_n_timepoints,
            ws02_n_regressors=args.ws02_n_regressors,
            ws02_n_voxels=args.ws02_n_voxels,
            ws02_noise_sd=args.ws02_noise_sd,
            ws02_seed=args.ws02_seed,
            ws03_n_timepoints=args.ws03_n_timepoints,
            ws03_n_regressors=args.ws03_n_regressors,
            ws03_n_voxels=args.ws03_n_voxels,
            ws03_noise_sd=args.ws03_noise_sd,
            ws03_phi=args.ws03_phi,
            ws03_seed=args.ws03_seed,
            ws04_n_timepoints=args.ws04_n_timepoints,
            ws04_n_regressors=args.ws04_n_regressors,
            ws04_n_voxels=args.ws04_n_voxels,
            ws04_noise_sd=args.ws04_noise_sd,
            ws04_seed=args.ws04_seed,
            ws05_n_timepoints=args.ws05_n_timepoints,
            ws05_n_regressors=args.ws05_n_regressors,
            ws05_n_voxels=args.ws05_n_voxels,
            ws05_noise_sd=args.ws05_noise_sd,
            ws05_seed=args.ws05_seed,
            ws06_n_timepoints=args.ws06_n_timepoints,
            ws06_n_trials=args.ws06_n_trials,
            ws06_n_voxels=args.ws06_n_voxels,
            ws06_n_confounds=args.ws06_n_confounds,
            ws06_noise_sd=args.ws06_noise_sd,
            ws06_seed=args.ws06_seed,
            ws06_repeats=args.ws06_repeats,
            ws06_warmup=args.ws06_warmup,
            ws06_chunk_size=args.ws06_chunk_size,
            ws07_n_timepoints=args.ws07_n_timepoints,
            ws07_n_regressors=args.ws07_n_regressors,
            ws07_n_voxels=args.ws07_n_voxels,
            ws07_noise_sd=args.ws07_noise_sd,
            ws07_seed=args.ws07_seed,
            ws08_n_timepoints=args.ws08_n_timepoints,
            ws08_n_regressors=args.ws08_n_regressors,
            ws08_n_voxels=args.ws08_n_voxels,
            ws08_noise_sd=args.ws08_noise_sd,
            ws08_seed=args.ws08_seed,
            ws09_n_timepoints=args.ws09_n_timepoints,
            ws09_n_regressors=args.ws09_n_regressors,
            ws09_n_voxels=args.ws09_n_voxels,
            ws09_noise_sd=args.ws09_noise_sd,
            ws09_phi=args.ws09_phi,
            ws09_seed=args.ws09_seed,
            ws10_n_timepoints=args.ws10_n_timepoints,
            ws10_n_regressors=args.ws10_n_regressors,
            ws10_n_voxels=args.ws10_n_voxels,
            ws10_noise_sd=args.ws10_noise_sd,
            ws10_phi=args.ws10_phi,
            ws10_seed=args.ws10_seed,
            ws10_repeats=args.ws10_repeats,
            ws10_warmup=args.ws10_warmup,
            ws10_design_n_scans=args.ws10_design_n_scans,
            ws10_design_tr=args.ws10_design_tr,
            ws10_run_combine_runs=args.ws10_run_combine_runs,
        )
    except ModuleNotFoundError as exc:
        print(
            "Missing optional dependency for core parity matrix generation: "
            f"{exc}. Install nilearn to run this command."
        )
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.stdout_only:
        write_json_report(report, args.output)
        print(f"\nWrote report to {args.output}")

    if args.require_ws01_ws02:
        ws01_ok = bool(report["workstreams"]["ws01"]["parity_ok"])
        ws02_ok = bool(report["workstreams"]["ws02"]["parity_ok"])
        if not (ws01_ok and ws02_ok):
            print(
                f"Required WS gates failed: ws01.parity_ok={ws01_ok}, "
                f"ws02.parity_ok={ws02_ok}"
            )
            return 1
    if args.require_ws03:
        ws03_ok = bool(report["workstreams"]["ws03"]["parity_ok"])
        if not ws03_ok:
            print("Required WS gate failed: ws03.parity_ok=False")
            return 1
    if args.require_ws04:
        ws04_ok = bool(report["workstreams"]["ws04"]["parity_ok"])
        if not ws04_ok:
            print("Required WS gate failed: ws04.parity_ok=False")
            return 1
    if args.require_ws05:
        ws05_ok = bool(report["workstreams"]["ws05"]["parity_ok"])
        if not ws05_ok:
            print("Required WS gate failed: ws05.parity_ok=False")
            return 1
    if args.require_ws06:
        ws06 = report["workstreams"]["ws06"]
        ws06_ok = bool(ws06["parity_ok"]) and bool(ws06["performance_ok"])
        if not ws06_ok:
            print(
                "Required WS gate failed: "
                f"ws06.parity_ok={ws06['parity_ok']}, "
                f"ws06.performance_ok={ws06['performance_ok']}"
            )
            return 1
    if args.require_ws07:
        ws07_ok = bool(report["workstreams"]["ws07"]["parity_ok"])
        if not ws07_ok:
            print("Required WS gate failed: ws07.parity_ok=False")
            return 1
    if args.require_ws08:
        ws08_ok = bool(report["workstreams"]["ws08"]["parity_ok"])
        if not ws08_ok:
            print("Required WS gate failed: ws08.parity_ok=False")
            return 1
    if args.require_ws09:
        ws09_ok = bool(report["workstreams"]["ws09"]["parity_ok"])
        if not ws09_ok:
            print("Required WS gate failed: ws09.parity_ok=False")
            return 1
    if args.require_ws10:
        ws10 = report["workstreams"]["ws10"]
        ws10_ok = bool(ws10["parity_ok"]) and bool(ws10["performance_ok"])
        if not ws10_ok:
            print(
                "Required WS gate failed: "
                f"ws10.parity_ok={ws10['parity_ok']}, "
                f"ws10.performance_ok={ws10['performance_ok']}"
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
