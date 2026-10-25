"""Exact leave-one-out diagnostic kernels for group examination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._examine_control import ExaminationTolerance
from ._reducers_kernels import _fe_weights_and_q, _safe_inverse

ScreeningMode = Literal["exact", "tau2_fixed_full"]


@dataclass
class BlockDiagnostic:
    """Per-contrast leave-one-out maps.

    Arrays are subject × sample unless noted. ``delta_*`` / ``influence_eligible``
    are subject × estimand × sample.
    """

    expected: NDArray[np.float64]
    predictive_resid: NDArray[np.float64]
    predictive_weight: NDArray[np.float64]
    leverage: NDArray[np.float64]
    delta_effect: NDArray[np.float64]
    delta_stat: NDArray[np.float64]
    deleted_stat: NDArray[np.float64]
    surprise_eligible: NDArray[np.bool_]
    influence_eligible: NDArray[np.bool_]
    full_effect: NDArray[np.float64]
    full_se: NDArray[np.float64]
    full_stat: NDArray[np.float64]
    coverage: NDArray[np.intp]
    tau2: NDArray[np.float64] | None
    mode: ScreeningMode
    estimand_names: tuple[str, ...]


def _empty(
    n_subject: int, n_sample: int, n_estimand: int
) -> dict[str, NDArray[np.floating]]:
    return {
        "expected": np.full((n_subject, n_sample), np.nan),
        "predictive_resid": np.full((n_subject, n_sample), np.nan),
        "predictive_weight": np.full((n_subject, n_sample), np.nan),
        "leverage": np.full((n_subject, n_sample), np.nan),
        "delta_effect": np.full((n_subject, n_estimand, n_sample), np.nan),
        "delta_stat": np.full((n_subject, n_estimand, n_sample), np.nan),
        "deleted_stat": np.full((n_subject, n_estimand, n_sample), np.nan),
        "surprise_eligible": np.zeros((n_subject, n_sample), dtype=bool),
        "influence_eligible": np.zeros((n_subject, n_estimand, n_sample), dtype=bool),
        "full_effect": np.full((n_estimand, n_sample), np.nan),
        "full_se": np.full((n_estimand, n_sample), np.nan),
        "full_stat": np.full((n_estimand, n_sample), np.nan),
        "coverage": np.zeros(n_sample, dtype=np.intp),
    }


def diagnose_ivw(
    beta: NDArray[np.float64],
    var: NDArray[np.float64],
    *,
    tau2: NDArray[np.float64] | None = None,
    tolerance: ExaminationTolerance,
    mode: ScreeningMode = "exact",
) -> BlockDiagnostic:
    """Exact inverse-variance leave-one-out for ``meta:fe`` / ``meta:re``.

    ``beta`` / ``var`` are sample × subject. Optional ``tau2`` is length
    ``n_sample`` and holds the *full-sample* DL estimate (fixed during LOO).
    """
    y = np.asarray(beta, dtype=np.float64)
    v = np.asarray(var, dtype=np.float64)
    if y.ndim != 2 or v.shape != y.shape:
        raise ValueError("beta and var must be sample × subject arrays")
    n_sample, n_subject = y.shape
    slots = _empty(n_subject, n_sample, 1)
    extra = (
        np.zeros(n_sample, dtype=np.float64)
        if tau2 is None
        else np.asarray(tau2, dtype=np.float64).reshape(n_sample)
    )
    degeneracy = float(tolerance.degeneracy)
    for b in range(n_sample):
        yb = y[b]
        vb = v[b]
        tau_b = extra[b]
        valid = np.isfinite(yb) & np.isfinite(vb) & (vb > 0)
        if tau2 is not None:
            valid = valid & np.isfinite(tau_b) & (tau_b >= 0)
        slots["coverage"][b] = int(np.count_nonzero(valid))
        if int(slots["coverage"][b]) < 2:
            continue
        w = np.full(n_subject, np.nan)
        w[valid] = 1.0 / (vb[valid] + (0.0 if tau2 is None else tau_b))
        weight_sum = float(np.sum(w[valid]))
        sum_wy = float(np.sum(w[valid] * yb[valid]))
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            continue
        mu = sum_wy / weight_sum
        se = np.sqrt(1.0 / weight_sum)
        z = mu / se
        slots["full_effect"][0, b] = mu
        slots["full_se"][0, b] = se
        slots["full_stat"][0, b] = z
        for i in np.flatnonzero(valid):
            w_i = float(w[i])
            w_minus = weight_sum - w_i
            if not np.isfinite(w_minus) or w_minus <= degeneracy:
                continue
            mu_minus = (sum_wy - w_i * float(yb[i])) / w_minus
            se_minus = np.sqrt(1.0 / w_minus)
            z_minus = mu_minus / se_minus
            pred_var = float(vb[i]) + (0.0 if tau2 is None else tau_b) + 1.0 / w_minus
            if not np.isfinite(pred_var) or pred_var <= degeneracy:
                continue
            slots["expected"][i, b] = mu_minus
            slots["predictive_resid"][i, b] = (float(yb[i]) - mu_minus) / np.sqrt(
                pred_var
            )
            slots["predictive_weight"][i, b] = 1.0 / pred_var
            slots["leverage"][i, b] = w_i / weight_sum
            slots["delta_effect"][i, 0, b] = mu - mu_minus
            slots["deleted_stat"][i, 0, b] = z_minus
            slots["delta_stat"][i, 0, b] = z - z_minus
            slots["surprise_eligible"][i, b] = True
            slots["influence_eligible"][i, 0, b] = True
    return BlockDiagnostic(
        expected=slots["expected"],
        predictive_resid=slots["predictive_resid"],
        predictive_weight=slots["predictive_weight"],
        leverage=slots["leverage"],
        delta_effect=slots["delta_effect"],
        delta_stat=slots["delta_stat"],
        deleted_stat=slots["deleted_stat"],
        surprise_eligible=slots["surprise_eligible"].astype(bool),
        influence_eligible=slots["influence_eligible"].astype(bool),
        full_effect=slots["full_effect"],
        full_se=slots["full_se"],
        full_stat=slots["full_stat"],
        coverage=slots["coverage"].astype(np.intp),
        tau2=None if tau2 is None else extra.copy(),
        mode=mode,
        estimand_names=("pooled_effect",),
    )


def dl_tau2(
    beta: NDArray[np.float64], var: NDArray[np.float64], *, eps: float = 1e-12
) -> NDArray[np.float64]:
    """DerSimonian-Laird ``tau2`` for sample × subject arrays."""
    y = np.asarray(beta, dtype=np.float64)
    v = np.asarray(var, dtype=np.float64)
    # kernels expect sample × subject × contrast
    y3 = y[:, :, np.newaxis]
    v3 = v[:, :, np.newaxis]
    w_fe, sw_fe, q, k = _fe_weights_and_q(y3, v3, eps=eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        c_term = sw_fe - np.sum(w_fe * w_fe, axis=1, keepdims=True) / sw_fe
        tau2 = np.maximum(0.0, (q - (k - 1.0)) / np.maximum(c_term, eps))
    return tau2[:, 0]


def diagnose_linear(
    beta: NDArray[np.float64],
    var: NDArray[np.float64] | None,
    X: NDArray[np.float64],
    estimands: NDArray[np.float64],
    estimand_names: tuple[str, ...],
    *,
    tau2: NDArray[np.float64] | None,
    tolerance: ExaminationTolerance,
    mode: ScreeningMode,
    min_obs: int | None = None,
) -> BlockDiagnostic:
    """Weighted-least-squares leave-one-out via Sherman-Morrison."""
    y = np.asarray(beta, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("beta must be sample × subject")
    n_sample, n_subject = y.shape
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] != n_subject:
        raise ValueError("X rows must match the subject axis")
    C = np.asarray(estimands, dtype=np.float64)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.shape[1] != X.shape[1]:
        raise ValueError("estimands columns must match design columns")
    n_estimand = C.shape[0]
    n_coef = X.shape[1]
    slots = _empty(n_subject, n_sample, n_estimand)
    extra = (
        None if tau2 is None else np.asarray(tau2, dtype=np.float64).reshape(n_sample)
    )
    v = None if var is None else np.asarray(var, dtype=np.float64)
    min_needed = n_coef + 1 if min_obs is None else max(int(min_obs), n_coef + 1)
    finite_design = np.all(np.isfinite(X), axis=1)
    rank_tol = float(tolerance.rank)
    lev_tol = float(tolerance.leverage)
    degeneracy = float(tolerance.degeneracy)

    for b in range(n_sample):
        yb = y[b]
        valid = np.isfinite(yb) & finite_design
        tau_b = 0.0
        if extra is not None:
            tau_b = float(extra[b])
            if not np.isfinite(tau_b) or tau_b < 0:
                continue
        if v is not None:
            vb = v[b]
            valid = valid & np.isfinite(vb) & (vb > 0)
        else:
            vb = np.ones(n_subject, dtype=np.float64)
        idx = np.flatnonzero(valid)
        slots["coverage"][b] = int(idx.size)
        if idx.size < min_needed:
            continue
        Xv = X[idx]
        if int(np.linalg.matrix_rank(Xv, tol=rank_tol)) < n_coef:
            continue
        yv = yb[idx]
        wv = 1.0 / (vb[idx] + tau_b) if v is not None else np.ones(idx.size)
        Xw = Xv * np.sqrt(wv)[:, np.newaxis]
        gram_inv = _safe_inverse(Xw.T @ Xw)
        if gram_inv is None:
            continue
        theta = gram_inv @ (Xv.T @ (wv * yv))
        residual = yv - Xv @ theta
        leverage = wv * np.sum((Xv @ gram_inv) * Xv, axis=1)
        df = float(idx.size - n_coef)
        sse = float(np.sum(wv * residual * residual))
        scale_full = sse / df if v is None else 1.0
        if not np.isfinite(scale_full) or scale_full < 0:
            continue
        est_var = np.sum((C @ gram_inv) * C, axis=1) * scale_full
        psi = C @ theta
        slots["full_effect"][:, b] = psi
        ok_est = np.isfinite(est_var) & (est_var > degeneracy)
        slots["full_se"][ok_est, b] = np.sqrt(est_var[ok_est])
        slots["full_stat"][ok_est, b] = psi[ok_est] / slots["full_se"][ok_est, b]

        for j, i in enumerate(idx):
            h = float(leverage[j])
            one_minus_h = 1.0 - h
            if not np.isfinite(one_minus_h) or one_minus_h <= lev_tol:
                continue
            keep: NDArray[np.bool_] = np.ones(idx.size, dtype=bool)
            keep[j] = False
            if int(np.count_nonzero(keep)) < min_needed:
                continue
            if int(np.linalg.matrix_rank(Xv[keep], tol=rank_tol)) < n_coef:
                continue
            ax = gram_inv @ Xv[j]
            delta_theta = ax * wv[j] * residual[j] / one_minus_h
            theta_minus = theta - delta_theta
            a_minus = gram_inv + (wv[j] / one_minus_h) * np.outer(ax, ax)
            expected = float(Xv[j] @ theta_minus)
            if v is None:
                sse_minus = sse - wv[j] * residual[j] ** 2 / one_minus_h
                df_minus = df - 1.0
                scale = sse_minus / df_minus if df_minus > 0 else np.nan
            else:
                scale = 1.0
            if not np.isfinite(scale) or scale < 0:
                continue
            pred_var = float(vb[i] + tau_b + scale * (Xv[j] @ a_minus @ Xv[j]))
            if not np.isfinite(pred_var) or pred_var <= degeneracy:
                continue
            slots["expected"][i, b] = expected
            slots["predictive_resid"][i, b] = (float(yb[i]) - expected) / np.sqrt(
                pred_var
            )
            slots["predictive_weight"][i, b] = 1.0 / pred_var
            slots["leverage"][i, b] = h
            slots["surprise_eligible"][i, b] = True
            psi_minus = C @ theta_minus
            est_var_minus = np.sum((C @ a_minus) * C, axis=1) * scale
            for e in range(n_estimand):
                if not (
                    np.isfinite(est_var_minus[e]) and est_var_minus[e] > degeneracy
                ):
                    continue
                se_minus = float(np.sqrt(est_var_minus[e]))
                z_minus = float(psi_minus[e] / se_minus)
                slots["delta_effect"][i, e, b] = float(psi[e] - psi_minus[e])
                slots["deleted_stat"][i, e, b] = z_minus
                slots["delta_stat"][i, e, b] = float(slots["full_stat"][e, b] - z_minus)
                slots["influence_eligible"][i, e, b] = True

    return BlockDiagnostic(
        expected=slots["expected"],
        predictive_resid=slots["predictive_resid"],
        predictive_weight=slots["predictive_weight"],
        leverage=slots["leverage"],
        delta_effect=slots["delta_effect"],
        delta_stat=slots["delta_stat"],
        deleted_stat=slots["deleted_stat"],
        surprise_eligible=slots["surprise_eligible"].astype(bool),
        influence_eligible=slots["influence_eligible"].astype(bool),
        full_effect=slots["full_effect"],
        full_se=slots["full_se"],
        full_stat=slots["full_stat"],
        coverage=slots["coverage"].astype(np.intp),
        tau2=None if extra is None else extra.copy(),
        mode=mode,
        estimand_names=tuple(estimand_names),
    )


def diagnose_ivw_refit_subject(
    beta: NDArray[np.float64],
    var: NDArray[np.float64],
    subject_index: int,
    *,
    tolerance: ExaminationTolerance,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exact DL refit without one subject.

    Returns ``(delta_stat, eligible)`` of shape ``(n_sample,)`` for the
    pooled-effect estimand. Used for ``tau2_refit_exact`` rows.
    """
    y = np.asarray(beta, dtype=np.float64)
    v = np.asarray(var, dtype=np.float64)
    n_sample, n_subject = y.shape
    drop = np.ones(n_subject, dtype=bool)
    drop[int(subject_index)] = False
    full = diagnose_ivw(
        y, v, tau2=dl_tau2(y, v), tolerance=tolerance, mode="tau2_fixed_full"
    )
    y_m = y[:, drop]
    v_m = v[:, drop]
    reduced = diagnose_ivw(
        y_m, v_m, tau2=dl_tau2(y_m, v_m), tolerance=tolerance, mode="exact"
    )
    delta = np.full(n_sample, np.nan)
    eligible = np.zeros(n_sample, dtype=bool)
    for b in range(n_sample):
        z_full = full.full_stat[0, b]
        z_minus = reduced.full_stat[0, b]
        if np.isfinite(z_full) and np.isfinite(z_minus):
            delta[b] = z_full - z_minus
            eligible[b] = True
    return delta, eligible
