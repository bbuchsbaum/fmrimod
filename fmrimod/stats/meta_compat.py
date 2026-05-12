"""Low-level fmrireg-style meta-analysis matrix helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from .meta import (
    _dl_tau2_intercept,
    _pm_tau2_intercept,
    _reml_tau2_intercept,
)


def _as_matrix(x: Any, name: str) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1-D or 2-D matrix")
    return arr


def _validate_yvx(
    Y: Any,
    V: Any,
    X: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    Y_arr = _as_matrix(Y, "Y")
    V_arr = _as_matrix(V, "V")
    X_arr = _as_matrix(X, "X")
    if Y_arr.shape != V_arr.shape:
        raise ValueError("Y and V must have the same dimensions")
    if X_arr.shape[0] != Y_arr.shape[0]:
        raise ValueError("X rows must match Y rows")
    if not np.all(np.isfinite(Y_arr)):
        raise ValueError("Y must be finite")
    if not np.all(np.isfinite(V_arr)) or np.any(V_arr <= 0):
        raise ValueError("V must be finite and > 0")
    return Y_arr, V_arr, X_arr


def _tau2_for(method: str, y: NDArray[np.float64], v: NDArray[np.float64]) -> float:
    if method == "fe":
        return 0.0
    if method == "dl":
        return float(_dl_tau2_intercept(y, v))
    if method == "pm":
        return float(_pm_tau2_intercept(y, v))
    if method == "reml":
        return float(_reml_tau2_intercept(y, v))
    raise ValueError("method must be one of: pm, dl, fe, reml")


def _fit_feature(
    y: NDArray[np.float64],
    v: NDArray[np.float64],
    X: NDArray[np.float64],
    method: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, float, float, bool]:
    tau2 = _tau2_for(method, y, v)
    w = 1.0 / (v + tau2)
    XtW = X.T * w[np.newaxis, :]
    XtWX = XtW @ X
    XtWy = XtW @ y
    ok = True
    try:
        cov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(XtWX)
        ok = False
    beta = cov @ XtWy
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, beta / se, 0.0)

    w_fe = 1.0 / v
    mu_fe = float(np.sum(w_fe * y) / np.sum(w_fe))
    q_fe = float(np.sum(w_fe * (y - mu_fe) ** 2))
    df = max(int(y.size - X.shape[1]), 0)
    i2 = max((q_fe - df) / q_fe, 0.0) if q_fe > 0 and df > 0 else 0.0
    return beta, se, z, tau2, q_fe, i2, ok


def _pack_cov_tri(covs: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    if not covs:
        return np.empty((0, 0), dtype=np.float64)
    idx = np.tril_indices(covs[0].shape[0])
    out = np.zeros((len(idx[0]), len(covs)), dtype=np.float64)
    for col, cov in enumerate(covs):
        out[:, col] = cov[idx]
    return out


def fmri_meta_fit(
    Y: Any,
    V: Any,
    X: Any,
    method: str = "pm",
    robust: str = "none",
    huber_c: float = 1.345,
    robust_iter: int = 2,
    n_threads: int = 0,
) -> Dict[str, Any]:
    """Fit feature-wise meta-regression from effect and variance matrices."""
    del huber_c, robust_iter, n_threads
    if robust != "none":
        raise NotImplementedError("robust meta estimation is not implemented")
    method = "pm" if method == "reml" else method
    if method not in {"pm", "dl", "fe"}:
        raise ValueError("method must be one of: pm, dl, fe, reml")

    Y_arr, V_arr, X_arr = _validate_yvx(Y, V, X)
    p = X_arr.shape[1]
    n_features = Y_arr.shape[1]
    beta = np.zeros((p, n_features), dtype=np.float64)
    se = np.zeros_like(beta)
    z = np.zeros_like(beta)
    tau2 = np.zeros(n_features, dtype=np.float64)
    q_fe = np.zeros(n_features, dtype=np.float64)
    i2_fe = np.zeros(n_features, dtype=np.float64)
    ok = np.ones(n_features, dtype=bool)
    df = np.full(n_features, max(Y_arr.shape[0] - p, 0), dtype=np.float64)

    for feature in range(n_features):
        b, s, zz, tt, q, i2, good = _fit_feature(
            Y_arr[:, feature],
            V_arr[:, feature],
            X_arr,
            method,
        )
        beta[:, feature] = b
        se[:, feature] = s
        z[:, feature] = zz
        tau2[feature] = tt
        q_fe[feature] = q
        i2_fe[feature] = i2
        ok[feature] = good

    return {
        "beta": beta,
        "se": se,
        "z": z,
        "p": 2.0 * sp_stats.norm.sf(np.abs(z)),
        "tau2": tau2,
        "Q_fe": q_fe,
        "I2_fe": i2_fe,
        "df": df,
        "ok": ok,
        "method": method,
        "robust": robust,
    }


def fmri_meta_fit_cov(
    Y: Any,
    V: Any,
    X: Any,
    method: str = "pm",
    robust: str = "none",
    huber_c: float = 1.345,
    robust_iter: int = 2,
    n_threads: int = 0,
) -> Dict[str, Any]:
    """Fit meta-regression and return packed covariance matrices per feature."""
    out = fmri_meta_fit(Y, V, X, method, robust, huber_c, robust_iter, n_threads)
    Y_arr, V_arr, X_arr = _validate_yvx(Y, V, X)
    covs = []
    for feature in range(Y_arr.shape[1]):
        tau2 = float(out["tau2"][feature])
        w = 1.0 / (V_arr[:, feature] + tau2)
        XtWX = (X_arr.T * w[np.newaxis, :]) @ X_arr
        try:
            covs.append(np.linalg.inv(XtWX))
        except np.linalg.LinAlgError:
            covs.append(np.linalg.pinv(XtWX))
    out["cov_tri"] = _pack_cov_tri(covs)
    return out


def fmri_meta_fit_contrasts(
    Y: Any,
    V: Any,
    X: Any,
    Cmat: Any,
    method: str = "pm",
    robust: str = "none",
    huber_c: float = 1.345,
    robust_iter: int = 2,
    n_threads: int = 0,
) -> Dict[str, Any]:
    """Fit meta-regression and return exact linear-contrast summaries."""
    out = fmri_meta_fit_cov(Y, V, X, method, robust, huber_c, robust_iter, n_threads)
    X_arr = _as_matrix(X, "X")
    C = _as_matrix(Cmat, "Cmat")
    if C.shape[0] != X_arr.shape[1]:
        raise ValueError("Cmat rows must equal X columns")

    cov_tri = np.asarray(out["cov_tri"], dtype=np.float64)
    tri = np.tril_indices(X_arr.shape[1])
    n_con = C.shape[1]
    n_features = out["beta"].shape[1]
    c_beta = C.T @ out["beta"]
    c_se = np.zeros((n_con, n_features), dtype=np.float64)
    for feature in range(n_features):
        cov = np.zeros((X_arr.shape[1], X_arr.shape[1]), dtype=np.float64)
        cov[tri] = cov_tri[:, feature]
        cov[(tri[1], tri[0])] = cov_tri[:, feature]
        c_cov = C.T @ cov @ C
        c_se[:, feature] = np.sqrt(np.clip(np.diag(c_cov), 0.0, np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        c_z = np.where(c_se > 0, c_beta / c_se, 0.0)
    out.update(
        {
            "c_beta": c_beta,
            "c_se": c_se,
            "c_z": c_z,
            "c_p": 2.0 * sp_stats.norm.sf(np.abs(c_z)),
        }
    )
    return out


def fmri_meta_fit_extended(
    Y: Any,
    V: Any,
    X: Any,
    method: str = "pm",
    robust: str = "none",
    huber_c: float = 1.345,
    robust_iter: int = 2,
    voxelwise: Optional[Any] = None,
    center_voxelwise: bool = True,
    voxel_name: str = "voxel_cov",
    n_threads: int = 0,
) -> Dict[str, Any]:
    """Fit meta-regression with an optional per-feature voxelwise covariate."""
    del voxel_name
    if voxelwise is None:
        return fmri_meta_fit(Y, V, X, method, robust, huber_c, robust_iter, n_threads)

    Y_arr, V_arr, X_arr = _validate_yvx(Y, V, X)
    C = _as_matrix(voxelwise, "voxelwise")
    if C.shape != Y_arr.shape:
        raise ValueError("voxelwise covariate must match Y dimensions")
    if center_voxelwise:
        C = C - np.nanmean(C, axis=0, keepdims=True)

    p = X_arr.shape[1] + 1
    n_features = Y_arr.shape[1]
    beta = np.zeros((p, n_features), dtype=np.float64)
    se = np.zeros_like(beta)
    z = np.zeros_like(beta)
    tau2 = np.zeros(n_features, dtype=np.float64)
    q_fe = np.zeros(n_features, dtype=np.float64)
    i2_fe = np.zeros(n_features, dtype=np.float64)
    ok = np.ones(n_features, dtype=bool)
    for feature in range(n_features):
        X_aug = np.column_stack([X_arr, C[:, feature]])
        b, s, zz, tt, q, i2, good = _fit_feature(
            Y_arr[:, feature],
            V_arr[:, feature],
            X_aug,
            "pm" if method == "reml" else method,
        )
        beta[:, feature] = b
        se[:, feature] = s
        z[:, feature] = zz
        tau2[feature] = tt
        q_fe[feature] = q
        i2_fe[feature] = i2
        ok[feature] = good
    return {
        "beta": beta,
        "se": se,
        "z": z,
        "p": 2.0 * sp_stats.norm.sf(np.abs(z)),
        "tau2": tau2,
        "Q_fe": q_fe,
        "I2_fe": i2_fe,
        "df": np.full(n_features, max(Y_arr.shape[0] - p, 0), dtype=np.float64),
        "ok": ok,
        "method": "pm" if method == "reml" else method,
        "robust": robust,
    }


def meta_effective_n(v: Any, tau2: Any) -> float:
    """Compute inverse-variance effective sample size."""
    vv = np.asarray(v, dtype=np.float64)
    tt = np.asarray(tau2, dtype=np.float64)
    w = 1.0 / (vv + tt)
    return float(np.sum(w) ** 2 / np.sum(w ** 2))
