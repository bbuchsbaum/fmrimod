"""Group-level meta-analysis helpers.

Initial parity slice for ``fmrireg::fmri_meta``:
- accepts ``GroupData`` inputs
- supports CSV-backed effect-size data (beta+se or beta+var)
- supports fixed-effects and intercept-only random-effects fits
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import warnings

import numpy as np
from numpy.typing import NDArray
from patsy import dmatrix
from scipy import stats as sp_stats
from scipy import optimize as sp_opt

from ..dataset.group_data import GroupData


MetaMethod = Literal["fe", "pm", "dl", "reml"]
MetaRobust = Literal["none", "huber", "t"]
MetaWeights = Literal["ivw", "equal", "custom"]


@dataclass
class FmriMetaResult:
    """Result object for group-level meta analysis."""

    coefficients: NDArray[np.float64]
    se: NDArray[np.float64]
    z: NDArray[np.float64]
    p: NDArray[np.float64]
    tau2: NDArray[np.float64]
    predictor_names: list[str]
    feature_names: list[str]
    method: str
    formula: str
    data: GroupData


def _as_formula(formula: str | object) -> str:
    if not isinstance(formula, str):
        raise TypeError("'formula' must be a string (for example '~ 1' or '~ 1 + age')")
    return formula


def _extract_feature_frames(data: GroupData) -> tuple[list[str], list[tuple[NDArray[np.float64], NDArray[np.float64]]]]:
    if data.format != "csv":
        raise NotImplementedError(
            "fmri_meta currently supports GroupData format='csv' only"
        )

    payload = data.data
    df = payload.get("data")
    if df is None:
        raise ValueError("CSV GroupData payload is missing 'data'")

    effect_cols = payload.get("effect_cols") or {}
    subject_col = payload.get("subject_col")
    roi_col = payload.get("roi_col")
    contrast_col = payload.get("contrast_col")
    subjects = list(data.subjects)

    beta_col = effect_cols.get("beta")
    se_col = effect_cols.get("se")
    var_col = effect_cols.get("var")
    if beta_col is None:
        raise NotImplementedError("fmri_meta currently requires a beta effect column")
    if se_col is None and var_col is None:
        raise ValueError("CSV GroupData for fmri_meta requires se or var effect column")

    group_cols = [c for c in (roi_col, contrast_col) if c is not None]
    if group_cols:
        grouped = list(df.groupby(group_cols, sort=False))
    else:
        grouped = [("__all__", df)]

    feature_names: list[str] = []
    feature_data: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for key, block in grouped:
        key_block = block.copy()
        if key_block[subject_col].duplicated().any():
            raise ValueError(
                "Each feature must have at most one row per subject; found duplicates"
            )
        by_subj = key_block.set_index(subject_col)
        missing = [s for s in subjects if s not in by_subj.index]
        if missing:
            raise ValueError(
                "Missing feature rows for subjects: " + ", ".join(str(s) for s in missing)
            )
        aligned = by_subj.loc[subjects]
        y = np.asarray(aligned[beta_col], dtype=np.float64)
        if se_col is not None:
            se = np.asarray(aligned[se_col], dtype=np.float64)
            v = se ** 2
        else:
            v = np.asarray(aligned[var_col], dtype=np.float64)  # type: ignore[index]

        if not np.all(np.isfinite(y)):
            raise ValueError("Effect-size values must be finite")
        if not np.all(np.isfinite(v)) or np.any(v <= 0):
            raise ValueError("Effect variances must be finite and > 0")

        if group_cols:
            if isinstance(key, tuple):
                pieces = [f"{col}={val}" for col, val in zip(group_cols, key)]
                fname = "|".join(pieces)
            else:
                fname = f"{group_cols[0]}={key}"
        else:
            fname = "__all__"

        feature_names.append(fname)
        feature_data.append((y, v))

    return feature_names, feature_data


def _build_design_matrix(data: GroupData, formula: str) -> tuple[NDArray[np.float64], list[str]]:
    cov = data.covariates
    if cov is None:
        clean = formula.replace(" ", "")
        if clean not in ("~1", "1"):
            raise ValueError(
                "Non-intercept formulas require covariates in GroupData"
            )
        X = np.ones((data.n_subjects, 1), dtype=np.float64)
        return X, ["Intercept"]

    frame = cov.copy().reset_index(drop=True)
    X_df = dmatrix(formula, frame, return_type="dataframe")
    return np.asarray(X_df, dtype=np.float64), list(X_df.columns)


def _dl_tau2_intercept(y: NDArray[np.float64], v: NDArray[np.float64]) -> float:
    w = 1.0 / v
    wsum = float(np.sum(w))
    if wsum <= 0:
        return 0.0
    mu = float(np.sum(w * y) / wsum)
    q = float(np.sum(w * (y - mu) ** 2))
    c = wsum - float(np.sum(w ** 2) / wsum)
    if c <= 0:
        return 0.0
    return max((q - (len(y) - 1)) / c, 0.0)


def _q_stat_intercept(y: NDArray[np.float64], v: NDArray[np.float64], tau2: float) -> float:
    """Cochran's Q for intercept-only model at fixed tau2."""
    w = 1.0 / (v + tau2)
    wsum = float(np.sum(w))
    if wsum <= 0:
        return 0.0
    mu = float(np.sum(w * y) / wsum)
    return float(np.sum(w * (y - mu) ** 2))


def _pm_tau2_intercept(y: NDArray[np.float64], v: NDArray[np.float64]) -> float:
    """Paule-Mandel tau2 estimator for intercept-only random-effects meta."""
    df = len(y) - 1
    if df <= 0:
        return 0.0
    q0 = _q_stat_intercept(y, v, tau2=0.0)
    if q0 <= df:
        return 0.0

    def f(tau: float) -> float:
        return _q_stat_intercept(y, v, tau2=float(tau)) - df

    hi = max(np.var(y), np.mean(v), 1e-6)
    while f(hi) > 0 and hi < 1e8:
        hi *= 2.0
    if f(hi) > 0:
        # Extremely heterogeneous edge case; return practical upper bound.
        return float(hi)

    return float(sp_opt.brentq(f, 0.0, hi))


def _reml_criterion_intercept(tau2: float, y: NDArray[np.float64], v: NDArray[np.float64]) -> float:
    """-2 restricted log-likelihood (constant dropped), intercept-only model."""
    if tau2 < 0:
        return np.inf
    vv = v + tau2
    if np.any(vv <= 0):
        return np.inf
    w = 1.0 / vv
    wsum = float(np.sum(w))
    if wsum <= 0:
        return np.inf
    mu = float(np.sum(w * y) / wsum)
    rss = float(np.sum(w * (y - mu) ** 2))
    return float(np.sum(np.log(vv)) + np.log(wsum) + rss)


def _reml_tau2_intercept(y: NDArray[np.float64], v: NDArray[np.float64]) -> float:
    """REML tau2 estimator for intercept-only random-effects meta."""
    hi = max(_dl_tau2_intercept(y, v) * 2.0, np.var(y), np.mean(v), 1e-6)
    for _ in range(8):
        res = sp_opt.minimize_scalar(
            _reml_criterion_intercept,
            bounds=(0.0, hi),
            args=(y, v),
            method="bounded",
        )
        if not res.success:
            warnings.warn(
                "REML tau2 optimization did not converge; falling back to DL",
                UserWarning,
                stacklevel=2,
            )
            return _dl_tau2_intercept(y, v)
        # If the optimum is at the boundary, expand search interval.
        if res.x >= 0.98 * hi:
            hi *= 2.0
            continue
        tau = float(max(res.x, 0.0))
        return 0.0 if tau < 1e-10 else tau

    tau = float(max(res.x, 0.0))
    return 0.0 if tau < 1e-10 else tau


def _solve_meta_wls(
    y: NDArray[np.float64],
    X: NDArray[np.float64],
    v: NDArray[np.float64],
    method: MetaMethod,
    weight_mode: MetaWeights,
    weight_custom: Optional[NDArray[np.float64]] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    tau2 = 0.0
    if method in ("dl", "pm", "reml"):
        if X.shape[1] != 1 or not np.allclose(X[:, 0], 1.0):
            raise NotImplementedError(
                "Random-effects fmri_meta currently supports intercept-only formula (~ 1)"
            )
        if method == "dl":
            tau2 = _dl_tau2_intercept(y, v)
        elif method == "pm":
            tau2 = _pm_tau2_intercept(y, v)
        else:
            tau2 = _reml_tau2_intercept(y, v)

    base_w = 1.0 / (v + tau2)
    if weight_mode == "ivw":
        w = base_w
    elif weight_mode == "equal":
        w = np.ones_like(base_w)
    else:
        if weight_custom is None:
            raise ValueError("Custom weights requested but no weights were provided")
        w = weight_custom

    if np.any(w <= 0) or not np.all(np.isfinite(w)):
        raise ValueError("Weights must be finite and > 0")

    XtW = X.T * w[np.newaxis, :]
    XtWX = XtW @ X
    XtWy = XtW @ y

    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Design is rank-deficient; using pseudo-inverse for meta-regression covariance",
            UserWarning,
            stacklevel=2,
        )
        XtWX_inv = np.linalg.pinv(XtWX)

    beta = XtWX_inv @ XtWy
    se = np.sqrt(np.clip(np.diag(XtWX_inv), 0.0, np.inf))
    return beta, se, tau2


def fmri_meta(
    data: GroupData,
    formula: str = "~ 1",
    method: MetaMethod = "pm",
    robust: MetaRobust = "none",
    weights: MetaWeights = "ivw",
    weights_custom: Optional[NDArray[np.float64]] = None,
    combine: Optional[str] = None,
) -> FmriMetaResult:
    """Fit group-level meta-regression for parity-oriented workflows.

    Notes
    -----
    This initial implementation targets CSV-backed effect-size data.
    """
    if not isinstance(data, GroupData):
        raise TypeError("'data' must be a GroupData instance")
    if robust != "none":
        raise NotImplementedError("robust meta estimation is not implemented yet")
    if combine is not None:
        raise NotImplementedError("t-only combine modes are not implemented yet")

    formula = _as_formula(formula)
    X, predictor_names = _build_design_matrix(data, formula)
    feature_names, feature_data = _extract_feature_frames(data)
    n_features = len(feature_data)
    p = X.shape[1]

    custom_vec = None
    custom_mat = None
    if weights == "custom":
        if weights_custom is None:
            raise ValueError("weights='custom' requires weights_custom")
        arr = np.asarray(weights_custom, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != data.n_subjects:
                raise ValueError("1-D custom weights must have length n_subjects")
            custom_vec = arr
        elif arr.ndim == 2:
            if arr.shape != (data.n_subjects, n_features):
                raise ValueError(
                    "2-D custom weights must have shape (n_subjects, n_features)"
                )
            custom_mat = arr
        else:
            raise ValueError("weights_custom must be 1-D or 2-D")
    elif weights_custom is not None:
        raise ValueError("weights_custom can only be provided when weights='custom'")

    coefs = np.zeros((n_features, p), dtype=np.float64)
    ses = np.zeros((n_features, p), dtype=np.float64)
    tau2 = np.zeros(n_features, dtype=np.float64)
    for idx, (y, v) in enumerate(feature_data):
        if custom_mat is not None:
            custom_w = custom_mat[:, idx]
        else:
            custom_w = custom_vec
        beta_i, se_i, tau2_i = _solve_meta_wls(
            y=y,
            X=X,
            v=v,
            method=method,
            weight_mode=weights,
            weight_custom=custom_w,
        )
        coefs[idx, :] = beta_i
        ses[idx, :] = se_i
        tau2[idx] = tau2_i

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(ses > 0, coefs / ses, 0.0)
    pvals = 2.0 * sp_stats.norm.sf(np.abs(z))

    return FmriMetaResult(
        coefficients=coefs,
        se=ses,
        z=z,
        p=pvals,
        tau2=tau2,
        predictor_names=predictor_names,
        feature_names=feature_names,
        method=method,
        formula=formula,
        data=data,
    )
