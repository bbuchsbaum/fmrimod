"""Compatibility helpers for fmrireg-style GLM workflows.

These functions provide small, explicit bridges from selected R ``fmrireg``
exports to the Python ``FmriLm`` and matrix-solver APIs. They are intentionally
thin: the stable Python surface remains ``fmri_lm`` plus result methods, while
these helpers make migration code and parity tests less ambiguous.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize

from ..model.config import FmriLmConfig
from .contrasts import ContrastResult, contrast_f_vectorized, contrast_t_batch
from .fmri_lm import FmriLm, fmri_lm
from .matrix import _design_matrix_from_model, _MatrixModel
from .solver import fast_lm_matrix, fast_preproject

ArrayLike = Any


@dataclass
class SoftProjection:
    """Ridge-regularized nuisance projection operator."""

    nuisance: NDArray[np.float64]
    lam: float
    method: str
    effective_df: float
    U: NDArray[np.float64] = field(repr=False)
    singular_values: NDArray[np.float64] = field(repr=False)

    @property
    def n_nuisance(self) -> int:
        return int(self.nuisance.shape[1])

    @property
    def n_timepoints(self) -> int:
        return int(self.nuisance.shape[0])

    def apply(self, x: ArrayLike) -> NDArray[np.float64]:
        """Apply the projection to a time-by-feature matrix."""
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, np.newaxis]
        if x_arr.ndim != 2:
            raise ValueError("x must be a 1-D or 2-D matrix")
        if x_arr.shape[0] != self.n_timepoints:
            raise ValueError("x rows must match projection timepoints")

        d2 = self.singular_values ** 2
        shrink = d2 / (d2 + self.lam)
        ux = self.U.T @ x_arr
        return x_arr - self.U @ (shrink[:, np.newaxis] * ux)


def _select_lambda_gcv(
    U: NDArray[np.float64],
    d2: NDArray[np.float64],
    Y: NDArray[np.float64],
) -> float:
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    UY = U.T @ Y
    n = Y.shape[0]

    positive = d2[d2 > 0]
    if positive.size == 0:
        return 0.0

    def score(log_lam: float) -> float:
        lam = float(np.exp(log_lam))
        shrink = d2 / (d2 + lam)
        df = float(np.sum(shrink))
        y_hat = U @ (shrink[:, np.newaxis] * UY)
        rss = float(np.sum((Y - y_hat) ** 2))
        denom = (1.0 - df / n) ** 2
        if denom < 1e-10:
            return float("inf")
        return rss / denom

    lo = float(np.log(np.min(positive) / 100.0))
    hi = float(np.log(np.max(positive) * 100.0))
    opt = optimize.minimize_scalar(score, bounds=(lo, hi), method="bounded")
    return float(np.exp(opt.x))


def soft_projection(
    N: ArrayLike,
    lam: Union[float, str] = "auto",
    Y: Optional[ArrayLike] = None,
    **kwargs: object,
) -> SoftProjection:
    """Create a ridge-regularized nuisance projection.

    Accepts the R-style ``lambda=`` alias through ``kwargs`` while exposing the
    Python-safe ``lam`` argument.
    """
    if "lambda" in kwargs:
        if lam != "auto":
            raise TypeError("Specify only one of 'lam' or 'lambda'")
        lam = kwargs.pop("lambda")
    if kwargs:
        extras = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected soft_projection argument(s): {extras}")

    N_arr = np.asarray(N, dtype=np.float64)
    if N_arr.ndim == 1:
        N_arr = N_arr[:, np.newaxis]
    if N_arr.ndim != 2:
        raise ValueError("N must be a 1-D or 2-D nuisance matrix")
    if N_arr.shape[0] == 0:
        raise ValueError("N must have at least one row")

    U, s, _ = np.linalg.svd(N_arr, full_matrices=False)
    d2 = s ** 2

    if isinstance(lam, str):
        if lam not in {"auto", "gcv"}:
            raise ValueError("lambda must be a non-negative number, 'auto', or 'gcv'")
        if lam == "auto" or Y is None:
            if lam == "gcv" and Y is None:
                warnings.warn("GCV requires Y; falling back to auto", RuntimeWarning, stacklevel=2)
            lam_val = float(np.median(d2))
            method = "singular_value_heuristic"
        else:
            lam_val = _select_lambda_gcv(U, d2, np.asarray(Y, dtype=np.float64))
            method = "gcv"
    else:
        lam_val = float(lam)
        if lam_val < 0:
            raise ValueError("lambda must be non-negative")
        method = "user_specified"

    effective_df = float(np.sum(d2 / (d2 + lam_val))) if d2.size else 0.0
    return SoftProjection(
        nuisance=N_arr,
        lam=lam_val,
        method=method,
        effective_df=effective_df,
        U=U,
        singular_values=s,
    )


def apply_soft_projection(
    proj: SoftProjection,
    Y: ArrayLike,
    X: ArrayLike,
) -> Dict[str, NDArray[np.float64]]:
    """Apply a :class:`SoftProjection` to both response and design matrices."""
    if not isinstance(proj, SoftProjection):
        raise TypeError("proj must be a SoftProjection")
    return {"Y": proj.apply(Y), "X": proj.apply(X)}


def _coerce_sigma2(
    sigma: Optional[ArrayLike],
    sigma2: Optional[ArrayLike],
    n_voxels: int,
) -> NDArray[np.float64]:
    if sigma2 is not None:
        out = np.asarray(sigma2, dtype=np.float64)
    elif sigma is not None:
        out = np.asarray(sigma, dtype=np.float64) ** 2
    else:
        raise ValueError("Provide either sigma or sigma2")
    out = np.ravel(out)
    if out.size == 1:
        out = np.repeat(out, n_voxels)
    if out.size != n_voxels:
        raise ValueError("sigma/sigma2 length must match number of voxels")
    return out


def _coef_columns(columns: Optional[Sequence[str]], p: int) -> Optional[list[str]]:
    if columns is None:
        return None
    out = [str(c) for c in columns]
    if len(out) != p:
        raise ValueError("columns length must match number of coefficients")
    return out


def _contrast_to_array(spec: Any, p: int, columns: Optional[list[str]]) -> NDArray[np.float64]:
    if isinstance(spec, pd.Series):
        if spec.index is not None and columns is not None:
            arr = np.zeros(p, dtype=np.float64)
            for name, value in spec.items():
                if str(name) not in columns:
                    raise ValueError(f"Contrast name {name!r} not found in columns")
                arr[columns.index(str(name))] = float(value)
            return arr
        return np.asarray(spec.to_numpy(), dtype=np.float64)
    if isinstance(spec, Mapping):
        if columns is None:
            raise ValueError("Named contrast mappings require columns")
        arr = np.zeros(p, dtype=np.float64)
        for name, value in spec.items():
            if str(name) not in columns:
                raise ValueError(f"Contrast name {name!r} not found in columns")
            arr[columns.index(str(name))] = float(value)
        return arr
    if isinstance(spec, pd.DataFrame):
        if columns is not None:
            missing = [c for c in spec.columns if str(c) not in columns]
            if missing:
                raise ValueError("F-contrast columns not found: " + ", ".join(map(str, missing)))
            arr = np.zeros((spec.shape[0], p), dtype=np.float64)
            for col in spec.columns:
                arr[:, columns.index(str(col))] = spec[col].to_numpy(dtype=np.float64)
            return arr
        return spec.to_numpy(dtype=np.float64)
    return np.asarray(spec, dtype=np.float64)


def _split_contrasts(
    p: int,
    contrasts: Optional[Mapping[str, object]],
    t_contrasts: Optional[Mapping[str, object]],
    f_contrasts: Optional[Mapping[str, object]],
    columns: Optional[list[str]],
) -> tuple[Dict[str, NDArray[np.float64]], Dict[str, NDArray[np.float64]]]:
    t_out: Dict[str, NDArray[np.float64]] = {}
    f_out: Dict[str, NDArray[np.float64]] = {}

    def add(name: str, spec: Any, force: Optional[str] = None) -> None:
        arr = _contrast_to_array(spec, p, columns)
        if force == "t" or (force is None and arr.ndim == 1):
            arr = np.ravel(arr)
            if arr.size != p:
                raise ValueError(f"t-contrast {name!r} must have length {p}")
            t_out[name] = arr
            return
        arr = np.atleast_2d(arr)
        if arr.shape[1] != p:
            raise ValueError(f"F-contrast {name!r} must have {p} columns")
        f_out[name] = arr

    for source, force in ((contrasts, None), (t_contrasts, "t"), (f_contrasts, "F")):
        if not source:
            continue
        for name, spec in source.items():
            add(str(name), spec, force)
    return t_out, f_out


def _contrast_result_frame(result: ContrastResult) -> pd.DataFrame:
    rows = []
    estimate = np.asarray(result.estimate)
    stat = np.ravel(result.stat)
    p_value = np.ravel(result.p_value)
    se = None if result.se is None else np.ravel(result.se)
    if result.stat_type == "F":
        estimate_out = np.nanmean(np.atleast_2d(estimate), axis=0)
    else:
        estimate_out = np.ravel(estimate)

    for voxel in range(stat.size):
        rows.append(
            {
                "contrast": result.name,
                "voxel": voxel,
                "estimate": estimate_out[voxel],
                "std_error": None if se is None else se[voxel],
                "stat": stat[voxel],
                "p_value": p_value[voxel],
                "df": result.df,
                "stat_type": result.stat_type,
            }
        )
    return pd.DataFrame(rows)


def compute_lm_contrasts(
    B: ArrayLike,
    XtXinv: ArrayLike,
    df: float,
    sigma: Optional[ArrayLike] = None,
    sigma2: Optional[ArrayLike] = None,
    contrasts: Optional[Mapping[str, object]] = None,
    t_contrasts: Optional[Mapping[str, object]] = None,
    f_contrasts: Optional[Mapping[str, object]] = None,
    columns: Optional[Sequence[str]] = None,
    output: str = "stacked",
    robust_weights: Optional[ArrayLike] = None,
    ar_order: int = 0,
    drop_failed: bool = True,
) -> Union[pd.DataFrame, Dict[str, ContrastResult]]:
    """Compute t/F contrast statistics from fitted GLM matrices."""
    del robust_weights, ar_order, drop_failed
    B_arr = np.asarray(B, dtype=np.float64)
    XtXinv_arr = np.asarray(XtXinv, dtype=np.float64)
    if B_arr.ndim != 2:
        raise ValueError("B must be a 2-D coefficient matrix")
    p, n_voxels = B_arr.shape
    if XtXinv_arr.shape != (p, p):
        raise ValueError("XtXinv must be p x p")
    sigma_vec = np.sqrt(_coerce_sigma2(sigma, sigma2, n_voxels))
    col_names = _coef_columns(columns, p)
    t_specs, f_specs = _split_contrasts(p, contrasts, t_contrasts, f_contrasts, col_names)

    out: Dict[str, ContrastResult] = {}
    if t_specs:
        names = list(t_specs)
        mat = np.vstack([t_specs[name] for name in names])
        for result in contrast_t_batch(mat, B_arr, XtXinv_arr, sigma_vec, float(df), names=names):
            out[result.name] = result
    for name, mat in f_specs.items():
        out[name] = contrast_f_vectorized(mat, B_arr, XtXinv_arr, sigma_vec, float(df), name=name)

    if output == "list":
        return out
    if output != "stacked":
        raise ValueError("output must be 'stacked' or 'list'")
    if not out:
        return pd.DataFrame(
            columns=["contrast", "voxel", "estimate", "std_error", "stat", "p_value", "df", "stat_type"]
        )
    return pd.concat([_contrast_result_frame(res) for res in out.values()], ignore_index=True)


def compute_lm_contrasts_from_suffstats(
    XtX: ArrayLike,
    XtS: ArrayLike,
    StS: ArrayLike,
    df: float,
    sigma: Optional[ArrayLike] = None,
    sigma2: Optional[ArrayLike] = None,
    **kwargs: object,
) -> Union[pd.DataFrame, Dict[str, ContrastResult]]:
    """Compute contrast statistics from design/data sufficient statistics."""
    XtX_arr = np.asarray(XtX, dtype=np.float64)
    XtS_arr = np.asarray(XtS, dtype=np.float64)
    StS_arr = np.ravel(np.asarray(StS, dtype=np.float64))
    p = XtX_arr.shape[0]
    if XtX_arr.shape != (p, p):
        raise ValueError("XtX must be square")
    if XtS_arr.ndim != 2 or XtS_arr.shape[0] != p:
        raise ValueError("XtS must be p x V")
    if StS_arr.size != XtS_arr.shape[1]:
        raise ValueError("StS length must match number of voxels")
    XtXinv = np.linalg.pinv(XtX_arr)
    B = XtXinv @ XtS_arr
    if sigma is None and sigma2 is None:
        sse = np.maximum(StS_arr - np.sum(B * XtS_arr, axis=0), 0.0)
        sigma2 = np.maximum(sse / float(df), np.finfo(np.float64).eps)
    return compute_lm_contrasts(
        B,
        XtXinv,
        df,
        sigma=sigma,
        sigma2=sigma2,
        **kwargs,
    )


def fit_contrasts(
    fit: object,
    contrasts: Mapping[str, object],
    output: str = "list",
    **kwargs: object,
) -> Union[Dict[str, ContrastResult], pd.DataFrame]:
    """Fit contrasts on an ``FmriLm`` result or raw matrix payload."""
    if isinstance(fit, FmriLm):
        results = fit.compute_contrasts(
            {str(name): _contrast_to_array(spec, fit.n_coefficients, None) for name, spec in contrasts.items()}
        )
        if output == "list":
            return results
        if output == "stacked":
            return pd.concat([_contrast_result_frame(res) for res in results.values()], ignore_index=True)
        raise ValueError("output must be 'list' or 'stacked'")

    if isinstance(fit, Mapping):
        return compute_lm_contrasts(contrasts=contrasts, output=output, **fit, **kwargs)
    raise TypeError("fit must be an FmriLm result or a matrix payload mapping")


def fit_glm_on_transformed_series(
    model: object,
    Y: ArrayLike,
    cfg: Optional[FmriLmConfig] = None,
    dataset: Optional[object] = None,
    strategy: str = "external",
    engine: str = "runwise",
) -> FmriLm:
    """Fit a GLM using a model design and externally supplied response matrix."""
    del dataset
    Y_arr = np.asarray(Y, dtype=np.float64)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr[:, np.newaxis]
    X = _design_matrix_from_model(model)
    if X.shape[0] != Y_arr.shape[0]:
        raise ValueError("Row mismatch between design matrix and response matrix")
    fit = fmri_lm(_MatrixModel(X, Y_arr, source=model), cfg or FmriLmConfig(), engine=engine)
    fit.strategy = strategy
    fit.engine = engine
    return fit


def fit_glm_with_config(
    model: object,
    Y: ArrayLike,
    cfg: Optional[FmriLmConfig] = None,
    dataset: Optional[object] = None,
    strategy: str = "external",
    engine: str = "runwise",
) -> FmriLm:
    """Fit a transformed-series GLM while honoring a full ``FmriLmConfig``."""
    return fit_glm_on_transformed_series(
        model,
        Y,
        cfg=cfg or FmriLmConfig(),
        dataset=dataset,
        strategy=strategy,
        engine=engine,
    )


def fmri_ols_fit(
    Y: ArrayLike,
    X: ArrayLike,
    voxelwise: Optional[ArrayLike] = None,
    center_voxelwise: bool = True,
    voxel_name: str = "voxel_cov",
) -> Dict[str, object]:
    """Fit matrix OLS and return beta, SE, and t matrices."""
    Y_arr = np.asarray(Y, dtype=np.float64)
    X_arr = np.asarray(X, dtype=np.float64)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr[:, np.newaxis]
    if X_arr.ndim != 2 or Y_arr.ndim != 2 or X_arr.shape[0] != Y_arr.shape[0]:
        raise ValueError("Y and X must be 2-D matrices with matching rows")

    if voxelwise is None:
        proj = fast_preproject(X_arr)
        lm = fast_lm_matrix(X_arr, Y_arr, proj)
        sigma = np.sqrt(lm.sigma2)
        se = sigma[np.newaxis, :] * np.sqrt(np.maximum(np.diag(proj.XtXinv), 0.0))[:, np.newaxis]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(se > 1e-15, lm.betas / se, 0.0)
        return {"beta": lm.betas, "se": se, "t": t, "df": lm.dfres, "XtXinv": proj.XtXinv}

    C = np.asarray(voxelwise, dtype=np.float64)
    if C.shape != Y_arr.shape:
        raise ValueError("voxelwise covariate must match Y dimensions")
    if center_voxelwise:
        C = C - np.nanmean(C, axis=0, keepdims=True)
    beta = []
    se = []
    t = []
    dfs = []
    for voxel in range(Y_arr.shape[1]):
        Xv = np.column_stack([X_arr, C[:, voxel]])
        proj = fast_preproject(Xv)
        lm = fast_lm_matrix(Xv, Y_arr[:, [voxel]], proj)
        sigma = np.sqrt(lm.sigma2)
        sev = sigma[np.newaxis, :] * np.sqrt(np.maximum(np.diag(proj.XtXinv), 0.0))[:, np.newaxis]
        beta.append(lm.betas[:, 0])
        se.append(sev[:, 0])
        with np.errstate(divide="ignore", invalid="ignore"):
            t.append(np.where(sev[:, 0] > 1e-15, lm.betas[:, 0] / sev[:, 0], 0.0))
        dfs.append(lm.dfres)
    names = [f"coef_{idx}" for idx in range(X_arr.shape[1])] + [voxel_name]
    return {
        "beta": pd.DataFrame(np.column_stack(beta), index=names),
        "se": pd.DataFrame(np.column_stack(se), index=names),
        "t": pd.DataFrame(np.column_stack(t), index=names),
        "df": np.asarray(dfs, dtype=np.float64),
    }


def fmri_rlm(model: object, config: Optional[FmriLmConfig] = None, **kwargs: object) -> FmriLm:
    """Fit a robust GLM using the normal Python ``fmri_lm`` model contract."""
    base = config or FmriLmConfig()
    cfg = replace(base, robust=replace(base.robust, enabled=True))
    return fmri_lm(model, cfg, **kwargs)


@dataclass
class LowRankControl:
    parcels: Optional[object] = None
    landmarks: Optional[int] = None
    k_neighbors: int = 16
    time_sketch: Optional[Mapping[str, object]] = None
    ncomp: Optional[int] = None
    noise_pcs: int = 0

    def to_engine_kwargs(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        if self.time_sketch:
            out.update(dict(self.time_sketch))
        if self.ncomp is not None:
            out["ncomp"] = self.ncomp
        return out


def lowrank_control(
    parcels: Optional[object] = None,
    landmarks: Optional[int] = None,
    k_neighbors: int = 16,
    time_sketch: Optional[Mapping[str, object]] = None,
    ncomp: Optional[int] = None,
    noise_pcs: int = 0,
) -> LowRankControl:
    """Create low-rank/sketch fitting options."""
    if landmarks is not None and landmarks <= 0:
        raise ValueError("landmarks must be positive")
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")
    if time_sketch is None:
        time_sketch = {"method": "gaussian", "m": None, "iters": 0}
    return LowRankControl(parcels, landmarks, int(k_neighbors), time_sketch, ncomp, int(noise_pcs))


def paired_diff_block(
    blkA: Mapping[str, object],
    blkB: Mapping[str, object],
    rho: ArrayLike = 0,
) -> Dict[str, object]:
    """Compute paired within-subject differences for group-data blocks."""
    YA = np.asarray(blkA["Y"], dtype=np.float64)
    YB = np.asarray(blkB["Y"], dtype=np.float64)
    if YA.shape != YB.shape:
        raise ValueError("Blocks must have identical Y dimensions")
    meta_a = dict(blkA.get("meta", {}))
    meta_b = dict(blkB.get("meta", {}))
    if meta_a.get("subjects") is not None and meta_b.get("subjects") is not None:
        if list(meta_a["subjects"]) != list(meta_b["subjects"]):
            raise ValueError("paired_diff_block: subjects must match between blocks")
    Y = YA - YB
    V = None
    if blkA.get("V") is not None and blkB.get("V") is not None:
        VA = np.asarray(blkA["V"], dtype=np.float64)
        VB = np.asarray(blkB["V"], dtype=np.float64)
        rho_arr = np.asarray(rho, dtype=np.float64)
        if rho_arr.size == 1:
            rho_mat = np.full(Y.shape, float(rho_arr))
        elif rho_arr.shape == (Y.shape[0],):
            rho_mat = np.repeat(rho_arr[:, np.newaxis], Y.shape[1], axis=1)
        elif rho_arr.shape == (Y.shape[1],):
            rho_mat = np.repeat(rho_arr[np.newaxis, :], Y.shape[0], axis=0)
        else:
            rho_mat = np.asarray(rho_arr, dtype=np.float64)
        V = VA + VB - 2.0 * rho_mat * np.sqrt(np.maximum(VA, 0.0) * np.maximum(VB, 0.0))
        V[~np.isfinite(V)] = np.nan
    if "contrast" in meta_a:
        meta_a["contrast"] = f"{meta_a['contrast']}_minus_{meta_b.get('contrast', 'B')}"
    return {
        "Y": Y,
        "V": V,
        "T": None,
        "DF": None,
        "index": blkA.get("index"),
        "meta": meta_a,
        "covars": blkA.get("covars"),
        "feature": blkA.get("feature"),
    }


def flip_sign(fit: object, coef: Optional[Sequence[str]] = None) -> object:
    """Flip coefficient-like signed outputs in a mapping or fitted object."""
    names = set(coef or [])
    elements = ("beta", "t", "z", "z_contrast", "t_contrast")
    if isinstance(fit, Mapping):
        out = dict(fit)
        for key in elements:
            if key not in out or out[key] is None:
                continue
            value = out[key]
            if coef is None:
                out[key] = -value
            elif isinstance(value, pd.DataFrame):
                idx = [name for name in value.index if str(name) in names]
                out[key] = value.copy()
                out[key].loc[idx, :] = -out[key].loc[idx, :]
            else:
                out[key] = value
        return out
    for attr in ("betas",):
        if hasattr(fit, attr) and coef is None:
            setattr(fit, attr, -getattr(fit, attr))
    return fit


def t_to_beta_se(t: ArrayLike, df: ArrayLike, n: Optional[ArrayLike] = None) -> Dict[str, NDArray[np.float64]]:
    """Convert t-statistics to approximate beta and SE estimates."""
    del df
    t_arr = np.asarray(t, dtype=np.float64)
    if n is None:
        se_t = np.ones_like(t_arr, dtype=np.float64)
    else:
        se_t = np.sqrt(1.0 / np.asarray(n, dtype=np.float64))
    return {"beta": t_arr * se_t, "se": np.broadcast_to(se_t, t_arr.shape).copy()}


def hrf_smoothing_kernel(
    length: int,
    tr: float = 2.0,
    form: str = "onset ~ trialwise()",
    buffer_scans: int = 3,
    normalize: bool = True,
    method: str = "gram",
) -> NDArray[np.float64]:
    """Compute a temporal smoothing kernel from a trialwise design matrix."""
    if method not in {"gram", "cosine"}:
        raise ValueError("method must be 'gram' or 'cosine'")
    if length <= 0:
        raise ValueError("length must be positive")
    from .. import SamplingFrame, event_model

    n_buf = int(buffer_scans)
    sf = SamplingFrame(blocklens=int(length) + 2 * n_buf, tr=tr)
    data = pd.DataFrame({"onset": sf.sample_times(), "block": 1})
    em = event_model(form, data=data, sampling_frame=sf, durations=tr)
    design = getattr(em, "design_matrix")
    X = np.asarray(design() if callable(design) else design, dtype=np.float64)
    if method == "gram":
        K = X @ X.T
    else:
        norms = np.sqrt(np.sum(X * X, axis=0))
        Xn = X / np.where(norms > 0, norms, 1.0)
        K = Xn @ Xn.T
    if normalize:
        diag = np.diag(K)
        K = K / np.where(diag[:, np.newaxis] != 0, diag[:, np.newaxis], 1.0)
    keep = slice(n_buf, n_buf + int(length))
    return K[keep, keep]


def estimate(*args: object, **kwargs: object) -> None:
    """Deprecated fmrireg helper retained only to guide migration."""
    raise RuntimeError("estimate() is deprecated. Use estimate_betas() instead.")
