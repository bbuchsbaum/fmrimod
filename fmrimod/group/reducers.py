"""Native group-analysis reducers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from math import ceil
from typing import Any, Literal, TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from patsy import dmatrix
from scipy import stats as sp_stats

from fmrimod.stats.inference import z_to_p

from .dataset import GroupDataset
from .errors import AdapterContractError, UnsupportedGroupFeatureError
from .registry import reducer_registry

Tail = Literal["two.sided"]
T = TypeVar("T")


@contextmanager
def _maybe_limit_blas_threads(blas_threads: int | None) -> Iterator[None]:
    if blas_threads is None:
        yield
        return
    if int(blas_threads) < 1:
        raise AdapterContractError("blas_threads must be >= 1")
    try:
        from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
    except Exception:  # pragma: no cover - optional dependency behavior
        yield
        return
    with threadpool_limits(limits=int(blas_threads)):
        yield


def _feature_chunks(
    n_features: int,
    *,
    n_jobs: int = 1,
    chunk_size: int | None = None,
) -> tuple[list[tuple[int, int]], int]:
    if int(n_jobs) < 1:
        raise AdapterContractError("n_jobs must be >= 1")
    if chunk_size is not None and int(chunk_size) < 1:
        raise AdapterContractError("chunk_size must be >= 1")
    if n_features < 1:
        return [], 1
    n_workers = min(int(n_jobs), n_features)
    size = int(chunk_size) if chunk_size is not None else max(1, ceil(n_features / n_workers))
    chunks = [
        (start, min(start + size, n_features))
        for start in range(0, n_features, size)
    ]
    return chunks, min(n_workers, len(chunks))


def _run_feature_chunks(
    n_features: int,
    worker: Callable[[int, int], T],
    *,
    n_jobs: int = 1,
    chunk_size: int | None = None,
    blas_threads: int | None = None,
) -> tuple[list[T], int, int | None]:
    chunks, n_workers = _feature_chunks(
        n_features,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    )
    if not chunks:
        return [], n_workers, chunk_size
    if n_workers == 1:
        with _maybe_limit_blas_threads(blas_threads):
            return [worker(start, end) for start, end in chunks], n_workers, chunk_size
    with _maybe_limit_blas_threads(blas_threads), ThreadPoolExecutor(
        max_workers=n_workers
    ) as pool:
        results = list(pool.map(lambda bounds: worker(*bounds), chunks))
    return results, n_workers, chunk_size


def _beta_and_var(
    dataset: GroupDataset,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    beta = dataset.assay("beta")
    if "var" in dataset.assays:
        var = dataset.assay("var")
    elif "se" in dataset.assays:
        se = dataset.assay("se")
        var = se * se
    else:
        raise AdapterContractError("meta:fe requires beta with var or se")
    return beta, var


def _group_col_data() -> pd.DataFrame:
    return pd.DataFrame(index=pd.Index(["group"], name="subject"))


def _reduced_dataset(
    dataset: GroupDataset,
    assays: dict[str, NDArray[np.float64]],
    *,
    method: str,
    metadata: dict[str, Any] | None = None,
) -> GroupDataset:
    return GroupDataset(
        assays=assays,
        space=dataset.space,
        subjects=["group"],
        contrasts=dataset.contrasts,
        col_data=_group_col_data(),
        row_data=dataset.row_data,
        contrast_data=dataset.contrast_data,
        metadata={
            **dict(dataset.metadata),
            "operation": "reduce",
            "reduce_method": method,
            **({} if metadata is None else metadata),
        },
    )


def _design_matrix(
    dataset: GroupDataset,
    *,
    X: NDArray[np.float64] | None = None,
    formula: str = "~ 1",
) -> tuple[NDArray[np.float64], list[str]]:
    if X is not None:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise AdapterContractError("X must be a 2-D subjects x predictors matrix")
        if X_arr.shape[0] != dataset.n_subjects:
            raise AdapterContractError("X rows must match number of subjects")
        if not np.all(np.isfinite(X_arr)):
            raise AdapterContractError("X must contain finite values")
        return X_arr, [f"x{i}" for i in range(X_arr.shape[1])]

    clean = formula.replace(" ", "")
    if clean in ("~1", "1"):
        return np.ones((dataset.n_subjects, 1), dtype=np.float64), ["Intercept"]
    if dataset.col_data is None:
        raise AdapterContractError("formula with covariates requires dataset.col_data")
    design = dmatrix(
        formula, dataset.col_data.reset_index(drop=True), return_type="dataframe"
    )
    return np.asarray(design, dtype=np.float64), list(design.columns)


def _lmm_formula(formula: str) -> str:
    if "|" in formula:
        raise UnsupportedGroupFeatureError(
            "LMM reducers do not support lmer-style random-effects syntax; "
            "choose method='lmm:ri' or method='lmm:ri_slope1' and pass reducer options"
        )
    clean = formula.strip()
    if not clean:
        clean = "~ 1"
    if not clean.startswith("~"):
        raise AdapterContractError("LMM formula must be one-sided, e.g. '~ condition'")
    return clean


def _lmm_observation_data(dataset: GroupDataset) -> pd.DataFrame:
    subject_rows = (
        pd.DataFrame(index=pd.Index(dataset.subjects, name="subject"))
        if dataset.col_data is None
        else dataset.col_data.copy()
    )
    subject_rows = subject_rows.copy()
    subject_rows.index = pd.Index(dataset.subjects, name="subject")
    if "subject" in subject_rows.columns:
        subject_rows = subject_rows.drop(columns=["subject"])

    contrast_rows = (
        pd.DataFrame(index=pd.Index(dataset.contrasts, name="contrast"))
        if dataset.contrast_data is None
        else dataset.contrast_data.copy()
    )
    contrast_rows = contrast_rows.copy()
    contrast_rows.index = pd.Index(dataset.contrasts, name="contrast")
    if "contrast" in contrast_rows.columns:
        contrast_rows = contrast_rows.drop(columns=["contrast"])
    contrast_rows["contrast"] = list(dataset.contrasts)

    rows: list[pd.DataFrame] = []
    for subject in dataset.subjects:
        subj = subject_rows.loc[[subject]].reset_index(drop=True)
        repeated_subj = pd.concat([subj] * dataset.n_contrasts, ignore_index=True)
        block = pd.concat(
            [
                pd.DataFrame({"subject": [subject] * dataset.n_contrasts}),
                repeated_subj,
                contrast_rows.reset_index(drop=True),
            ],
            axis=1,
        )
        rows.append(block)
    obs = pd.concat(rows, ignore_index=True)
    obs["subject"] = pd.Categorical(obs["subject"], categories=list(dataset.subjects))
    obs["contrast"] = pd.Categorical(obs["contrast"], categories=list(dataset.contrasts))
    return obs


def _lmm_fixed_design(
    dataset: GroupDataset,
    *,
    formula: str,
) -> tuple[pd.DataFrame, NDArray[np.float64], list[str]]:
    obs = _lmm_observation_data(dataset)
    design = dmatrix(_lmm_formula(formula), obs, return_type="dataframe")
    X = np.asarray(design, dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise AdapterContractError("LMM fixed-effects design contains non-finite values")
    return obs, X, list(design.columns)


def _lmm_beta_matrix(dataset: GroupDataset) -> NDArray[np.float64]:
    beta = dataset.assay("beta")
    return beta.reshape(dataset.n_samples, dataset.n_subjects * dataset.n_contrasts).T


def _lmm_not_available(message: str) -> UnsupportedGroupFeatureError:
    return UnsupportedGroupFeatureError(
        f"{message}. Use backend='fmrigds-r' as the explicit R oracle/fallback "
        "during migration, or use theta_mode='voxelwise' for the native "
        "statsmodels first slice where supported."
    )


def _fit_mixedlm_feature(
    y: NDArray[np.float64],
    X: NDArray[np.float64],
    groups: Any,
    exog_re: NDArray[np.float64],
    *,
    reml: bool,
    covariance: Literal["diag", "full"] = "full",
) -> Any:
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMParams
    except Exception as exc:  # pragma: no cover - dependency declared by package
        raise UnsupportedGroupFeatureError(
            "native LMM reducers require optional dependency 'statsmodels'"
        ) from exc

    model = MixedLM(endog=y, exog=X, groups=groups, exog_re=exog_re)
    fit_options: dict[str, Any] = {}
    if covariance == "diag" and exog_re.shape[1] > 1:
        fit_options["free"] = MixedLMParams.from_components(
            fe_params=np.ones(X.shape[1]),
            cov_re=np.eye(exog_re.shape[1]),
        )
    return model.fit(reml=reml, method="lbfgs", disp=False, **fit_options)


def _flatten_feature_axis(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.transpose(arr, (1, 0, 2)).reshape(
        arr.shape[1], arr.shape[0] * arr.shape[2]
    )


def _unflatten_feature_axis(
    arr: NDArray[np.float64],
    *,
    n_sample: int,
    n_contrast: int,
) -> NDArray[np.float64]:
    return arr.reshape(n_sample, n_contrast).reshape(n_sample, 1, n_contrast)


def _safe_inverse(matrix: NDArray[np.float64]) -> NDArray[np.float64] | None:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None


def _two_sided_perm_count(null_stats: NDArray[np.float64], observed: float) -> float:
    if not np.isfinite(observed):
        return np.nan
    return float(
        np.sum(np.isfinite(null_stats) & (np.abs(null_stats) >= abs(observed)))
    )


def _max_abs_null(null_stats: NDArray[np.float64]) -> NDArray[np.float64]:
    abs_null = np.abs(null_stats)
    safe = np.where(np.isfinite(abs_null), abs_null, -np.inf)
    max_abs = np.max(safe, axis=1)
    max_abs[~np.isfinite(max_abs)] = 0.0
    return cast(NDArray[np.float64], max_abs)


def _t_p_two_sided(t_value: float, df: float) -> float:
    if not np.isfinite(t_value) or not np.isfinite(df) or df <= 0:
        return np.nan
    return float(2.0 * sp_stats.t.sf(abs(t_value), df))


def _clamp_cpp_p_values(p: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(
        np.isfinite(p),
        np.clip(p, 1e-300, 1.0 - 1e-16),
        np.nan,
    )


def _fe_weights_and_q(
    beta: NDArray[np.float64],
    var: NDArray[np.float64],
    *,
    eps: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    ok = np.isfinite(beta) & np.isfinite(var) & (var > 0)
    safe_var = np.where(ok, np.maximum(var, eps), np.nan)
    weights = np.where(ok, 1.0 / safe_var, 0.0)
    sw = np.sum(weights, axis=1, keepdims=True)
    wy = np.sum(weights * np.where(ok, beta, 0.0), axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_fe = wy / sw
        resid = beta - mu_fe
        q = np.sum(weights * np.where(ok, resid * resid, 0.0), axis=1, keepdims=True)
    k = np.sum(ok, axis=1, keepdims=True)
    return weights, sw, q, k


def meta_fe(
    dataset: GroupDataset,
    *,
    eps: float = 1e-12,
    min_subjects: int = 2,
    alternative: Tail = "two.sided",
) -> GroupDataset:
    """Fixed-effects inverse-variance reducer.

    This mirrors the fmrigds ``meta:fe`` assay contract for the native
    ``sample x subject x contrast`` representation.
    """
    if alternative != "two.sided":
        raise AdapterContractError("meta:fe currently supports only two.sided tests")
    beta, var = _beta_and_var(dataset)
    if beta.shape != var.shape:
        raise AdapterContractError("beta and var/se assays must have matching shapes")
    if eps <= 0:
        raise AdapterContractError("eps must be > 0")
    if min_subjects < 1:
        raise AdapterContractError("min_subjects must be >= 1")

    ok = np.isfinite(beta) & np.isfinite(var) & (var > 0)
    weights, sw, q, k = _fe_weights_and_q(beta, var, eps=eps)
    wy = np.sum(weights * np.where(ok, beta, 0.0), axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        beta_g = wy / sw
        var_g = 1.0 / sw
        se_g = np.sqrt(var_g)
        i2 = np.maximum(0.0, (q - (k - 1.0)) / np.maximum(q, eps))
        z_g = beta_g / se_g
        p_g = z_to_p(z_g)

    bad = (k < int(min_subjects)) | ~np.isfinite(sw) | (sw <= 0)
    assays = {
        "beta_g": beta_g,
        "var_g": var_g,
        "se_g": se_g,
        "z_g": z_g,
        "p_g": p_g,
        "Q": q,
        "I2": i2,
    }
    for arr in assays.values():
        arr[bad] = np.nan

    return _reduced_dataset(
        dataset,
        assays,
        method="meta:fe",
        metadata={"weights": "ivw", "min_subjects": int(min_subjects)},
    )


def meta_re(
    dataset: GroupDataset,
    *,
    eps: float = 1e-12,
    min_subjects: int = 2,
    alternative: Tail = "two.sided",
) -> GroupDataset:
    """DerSimonian-Laird random-effects inverse-variance reducer."""
    if alternative != "two.sided":
        raise AdapterContractError("meta:re currently supports only two.sided tests")
    beta, var = _beta_and_var(dataset)
    if beta.shape != var.shape:
        raise AdapterContractError("beta and var/se assays must have matching shapes")
    if eps <= 0:
        raise AdapterContractError("eps must be > 0")
    if min_subjects < 1:
        raise AdapterContractError("min_subjects must be >= 1")

    ok = np.isfinite(beta) & np.isfinite(var) & (var > 0)
    w_fe, sw_fe, q, k = _fe_weights_and_q(beta, var, eps=eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        c_term = sw_fe - np.sum(w_fe * w_fe, axis=1, keepdims=True) / sw_fe
        tau2 = np.maximum(0.0, (q - (k - 1.0)) / np.maximum(c_term, eps))
        w_star = np.where(ok, 1.0 / (np.maximum(var, eps) + tau2), 0.0)
        sws = np.sum(w_star, axis=1, keepdims=True)
        wys = np.sum(w_star * np.where(ok, beta, 0.0), axis=1, keepdims=True)
        beta_g = wys / sws
        var_g = 1.0 / sws
        se_g = np.sqrt(var_g)
        z_g = beta_g / se_g
        p_g = z_to_p(z_g)
        i2 = np.maximum(0.0, (q - (k - 1.0)) / np.maximum(q, eps))

    bad = (k < int(min_subjects)) | ~np.isfinite(sws) | (sws <= 0)
    assays = {
        "beta_g": beta_g,
        "var_g": var_g,
        "se_g": se_g,
        "z_g": z_g,
        "p_g": p_g,
        "tau2": tau2,
        "Q": q,
        "I2": i2,
    }
    for arr in assays.values():
        arr[bad] = np.nan

    return _reduced_dataset(
        dataset,
        assays,
        method="meta:re",
        metadata={"weights": "ivw", "tau2": "DL", "min_subjects": int(min_subjects)},
    )


def combine_stouffer(
    dataset: GroupDataset,
    *,
    weights: NDArray[np.float64] | list[float] | float | None = None,
    min_subjects: int = 1,
) -> GroupDataset:
    """Combine subject-level z-scores with Stouffer's method."""
    z = dataset.assay("z")
    if min_subjects < 1:
        raise AdapterContractError("min_subjects must be >= 1")
    finite = np.isfinite(z)
    if weights is None:
        numerator = np.sum(np.where(finite, z, 0.0), axis=1, keepdims=True)
        k = np.sum(finite, axis=1, keepdims=True)
        denominator = np.sqrt(k)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim == 0:
            w = np.full(dataset.n_subjects, float(w))
        if w.shape != (dataset.n_subjects,):
            raise AdapterContractError("weights length must equal number of subjects")
        if not np.all(np.isfinite(w)):
            raise AdapterContractError("weights must be finite")
        w3 = w.reshape(1, dataset.n_subjects, 1)
        numerator = np.sum(np.where(finite, z * w3, 0.0), axis=1, keepdims=True)
        denominator = np.sqrt(
            np.sum(np.where(finite, w3 * w3, 0.0), axis=1, keepdims=True)
        )
        k = np.sum(finite, axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        z_g = numerator / denominator
    bad = (k < int(min_subjects)) | ~np.isfinite(denominator) | (denominator <= 0)
    z_g[bad] = np.nan
    p_g = z_to_p(z_g)

    return _reduced_dataset(
        dataset,
        {"z_g": z_g, "p_g": p_g},
        method="combine:stouffer",
        metadata={"min_subjects": int(min_subjects)},
    )


def combine_fisher(
    dataset: GroupDataset,
    *,
    min_subjects: int = 1,
) -> GroupDataset:
    """Combine subject-level p-values with Fisher's method."""
    p = dataset.assay("p")
    if min_subjects < 1:
        raise AdapterContractError("min_subjects must be >= 1")
    finite = np.isfinite(p)
    clamped = _clamp_cpp_p_values(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = -2.0 * np.sum(
            np.where(finite, np.log(clamped), 0.0),
            axis=1,
            keepdims=True,
        )
    k = np.sum(finite, axis=1, keepdims=True)
    df = 2.0 * k.astype(np.float64)
    p_g = sp_stats.chi2.sf(chi2, df)
    bad = k < int(min_subjects)
    chi2[bad] = np.nan
    df[bad] = np.nan
    p_g[bad] = np.nan

    return _reduced_dataset(
        dataset,
        {"p_g": p_g, "chi2": chi2, "df": df},
        method="combine:fisher",
        metadata={"min_subjects": int(min_subjects)},
    )


def combine_lancaster(
    dataset: GroupDataset,
    *,
    dfw: NDArray[np.float64] | list[float],
    min_subjects: int = 1,
) -> GroupDataset:
    """Combine subject-level p-values with Lancaster's weighted chi-square method."""
    p = dataset.assay("p")
    raw_weights = np.asarray(dfw, dtype=np.float64)
    if raw_weights.shape != (dataset.n_subjects,):
        raise AdapterContractError("dfw length must equal number of subjects")
    if not np.all(np.isfinite(raw_weights)):
        raise AdapterContractError("dfw must contain finite values")
    weights = np.maximum(raw_weights.astype(np.int64), 1).astype(np.float64)
    if min_subjects < 1:
        raise AdapterContractError("min_subjects must be >= 1")

    finite = np.isfinite(p)
    clamped = _clamp_cpp_p_values(p)
    w3 = weights.reshape(1, dataset.n_subjects, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_terms = sp_stats.chi2.ppf(1.0 - clamped, df=2.0 * w3)
    chi2 = np.sum(np.where(finite, chi_terms, 0.0), axis=1, keepdims=True)
    df = 2.0 * np.sum(np.where(finite, w3, 0.0), axis=1, keepdims=True)
    p_g = sp_stats.chi2.sf(chi2, df)
    k = np.sum(finite, axis=1, keepdims=True)
    bad = k < int(min_subjects)
    chi2[bad] = np.nan
    df[bad] = np.nan
    p_g[bad] = np.nan

    return _reduced_dataset(
        dataset,
        {"p_g": p_g, "chi2": chi2, "df": df},
        method="combine:lancaster",
        metadata={"min_subjects": int(min_subjects)},
    )


def meta_fe_reg(
    dataset: GroupDataset,
    *,
    formula: str = "~ 1",
    X: NDArray[np.float64] | None = None,
    eps: float = 1e-12,
) -> GroupDataset:
    """Fixed-effects meta-regression reducer."""
    beta, var = _beta_and_var(dataset)
    if beta.shape != var.shape:
        raise AdapterContractError("beta and var/se assays must have matching shapes")
    if eps <= 0:
        raise AdapterContractError("eps must be > 0")
    X_mat, predictor_names = _design_matrix(dataset, X=X, formula=formula)
    pcols = X_mat.shape[1]
    beta_2d = _flatten_feature_axis(beta)
    var_2d = _flatten_feature_axis(var)
    n_features = beta_2d.shape[1]

    coef = np.full((pcols, n_features), np.nan, dtype=np.float64)
    se_coef = np.full((pcols, n_features), np.nan, dtype=np.float64)
    q = np.full(n_features, np.nan, dtype=np.float64)
    df_res = np.full(n_features, np.nan, dtype=np.float64)

    for feature_idx in range(n_features):
        y = beta_2d[:, feature_idx]
        v = var_2d[:, feature_idx]
        w = 1.0 / np.maximum(v, eps)
        ok = np.isfinite(y) & np.isfinite(w)
        if np.sum(ok) < (pcols + 1):
            continue
        Xok = X_mat[ok, :]
        wok = w[ok]
        yok = y[ok]
        Xw = Xok * np.sqrt(wok)[:, np.newaxis]
        gram = Xw.T @ Xw
        gram_inv = _safe_inverse(gram)
        if gram_inv is None:
            continue
        bhat = gram_inv @ (Xok.T @ (wok * yok))
        resid = yok - Xok @ bhat
        coef[:, feature_idx] = bhat
        se_coef[:, feature_idx] = np.sqrt(np.maximum(np.diag(gram_inv), 0.0))
        q[feature_idx] = np.sum(wok * resid * resid)
        df_res[feature_idx] = np.sum(ok) - pcols

    assays: dict[str, NDArray[np.float64]] = {
        "Q": _unflatten_feature_axis(
            q, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
        "df_res": _unflatten_feature_axis(
            df_res,
            n_sample=dataset.n_samples,
            n_contrast=dataset.n_contrasts,
        ),
    }
    for pred_idx, name in enumerate(predictor_names):
        coef_arr = _unflatten_feature_axis(
            coef[pred_idx, :],
            n_sample=dataset.n_samples,
            n_contrast=dataset.n_contrasts,
        )
        se_arr = _unflatten_feature_axis(
            se_coef[pred_idx, :],
            n_sample=dataset.n_samples,
            n_contrast=dataset.n_contrasts,
        )
        z_arr = coef_arr / se_arr
        assays[f"coef:{name}"] = coef_arr
        assays[f"se_coef:{name}"] = se_arr
        assays[f"z_coef:{name}"] = z_arr
        assays[f"p_coef:{name}"] = z_to_p(z_arr)

    return _reduced_dataset(
        dataset,
        assays,
        method="meta:fe_reg",
        metadata={"formula": formula, "predictor_names": tuple(predictor_names)},
    )


def meta_re_reg(
    dataset: GroupDataset,
    *,
    formula: str = "~ 1",
    X: NDArray[np.float64] | None = None,
    eps: float = 1e-12,
) -> GroupDataset:
    """DerSimonian-Laird random-effects meta-regression reducer."""
    beta, var = _beta_and_var(dataset)
    if beta.shape != var.shape:
        raise AdapterContractError("beta and var/se assays must have matching shapes")
    if eps <= 0:
        raise AdapterContractError("eps must be > 0")
    X_mat, predictor_names = _design_matrix(dataset, X=X, formula=formula)
    pcols = X_mat.shape[1]
    beta_2d = _flatten_feature_axis(beta)
    var_2d = _flatten_feature_axis(var)
    n_features = beta_2d.shape[1]

    coef = np.full((pcols, n_features), np.nan, dtype=np.float64)
    se_coef = np.full((pcols, n_features), np.nan, dtype=np.float64)
    tau2 = np.full(n_features, np.nan, dtype=np.float64)
    q = np.full(n_features, np.nan, dtype=np.float64)
    df_res = np.full(n_features, np.nan, dtype=np.float64)

    for feature_idx in range(n_features):
        y = beta_2d[:, feature_idx]
        v = var_2d[:, feature_idx]
        w = 1.0 / np.maximum(v, eps)
        ok = np.isfinite(y) & np.isfinite(w)
        if np.sum(ok) < (pcols + 1):
            continue
        Xok = X_mat[ok, :]
        wok = w[ok]
        yok = y[ok]
        Xw = Xok * np.sqrt(wok)[:, np.newaxis]
        gram = Xw.T @ Xw
        gram_inv = _safe_inverse(gram)
        if gram_inv is None:
            continue
        bhat_fe = gram_inv @ (Xok.T @ (wok * yok))
        resid = yok - Xok @ bhat_fe
        q_val = float(np.sum(wok * resid * resid))
        tr_h = float(np.sum(wok * np.sum((Xok @ gram_inv) * Xok, axis=1)))
        c_term = float(np.sum(wok) - tr_h)
        df_val = float(np.sum(ok) - pcols)
        tau_val = max((q_val - df_val) / max(c_term, eps), 0.0)

        w_star = 1.0 / ((1.0 / wok) + tau_val)
        Xws = Xok * np.sqrt(w_star)[:, np.newaxis]
        gram_star = Xws.T @ Xws
        gram_star_inv = _safe_inverse(gram_star)
        if gram_star_inv is None:
            continue
        bhat = gram_star_inv @ (Xok.T @ (w_star * yok))

        coef[:, feature_idx] = bhat
        se_coef[:, feature_idx] = np.sqrt(np.maximum(np.diag(gram_star_inv), 0.0))
        tau2[feature_idx] = tau_val
        q[feature_idx] = q_val
        df_res[feature_idx] = df_val

    assays = {
        "tau2": _unflatten_feature_axis(
            tau2, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
        "Q": _unflatten_feature_axis(
            q, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
        "df_res": _unflatten_feature_axis(
            df_res,
            n_sample=dataset.n_samples,
            n_contrast=dataset.n_contrasts,
        ),
    }
    for pred_idx, name in enumerate(predictor_names):
        coef_arr = _unflatten_feature_axis(
            coef[pred_idx, :],
            n_sample=dataset.n_samples,
            n_contrast=dataset.n_contrasts,
        )
        se_arr = _unflatten_feature_axis(
            se_coef[pred_idx, :],
            n_sample=dataset.n_samples,
            n_contrast=dataset.n_contrasts,
        )
        z_arr = coef_arr / se_arr
        assays[f"coef:{name}"] = coef_arr
        assays[f"se_coef:{name}"] = se_arr
        assays[f"z_coef:{name}"] = z_arr
        assays[f"p_coef:{name}"] = z_to_p(z_arr)

    return _reduced_dataset(
        dataset,
        assays,
        method="meta:re_reg",
        metadata={
            "formula": formula,
            "predictor_names": tuple(predictor_names),
            "tau2": "DL",
        },
    )


def perm_onesample(
    dataset: GroupDataset,
    *,
    signs: NDArray[np.integer[Any]] | None = None,
    n_perm: int = 5000,
    seed: int | None = None,
    include_observed: bool = False,
    min_subjects: int = 2,
    alternative: Tail = "two.sided",
    n_jobs: int = 1,
    chunk_size: int | None = None,
    blas_threads: int | None = None,
) -> GroupDataset:
    """One-sample sign-flip permutation t reducer."""
    if alternative != "two.sided":
        raise AdapterContractError(
            "perm:onesample currently supports only two.sided tests"
        )
    beta = dataset.assay("beta")
    if min_subjects < 2:
        raise AdapterContractError("min_subjects must be >= 2")
    if signs is None:
        if n_perm < 1:
            raise AdapterContractError("n_perm must be positive")
        rng = np.random.default_rng(seed)
        sign_mat = rng.choice(
            np.array([-1, 1], dtype=np.int8),
            size=(int(n_perm), dataset.n_subjects),
            replace=True,
        )
        if include_observed:
            sign_mat[0, :] = 1
    else:
        sign_mat = np.asarray(signs, dtype=np.int8)
        if sign_mat.ndim != 2 or sign_mat.shape[1] != dataset.n_subjects:
            raise AdapterContractError("signs must have shape n_perm x n_subjects")
        if not np.all(np.isin(sign_mat, [-1, 1])):
            raise AdapterContractError("signs must contain only -1 and 1")

    beta_2d = _flatten_feature_axis(beta)
    n_features = beta_2d.shape[1]
    beta_g = np.full(n_features, np.nan, dtype=np.float64)
    se_g = np.full(n_features, np.nan, dtype=np.float64)
    t_g = np.full(n_features, np.nan, dtype=np.float64)
    df = np.full(n_features, np.nan, dtype=np.float64)
    p_g = np.full(n_features, np.nan, dtype=np.float64)
    p_perm = np.full(n_features, np.nan, dtype=np.float64)
    p_fwer = np.full(n_features, np.nan, dtype=np.float64)
    null_stats = np.full((sign_mat.shape[0], n_features), np.nan, dtype=np.float64)

    def worker(
        start: int,
        end: int,
    ) -> tuple[
        int,
        int,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        width = end - start
        beta_chunk = np.full(width, np.nan, dtype=np.float64)
        se_chunk = np.full(width, np.nan, dtype=np.float64)
        t_chunk = np.full(width, np.nan, dtype=np.float64)
        df_chunk = np.full(width, np.nan, dtype=np.float64)
        p_chunk = np.full(width, np.nan, dtype=np.float64)
        p_perm_chunk = np.full(width, np.nan, dtype=np.float64)
        null_chunk = np.full((sign_mat.shape[0], width), np.nan, dtype=np.float64)

        for local_idx, feature_idx in enumerate(range(start, end)):
            y = beta_2d[:, feature_idx]
            ok = np.isfinite(y)
            n_ok = int(np.sum(ok))
            if n_ok < min_subjects:
                continue
            y_ok = y[ok]
            mean = float(np.mean(y_ok))
            sd = float(np.std(y_ok, ddof=1))
            se = sd / np.sqrt(n_ok)
            if se <= 0 or not np.isfinite(se):
                continue
            obs_t = mean / se
            beta_chunk[local_idx] = mean
            se_chunk[local_idx] = se
            t_chunk[local_idx] = obs_t
            df_chunk[local_idx] = n_ok - 1
            p_chunk[local_idx] = _t_p_two_sided(obs_t, df_chunk[local_idx])

            feature_signs = sign_mat[:, ok]
            perm_y = feature_signs * y_ok[np.newaxis, :]
            perm_mean = np.mean(perm_y, axis=1)
            perm_sd = np.std(perm_y, axis=1, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                null_chunk[:, local_idx] = perm_mean / (perm_sd / np.sqrt(n_ok))
            count = _two_sided_perm_count(null_chunk[:, local_idx], obs_t)
            p_perm_chunk[local_idx] = (count + 1.0) / (sign_mat.shape[0] + 1.0)

        return (
            start,
            end,
            beta_chunk,
            se_chunk,
            t_chunk,
            df_chunk,
            p_chunk,
            p_perm_chunk,
            null_chunk,
        )

    chunks, n_workers, effective_chunk_size = _run_feature_chunks(
        n_features,
        worker,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        blas_threads=blas_threads,
    )
    for (
        start,
        end,
        beta_chunk,
        se_chunk,
        t_chunk,
        df_chunk,
        p_chunk,
        p_perm_chunk,
        null_chunk,
    ) in chunks:
        beta_g[start:end] = beta_chunk
        se_g[start:end] = se_chunk
        t_g[start:end] = t_chunk
        df[start:end] = df_chunk
        p_g[start:end] = p_chunk
        p_perm[start:end] = p_perm_chunk
        null_stats[:, start:end] = null_chunk

    max_abs = _max_abs_null(null_stats)
    for feature_idx in range(n_features):
        count = _two_sided_perm_count(max_abs, t_g[feature_idx])
        p_fwer[feature_idx] = (count + 1.0) / (sign_mat.shape[0] + 1.0)

    assays = {
        "beta_g": _unflatten_feature_axis(
            beta_g, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
        "se_g": _unflatten_feature_axis(
            se_g, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
        "t_g": _unflatten_feature_axis(
            t_g, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
        "df": _unflatten_feature_axis(
            df, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
        "p_g": _unflatten_feature_axis(
            p_g, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
        "p_perm": _unflatten_feature_axis(
            p_perm, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
        "p_fwer": _unflatten_feature_axis(
            p_fwer, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
    }
    return _reduced_dataset(
        dataset,
        assays,
        method="perm:onesample",
        metadata={
            "n_perm": int(sign_mat.shape[0]),
            "min_subjects": int(min_subjects),
            "n_jobs": int(n_workers),
            "chunk_size": effective_chunk_size,
            "blas_threads": blas_threads,
        },
    )


def perm_twosample(
    dataset: GroupDataset,
    *,
    group: NDArray[np.integer[Any]] | list[int] | list[str],
    group_mat: NDArray[np.integer[Any]] | None = None,
    n_perm: int = 5000,
    seed: int | None = None,
    include_observed: bool = True,
    variance: Literal["welch", "pooled"] = "welch",
    min_group: int = 2,
    alternative: Tail = "two.sided",
    n_jobs: int = 1,
    chunk_size: int | None = None,
    blas_threads: int | None = None,
) -> GroupDataset:
    """Two-sample label-permutation t reducer."""
    if alternative != "two.sided":
        raise AdapterContractError(
            "perm:twosample currently supports only two.sided tests"
        )
    if variance not in ("welch", "pooled"):
        raise AdapterContractError("variance must be 'welch' or 'pooled'")
    if min_group < 1:
        raise AdapterContractError("min_group must be >= 1")
    beta = dataset.assay("beta")
    raw_group = np.asarray(group)
    if raw_group.shape != (dataset.n_subjects,):
        raise AdapterContractError("group must have length n_subjects")
    _, group_codes = np.unique(raw_group, return_inverse=True)
    if set(group_codes.tolist()) != {0, 1}:
        raise AdapterContractError("group must contain exactly two groups")

    if group_mat is None:
        if n_perm < 1:
            raise AdapterContractError("n_perm must be positive")
        rng = np.random.default_rng(seed)
        perm_mat = np.vstack(
            [rng.permutation(group_codes) for _ in range(int(n_perm))]
        ).astype(np.int8)
        if include_observed:
            perm_mat[0, :] = group_codes
    else:
        perm_mat = np.asarray(group_mat, dtype=np.int8)
        if perm_mat.ndim != 2 or perm_mat.shape[1] != dataset.n_subjects:
            raise AdapterContractError("group_mat must have shape n_perm x n_subjects")
        if not np.all(np.isin(perm_mat, [0, 1])):
            raise AdapterContractError("group_mat must encode groups as 0/1")

    beta_2d = _flatten_feature_axis(beta)
    n_features = beta_2d.shape[1]
    beta_g = np.full(n_features, np.nan, dtype=np.float64)
    se_g = np.full(n_features, np.nan, dtype=np.float64)
    t_g = np.full(n_features, np.nan, dtype=np.float64)
    df = np.full(n_features, np.nan, dtype=np.float64)
    p_g = np.full(n_features, np.nan, dtype=np.float64)
    p_perm = np.full(n_features, np.nan, dtype=np.float64)
    p_fwer = np.full(n_features, np.nan, dtype=np.float64)
    null_stats = np.full((perm_mat.shape[0], n_features), np.nan, dtype=np.float64)

    def two_sample_t(
        y: NDArray[np.float64], labels: NDArray[np.integer[Any]]
    ) -> tuple[float, float, float, float]:
        g0 = y[labels == 0]
        g1 = y[labels == 1]
        if len(g0) < min_group or len(g1) < min_group:
            return np.nan, np.nan, np.nan, np.nan
        m0 = float(np.mean(g0))
        m1 = float(np.mean(g1))
        v0 = float(np.var(g0, ddof=1))
        v1 = float(np.var(g1, ddof=1))
        if variance == "pooled":
            df_val = float(len(g0) + len(g1) - 2)
            pooled = ((len(g0) - 1) * v0 + (len(g1) - 1) * v1) / df_val
            se_val = np.sqrt(pooled * (1.0 / len(g0) + 1.0 / len(g1)))
        else:
            a = v0 / len(g0)
            b = v1 / len(g1)
            se_val = np.sqrt(a + b)
            df_val = (a + b) ** 2 / ((a * a) / (len(g0) - 1) + (b * b) / (len(g1) - 1))
        if se_val <= 0 or not np.isfinite(se_val):
            return np.nan, np.nan, np.nan, np.nan
        diff = m1 - m0
        return diff, se_val, diff / se_val, float(df_val)

    def worker(
        start: int,
        end: int,
    ) -> tuple[
        int,
        int,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        width = end - start
        beta_chunk = np.full(width, np.nan, dtype=np.float64)
        se_chunk = np.full(width, np.nan, dtype=np.float64)
        t_chunk = np.full(width, np.nan, dtype=np.float64)
        df_chunk = np.full(width, np.nan, dtype=np.float64)
        p_chunk = np.full(width, np.nan, dtype=np.float64)
        p_perm_chunk = np.full(width, np.nan, dtype=np.float64)
        null_chunk = np.full((perm_mat.shape[0], width), np.nan, dtype=np.float64)

        for local_idx, feature_idx in enumerate(range(start, end)):
            y = beta_2d[:, feature_idx]
            ok = np.isfinite(y)
            y_ok = y[ok]
            observed_labels = group_codes[ok]
            diff, se, obs_t, df_val = two_sample_t(y_ok, observed_labels)
            beta_chunk[local_idx] = diff
            se_chunk[local_idx] = se
            t_chunk[local_idx] = obs_t
            df_chunk[local_idx] = df_val
            p_chunk[local_idx] = _t_p_two_sided(obs_t, df_val)

            for perm_idx, labels in enumerate(perm_mat[:, ok]):
                _, _, perm_t, _ = two_sample_t(y_ok, labels)
                null_chunk[perm_idx, local_idx] = perm_t
            count = _two_sided_perm_count(null_chunk[1:, local_idx], obs_t)
            p_perm_chunk[local_idx] = (count + 1.0) / float(perm_mat.shape[0])

        return (
            start,
            end,
            beta_chunk,
            se_chunk,
            t_chunk,
            df_chunk,
            p_chunk,
            p_perm_chunk,
            null_chunk,
        )

    chunks, n_workers, effective_chunk_size = _run_feature_chunks(
        n_features,
        worker,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        blas_threads=blas_threads,
    )
    for (
        start,
        end,
        beta_chunk,
        se_chunk,
        t_chunk,
        df_chunk,
        p_chunk,
        p_perm_chunk,
        null_chunk,
    ) in chunks:
        beta_g[start:end] = beta_chunk
        se_g[start:end] = se_chunk
        t_g[start:end] = t_chunk
        df[start:end] = df_chunk
        p_g[start:end] = p_chunk
        p_perm[start:end] = p_perm_chunk
        null_stats[:, start:end] = null_chunk

    max_abs = _max_abs_null(null_stats)
    for feature_idx in range(n_features):
        count = _two_sided_perm_count(max_abs[1:], t_g[feature_idx])
        p_fwer[feature_idx] = (count + 1.0) / float(perm_mat.shape[0])

    assays = {
        "beta_g": _unflatten_feature_axis(
            beta_g, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
        "se_g": _unflatten_feature_axis(
            se_g, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
        "t_g": _unflatten_feature_axis(
            t_g, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
        "df": _unflatten_feature_axis(
            df, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
        "p_g": _unflatten_feature_axis(
            p_g, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
        "p_perm": _unflatten_feature_axis(
            p_perm, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
        "p_fwer": _unflatten_feature_axis(
            p_fwer, n_sample=dataset.n_samples, n_contrast=dataset.n_contrasts
        ),
    }
    return _reduced_dataset(
        dataset,
        assays,
        method="perm:twosample",
        metadata={
            "n_perm": int(perm_mat.shape[0]),
            "variance": variance,
            "min_group": int(min_group),
            "n_jobs": int(n_workers),
            "chunk_size": effective_chunk_size,
            "blas_threads": blas_threads,
        },
    )


def _pack_upper_tri(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    idx = np.triu_indices(matrix.shape[0])
    return matrix[idx]


def ols_voxelwise(
    dataset: GroupDataset,
    *,
    formula: str = "~ 1",
    X: NDArray[np.float64] | None = None,
    return_cov: Literal["none", "tri"] = "none",
    n_jobs: int = 1,
    chunk_size: int | None = None,
    blas_threads: int | None = None,
) -> GroupDataset:
    """OLS reducer across subjects for each sample/contrast feature."""
    if return_cov not in ("none", "tri"):
        raise AdapterContractError("return_cov must be 'none' or 'tri'")
    beta = dataset.assay("beta")
    X_mat, predictor_names = _design_matrix(dataset, X=X, formula=formula)
    pcols = X_mat.shape[1]
    beta_2d = _flatten_feature_axis(beta)
    n_features = beta_2d.shape[1]
    tri_len = pcols * (pcols + 1) // 2

    coef = np.full((pcols, n_features), np.nan, dtype=np.float64)
    se_coef = np.full((pcols, n_features), np.nan, dtype=np.float64)
    sigma2 = np.full(n_features, np.nan, dtype=np.float64)
    df_res = np.full(n_features, np.nan, dtype=np.float64)
    cov_tri = np.full((tri_len, n_features), np.nan, dtype=np.float64)

    def worker(
        start: int,
        end: int,
    ) -> tuple[
        int,
        int,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        width = end - start
        coef_chunk = np.full((pcols, width), np.nan, dtype=np.float64)
        se_chunk = np.full((pcols, width), np.nan, dtype=np.float64)
        sigma2_chunk = np.full(width, np.nan, dtype=np.float64)
        df_chunk = np.full(width, np.nan, dtype=np.float64)
        cov_chunk = np.full((tri_len, width), np.nan, dtype=np.float64)

        for local_idx, feature_idx in enumerate(range(start, end)):
            y = beta_2d[:, feature_idx]
            ok = np.isfinite(y) & np.all(np.isfinite(X_mat), axis=1)
            if np.sum(ok) <= pcols:
                continue
            Xok = X_mat[ok, :]
            yok = y[ok]
            xtx_inv = _safe_inverse(Xok.T @ Xok)
            if xtx_inv is None:
                continue
            bhat = xtx_inv @ (Xok.T @ yok)
            resid = yok - Xok @ bhat
            df_val = float(np.sum(ok) - pcols)
            sigma2_val = float(np.sum(resid * resid) / df_val)
            cov = xtx_inv * sigma2_val
            coef_chunk[:, local_idx] = bhat
            se_chunk[:, local_idx] = np.sqrt(np.maximum(np.diag(cov), 0.0))
            sigma2_chunk[local_idx] = sigma2_val
            df_chunk[local_idx] = df_val
            cov_chunk[:, local_idx] = _pack_upper_tri(cov)

        return start, end, coef_chunk, se_chunk, sigma2_chunk, df_chunk, cov_chunk

    chunks, n_workers, effective_chunk_size = _run_feature_chunks(
        n_features,
        worker,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        blas_threads=blas_threads,
    )
    for start, end, coef_chunk, se_chunk, sigma2_chunk, df_chunk, cov_chunk in chunks:
        coef[:, start:end] = coef_chunk
        se_coef[:, start:end] = se_chunk
        sigma2[start:end] = sigma2_chunk
        df_res[start:end] = df_chunk
        cov_tri[:, start:end] = cov_chunk

    assays: dict[str, NDArray[np.float64]] = {
        "sigma2": _unflatten_feature_axis(
            sigma2,
            n_sample=dataset.n_samples,
            n_contrast=dataset.n_contrasts,
        ),
        "df_res": _unflatten_feature_axis(
            df_res,
            n_sample=dataset.n_samples,
            n_contrast=dataset.n_contrasts,
        ),
    }
    for pred_idx, name in enumerate(predictor_names):
        coef_arr = _unflatten_feature_axis(
            coef[pred_idx, :],
            n_sample=dataset.n_samples,
            n_contrast=dataset.n_contrasts,
        )
        se_arr = _unflatten_feature_axis(
            se_coef[pred_idx, :],
            n_sample=dataset.n_samples,
            n_contrast=dataset.n_contrasts,
        )
        t_arr = coef_arr / se_arr
        assays[f"coef:{name}"] = coef_arr
        assays[f"se_coef:{name}"] = se_arr
        assays[f"t_coef:{name}"] = t_arr
        assays[f"p_coef:{name}"] = 2.0 * sp_stats.t.sf(np.abs(t_arr), assays["df_res"])

    if return_cov == "tri":
        for tri_idx in range(tri_len):
            assays[f"cov_tri:{tri_idx}"] = _unflatten_feature_axis(
                cov_tri[tri_idx, :],
                n_sample=dataset.n_samples,
                n_contrast=dataset.n_contrasts,
            )

    return _reduced_dataset(
        dataset,
        assays,
        method="ols:voxelwise",
        metadata={
            "formula": formula,
            "predictor_names": tuple(predictor_names),
            "return_cov": return_cov,
            "n_jobs": int(n_workers),
            "chunk_size": effective_chunk_size,
            "blas_threads": blas_threads,
        },
    )


def lmm_unavailable(
    dataset: GroupDataset,
    *,
    method: Literal["lmm:ri", "lmm:ri_slope1"],
    **options: Any,
) -> GroupDataset:
    """Explicit native gap for fmrigds LMM reducers."""
    raise UnsupportedGroupFeatureError(
        f"{method} is not yet implemented natively in fmrimod.group. "
        "Use the explicit R fmrigds oracle/fallback during migration, or wait for "
        "the dedicated native LMM milestone against fmrigds lmm_core.cpp."
    )


def _lmm_result_dataset(
    dataset: GroupDataset,
    assays: dict[str, NDArray[np.float64]],
    *,
    method: str,
    metadata: dict[str, Any],
) -> GroupDataset:
    return GroupDataset(
        assays=assays,
        space=dataset.space,
        subjects=["meta"],
        contrasts=["model"],
        col_data=pd.DataFrame(index=pd.Index(["meta"], name="subject")),
        row_data=dataset.row_data,
        contrast_data=pd.DataFrame(
            {"label": ["model"]},
            index=pd.Index(["model"], name="contrast"),
        ),
        metadata={
            **dict(dataset.metadata),
            "operation": "reduce",
            "reduce_method": method,
            **metadata,
        },
    )


def _flat_lmm(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return values.reshape(values.shape[0], 1, 1)


def lmm_ri(
    dataset: GroupDataset,
    *,
    formula: str = "~ 1",
    fit: Literal["REML", "ML"] = "REML",
    theta_mode: Literal["pooled", "voxelwise"] = "pooled",
) -> GroupDataset:
    """Native random-intercept LMM reducer for voxelwise theta mode."""
    if theta_mode != "voxelwise":
        raise _lmm_not_available(
            "native lmm:ri currently supports theta_mode='voxelwise' only"
        )
    if fit not in ("REML", "ML"):
        raise AdapterContractError("fit must be 'REML' or 'ML'")

    obs, X, predictor_names = _lmm_fixed_design(dataset, formula=formula)
    y_mat = _lmm_beta_matrix(dataset)
    n_obs, n_samples = y_mat.shape
    pcols = X.shape[1]
    groups = obs["subject"].to_numpy()
    exog_re = np.ones((n_obs, 1), dtype=np.float64)
    df_val = float(max(n_obs - pcols, 1))

    coef = np.full((pcols, n_samples), np.nan, dtype=np.float64)
    se_coef = np.full((pcols, n_samples), np.nan, dtype=np.float64)
    t_coef = np.full((pcols, n_samples), np.nan, dtype=np.float64)
    p_coef = np.full((pcols, n_samples), np.nan, dtype=np.float64)
    sigma2 = np.full(n_samples, np.nan, dtype=np.float64)
    vc_intercept = np.full(n_samples, np.nan, dtype=np.float64)
    log_lik = np.full(n_samples, np.nan, dtype=np.float64)
    converged = np.zeros(n_samples, dtype=np.float64)
    lambda_intercept = np.full(n_samples, np.nan, dtype=np.float64)

    for feature_idx in range(n_samples):
        y = y_mat[:, feature_idx]
        if not np.all(np.isfinite(y)):
            continue
        try:
            result = _fit_mixedlm_feature(
                y,
                X,
                groups,
                exog_re,
                reml=fit == "REML",
            )
        except Exception:
            continue
        fe = np.asarray(result.fe_params, dtype=np.float64)
        se = np.asarray(result.bse_fe, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_val = fe / se
            p_val = 2.0 * sp_stats.t.sf(np.abs(t_val), df_val)
        scale = float(result.scale)
        cov_re = np.asarray(result.cov_re, dtype=np.float64)
        vc_i = float(cov_re[0, 0]) if cov_re.size else np.nan
        coef[:, feature_idx] = fe
        se_coef[:, feature_idx] = se
        t_coef[:, feature_idx] = t_val
        p_coef[:, feature_idx] = p_val
        sigma2[feature_idx] = scale
        vc_intercept[feature_idx] = vc_i
        log_lik[feature_idx] = float(result.llf)
        converged[feature_idx] = 1.0 if bool(getattr(result, "converged", False)) else 0.0
        lambda_intercept[feature_idx] = vc_i / scale if scale > 0 else np.nan

    df_res = np.full(n_samples, df_val, dtype=np.float64)
    assays: dict[str, NDArray[np.float64]] = {
        "sigma2": _flat_lmm(sigma2),
        "vc_intercept": _flat_lmm(vc_intercept),
        "vc_resid": _flat_lmm(sigma2),
        "df_res": _flat_lmm(df_res),
        "logLik": _flat_lmm(log_lik),
        "converged": _flat_lmm(converged),
        "lambda": _flat_lmm(lambda_intercept),
    }
    for pred_idx, name in enumerate(predictor_names):
        assays[f"coef:{name}"] = _flat_lmm(coef[pred_idx])
        assays[f"se_coef:{name}"] = _flat_lmm(se_coef[pred_idx])
        assays[f"t_coef:{name}"] = _flat_lmm(t_coef[pred_idx])
        assays[f"p_coef:{name}"] = _flat_lmm(p_coef[pred_idx])

    return _lmm_result_dataset(
        dataset,
        assays,
        method="lmm:ri",
        metadata={
            "formula": formula,
            "fit": fit,
            "theta_mode": theta_mode,
            "predictor_names": tuple(predictor_names),
            "engine": "statsmodels.MixedLM",
        },
    )


def lmm_ri_slope1(
    dataset: GroupDataset,
    *,
    formula: str = "~ 1",
    slope: str | None = None,
    covariance: Literal["diag", "full"] = "diag",
    fit: Literal["REML", "ML"] = "REML",
    theta_mode: Literal["pooled", "voxelwise"] = "pooled",
    center_slope: bool = False,
) -> GroupDataset:
    """Native random-intercept plus one-slope LMM for voxelwise full covariance."""
    if slope is None or not slope:
        raise AdapterContractError("lmm:ri_slope1 requires a slope option")
    if theta_mode != "voxelwise":
        raise _lmm_not_available(
            "native lmm:ri_slope1 currently supports theta_mode='voxelwise' only"
        )
    if covariance not in ("diag", "full"):
        raise AdapterContractError("covariance must be 'diag' or 'full'")
    if fit not in ("REML", "ML"):
        raise AdapterContractError("fit must be 'REML' or 'ML'")

    obs, X, predictor_names = _lmm_fixed_design(dataset, formula=formula)
    if slope not in obs:
        raise AdapterContractError(f"slope variable '{slope}' is not present")
    slope_values = np.asarray(obs[slope], dtype=np.float64)
    if not np.all(np.isfinite(slope_values)):
        raise AdapterContractError(f"slope variable '{slope}' must be numeric and finite")
    if center_slope:
        slope_values = slope_values - float(np.mean(slope_values))
    if len(np.unique(slope_values)) < 2:
        raise AdapterContractError(f"slope variable '{slope}' must vary")

    y_mat = _lmm_beta_matrix(dataset)
    n_obs, n_samples = y_mat.shape
    pcols = X.shape[1]
    groups = obs["subject"].to_numpy()
    exog_re = np.column_stack([np.ones(n_obs, dtype=np.float64), slope_values])
    df_val = float(max(n_obs - pcols, 1))

    coef = np.full((pcols, n_samples), np.nan, dtype=np.float64)
    se_coef = np.full((pcols, n_samples), np.nan, dtype=np.float64)
    t_coef = np.full((pcols, n_samples), np.nan, dtype=np.float64)
    p_coef = np.full((pcols, n_samples), np.nan, dtype=np.float64)
    sigma2 = np.full(n_samples, np.nan, dtype=np.float64)
    vc_intercept = np.full(n_samples, np.nan, dtype=np.float64)
    vc_slope = np.full(n_samples, np.nan, dtype=np.float64)
    vc_cov = np.full(n_samples, np.nan, dtype=np.float64)
    log_lik = np.full(n_samples, np.nan, dtype=np.float64)
    converged = np.zeros(n_samples, dtype=np.float64)
    lambda_intercept = np.full(n_samples, np.nan, dtype=np.float64)
    lambda_slope = np.full(n_samples, np.nan, dtype=np.float64)
    lambda_cov = np.full(n_samples, np.nan, dtype=np.float64)
    corr = np.full(n_samples, np.nan, dtype=np.float64)

    for feature_idx in range(n_samples):
        y = y_mat[:, feature_idx]
        if not np.all(np.isfinite(y)):
            continue
        try:
            result = _fit_mixedlm_feature(
                y,
                X,
                groups,
                exog_re,
                reml=fit == "REML",
                covariance=covariance,
            )
        except Exception:
            continue
        fe = np.asarray(result.fe_params, dtype=np.float64)
        se = np.asarray(result.bse_fe, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_val = fe / se
            p_val = 2.0 * sp_stats.t.sf(np.abs(t_val), df_val)
        scale = float(result.scale)
        cov_re = np.asarray(result.cov_re, dtype=np.float64)
        vc_i = float(cov_re[0, 0])
        vc_s = float(cov_re[1, 1])
        vc_is = float(cov_re[0, 1])
        coef[:, feature_idx] = fe
        se_coef[:, feature_idx] = se
        t_coef[:, feature_idx] = t_val
        p_coef[:, feature_idx] = p_val
        sigma2[feature_idx] = scale
        vc_intercept[feature_idx] = vc_i
        vc_slope[feature_idx] = vc_s
        vc_cov[feature_idx] = vc_is
        log_lik[feature_idx] = float(result.llf)
        converged[feature_idx] = 1.0 if bool(getattr(result, "converged", False)) else 0.0
        lambda_intercept[feature_idx] = vc_i / scale if scale > 0 else np.nan
        lambda_slope[feature_idx] = vc_s / scale if scale > 0 else np.nan
        lambda_cov[feature_idx] = vc_is / scale if scale > 0 else np.nan
        denom = np.sqrt(vc_i * vc_s)
        corr[feature_idx] = vc_is / denom if denom > 0 else 0.0

    df_res = np.full(n_samples, df_val, dtype=np.float64)
    assays: dict[str, NDArray[np.float64]] = {
        "sigma2": _flat_lmm(sigma2),
        "vc_intercept": _flat_lmm(vc_intercept),
        "vc_slope": _flat_lmm(vc_slope),
        "vc_cov_intercept_slope": _flat_lmm(vc_cov),
        "vc_resid": _flat_lmm(sigma2),
        "df_res": _flat_lmm(df_res),
        "logLik": _flat_lmm(log_lik),
        "converged": _flat_lmm(converged),
        "lambda_intercept": _flat_lmm(lambda_intercept),
        "lambda_slope": _flat_lmm(lambda_slope),
        "lambda_cov_intercept_slope": _flat_lmm(lambda_cov),
        "corr_intercept_slope": _flat_lmm(corr),
    }
    for pred_idx, name in enumerate(predictor_names):
        assays[f"coef:{name}"] = _flat_lmm(coef[pred_idx])
        assays[f"se_coef:{name}"] = _flat_lmm(se_coef[pred_idx])
        assays[f"t_coef:{name}"] = _flat_lmm(t_coef[pred_idx])
        assays[f"p_coef:{name}"] = _flat_lmm(p_coef[pred_idx])

    return _lmm_result_dataset(
        dataset,
        assays,
        method="lmm:ri_slope1",
        metadata={
            "formula": formula,
            "slope": slope,
            "covariance": covariance,
            "fit": fit,
            "theta_mode": theta_mode,
            "center_slope": bool(center_slope),
            "predictor_names": tuple(predictor_names),
            "engine": "statsmodels.MixedLM",
        },
    )


def register_core_reducers(*, overwrite: bool = True) -> None:
    """Register built-in native reducers."""
    reducer_registry.register(
        "meta:fe",
        meta_fe,
        description="Fixed-effects inverse-variance reducer",
        overwrite=overwrite,
    )
    reducer_registry.register(
        "meta:re",
        meta_re,
        description="DerSimonian-Laird random-effects inverse-variance reducer",
        overwrite=overwrite,
    )
    reducer_registry.register(
        "combine:stouffer",
        combine_stouffer,
        description="Stouffer z-score combiner",
        overwrite=overwrite,
    )
    reducer_registry.register(
        "combine:fisher",
        combine_fisher,
        description="Fisher p-value combiner",
        overwrite=overwrite,
    )
    reducer_registry.register(
        "combine:lancaster",
        combine_lancaster,
        description="Lancaster weighted p-value combiner",
        overwrite=overwrite,
    )
    reducer_registry.register(
        "meta:fe_reg",
        meta_fe_reg,
        description="Fixed-effects meta-regression reducer",
        overwrite=overwrite,
    )
    reducer_registry.register(
        "meta:re_reg",
        meta_re_reg,
        description="DerSimonian-Laird random-effects meta-regression reducer",
        overwrite=overwrite,
    )
    reducer_registry.register(
        "perm:onesample",
        perm_onesample,
        description="One-sample sign-flip permutation t reducer",
        overwrite=overwrite,
    )
    reducer_registry.register(
        "perm:twosample",
        perm_twosample,
        description="Two-sample label-permutation t reducer",
        overwrite=overwrite,
    )
    reducer_registry.register(
        "ols:voxelwise",
        ols_voxelwise,
        description="OLS reducer across subjects for each feature",
        overwrite=overwrite,
    )
    reducer_registry.register(
        "lmm:ri",
        lmm_ri,
        description="Native LMM random-intercept reducer milestone placeholder",
        overwrite=overwrite,
    )
    reducer_registry.register(
        "lmm:ri_slope1",
        lmm_ri_slope1,
        description="Native LMM random-intercept plus one-slope reducer milestone placeholder",
        overwrite=overwrite,
    )


register_core_reducers()
