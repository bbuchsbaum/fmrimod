"""Compatibility accessors for fmrireg-style result and data objects."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as sp_stats


def _as_array(value: Any) -> NDArray[np.float64]:
    return np.asarray(value, dtype=np.float64)


def _model_from_result(x: Any) -> Any:
    return getattr(x, "model", x)


def coef_names(x: Any, include_baseline: bool = True, **kwargs) -> list[str]:
    """Return coefficient names for a model or fitted GLM result."""
    model = _model_from_result(x)
    if hasattr(model, "design_matrix"):
        dm = model.design_matrix()
        names = [str(name) for name in getattr(dm, "columns", [])]
        if names:
            if include_baseline:
                return names
            n_event = getattr(model, "n_event_columns", None)
            return names[: int(n_event)] if n_event is not None else names

    event_model = getattr(model, "event_model", model)
    names = getattr(event_model, "column_names", None)
    if names is not None:
        return [str(name) for name in names]

    n_coef = getattr(x, "n_coefficients", None)
    if n_coef is not None:
        return [f"coef_{idx}" for idx in range(int(n_coef))]
    raise NotImplementedError(f"coef_names not implemented for {type(x)}")


def ar_parameters(x: Any, scope: str = "average", **kwargs) -> Optional[Any]:
    """Return AR parameters stored on an ``FmriLm``-like result."""
    if scope not in {"average", "per_run", "raw"}:
        raise ValueError("scope must be 'average', 'per_run', or 'raw'")
    ar = getattr(x, "ar_params", None)
    if ar is None:
        return None
    if scope == "raw":
        return ar
    arr = _as_array(ar)
    if arr.size == 0:
        return None
    if scope == "per_run":
        if arr.ndim <= 1:
            return [arr]
        return [np.asarray(arr[..., i]).squeeze() for i in range(arr.shape[-1])]
    if arr.ndim == 1:
        return arr
    return np.nanmean(arr, axis=-1)


def standard_error(x: Any, type: str = "estimates", **kwargs) -> Any:
    """Return standard errors for coefficient estimates or contrasts."""
    if type == "estimates":
        method = getattr(x, "se", None)
        if callable(method):
            return method()
        se_val = getattr(x, "se", None)
        if se_val is not None:
            return se_val
    if type == "contrasts":
        return {
            name: result.se
            for name, result in getattr(x, "contrasts", {}).items()
            if getattr(result, "se", None) is not None
        }
    raise ValueError("type must be 'estimates' or 'contrasts'")


def se(x: Any, *args, **kwargs) -> Any:
    """Alias for :func:`standard_error`."""
    return standard_error(x, *args, **kwargs)


def stats(x: Any, type: str = "estimates", **kwargs) -> Any:
    """Return coefficient, t-contrast, or F-contrast statistics."""
    if type == "estimates":
        method = getattr(x, "tstat", None)
        if callable(method):
            return method()
    if type == "contrasts":
        return {
            name: result.stat
            for name, result in getattr(x, "contrasts", {}).items()
            if getattr(result, "stat_type", None) == "t"
        }
    if type == "F":
        return {
            name: result.stat
            for name, result in getattr(x, "contrasts", {}).items()
            if getattr(result, "stat_type", None) == "F"
        }
    raise ValueError("type must be 'estimates', 'contrasts', or 'F'")


def p_values(x: Any, type: str = "estimates", **kwargs) -> Any:
    """Return two-sided p-values for coefficient estimates or contrasts."""
    if type == "estimates":
        t = stats(x, type="estimates")
        df = float(getattr(x, "residual_df"))
        return 2.0 * sp_stats.t.sf(np.abs(t), df=df)
    if type in {"contrasts", "F"}:
        wanted = "F" if type == "F" else "t"
        return {
            name: result.p_value
            for name, result in getattr(x, "contrasts", {}).items()
            if getattr(result, "stat_type", None) == wanted
        }
    raise ValueError("type must be 'estimates', 'contrasts', or 'F'")


def pvalues(x: Any, *args, **kwargs) -> Any:
    """Alias for :func:`p_values`."""
    return p_values(x, *args, **kwargs)


def zscores(x: Any, type: str = "estimates", **kwargs) -> Any:
    """Convert two-sided p-values and statistic signs to z scores."""
    if type == "estimates":
        t = stats(x, type="estimates")
        p = p_values(x, type="estimates")
        return sp_stats.norm.isf(np.clip(p, 1e-300, 1.0) / 2.0) * np.sign(t)
    stat_map = stats(x, type=type)
    p_map = p_values(x, type=type)
    return {
        name: sp_stats.norm.isf(np.clip(p_map[name], 1e-300, 1.0) / 2.0)
        * np.sign(stat)
        for name, stat in stat_map.items()
    }


def _coef_matrix(x: Any, include_baseline: bool = True) -> NDArray[np.float64]:
    method = getattr(x, "coef", None)
    if callable(method):
        vals = method()
    else:
        vals = getattr(x, "betas", None)
    if vals is None:
        raise NotImplementedError(f"coefficient extraction not implemented for {type(x)}")
    arr = _as_array(vals)
    if include_baseline:
        return arr
    n_event = getattr(getattr(x, "model", None), "n_event_columns", None)
    return arr[: int(n_event)] if n_event is not None else arr


def coef_image(
    x: Any,
    coef: Union[int, str] = 0,
    statistic: str = "estimate",
    mask: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.float64]:
    """Return a coefficient/statistic vector or reconstruct it into a mask."""
    statistic = {"beta": "estimate", "t": "stat", "tstat": "stat", "p": "pvalue"}.get(
        statistic,
        statistic,
    )
    if statistic == "estimate":
        values = _coef_matrix(x, include_baseline=True)
    elif statistic in {"se", "std_error"}:
        values = standard_error(x, type="estimates")
    elif statistic == "stat":
        values = stats(x, type="estimates")
    elif statistic in {"pvalue", "p_values", "pvalues"}:
        values = p_values(x, type="estimates")
    else:
        raise ValueError("Unsupported statistic")

    names = coef_names(x, include_baseline=True)
    idx = names.index(coef) if isinstance(coef, str) else int(coef)
    vec = _as_array(values)[idx]

    if mask is None:
        dataset = getattr(getattr(x, "model", None), "dataset", None)
        if dataset is not None and hasattr(dataset, "get_mask"):
            try:
                mask = dataset.get_mask()
            except Exception:
                mask = None
    if mask is None:
        return vec
    mask_arr = np.asarray(mask, dtype=bool)
    out = np.full(mask_arr.shape, np.nan, dtype=np.float64)
    out[mask_arr] = vec
    return out


def get_data(x: Any, run: int = 0, **kwargs) -> NDArray[np.float64]:
    """Return run data from a dataset-like object."""
    method = getattr(x, "get_data", None)
    if callable(method):
        return _as_array(method(run, **kwargs))
    raise NotImplementedError(f"get_data not implemented for {type(x)}")


def get_data_matrix(x: Any, **kwargs) -> NDArray[np.float64]:
    """Return all run data concatenated along time."""
    if hasattr(x, "n_runs") and hasattr(x, "get_data"):
        return np.vstack([get_data(x, run=i, **kwargs) for i in range(int(x.n_runs))])
    return get_data(x, **kwargs)


def get_mask(x: Any, **kwargs) -> NDArray[np.bool_]:
    """Return a boolean mask from a dataset-like object."""
    method = getattr(x, "get_mask", None)
    if callable(method):
        return np.asarray(method(**kwargs), dtype=bool)
    dataset = getattr(x, "dataset", None)
    if dataset is not None:
        return get_mask(dataset, **kwargs)
    raise NotImplementedError(f"get_mask not implemented for {type(x)}")


def get_formula(x: Any, **kwargs) -> Optional[str]:
    """Return the stored formula when available, otherwise a term summary."""
    obj = _model_from_result(x)
    direct = getattr(obj, "formula", None)
    if direct is not None:
        return str(direct)
    event_model = getattr(obj, "event_model", obj)
    direct = getattr(event_model, "formula", None)
    if direct is not None:
        return str(direct)
    terms = getattr(event_model, "terms", None)
    if terms is not None:
        names = [getattr(term, "name", str(term)) for term in terms]
        return " + ".join(str(name) for name in names)
    return None


def get_subjects(x: Any, **kwargs) -> list[str]:
    """Return subject identifiers from a group-data object."""
    subjects = getattr(x, "subjects", None)
    if subjects is not None:
        return list(subjects)
    data = getattr(x, "data", None)
    if isinstance(data, dict) and "subjects" in data:
        return list(data["subjects"])
    raise NotImplementedError(f"get_subjects not implemented for {type(x)}")


def n_subjects(x: Any, **kwargs) -> int:
    """Return the number of subjects in a group-data object."""
    if hasattr(x, "n_subjects"):
        return int(getattr(x, "n_subjects"))
    return len(get_subjects(x))


def get_covariates(x: Any, **kwargs) -> Optional[pd.DataFrame]:
    """Return group-data covariates."""
    cov = getattr(x, "covariates", None)
    if cov is None:
        return None
    return cov.copy() if isinstance(cov, pd.DataFrame) else cov


def get_rois(x: Any, **kwargs) -> list[str]:
    """Return ROI labels from CSV-backed group data when present."""
    data = getattr(x, "data", None)
    if isinstance(data, dict):
        df = data.get("data")
        roi_col = data.get("roi_col")
        if isinstance(df, pd.DataFrame) and roi_col in df.columns:
            return list(pd.unique(df[roi_col].dropna()))
    return []


def get_contrasts(x: Any, **kwargs) -> Any:
    """Return contrast names/specifications from fitted or group-data objects."""
    contrasts = getattr(x, "contrasts", None)
    if isinstance(contrasts, dict):
        return list(contrasts.keys())
    data = getattr(x, "data", None)
    if isinstance(data, dict):
        df = data.get("data")
        contrast_col = data.get("contrast_col")
        if isinstance(df, pd.DataFrame) and contrast_col in df.columns:
            return list(pd.unique(df[contrast_col].dropna()))
        contrast = data.get("contrast")
        if contrast is not None:
            return [contrast]
    return []


def tidy(x: Any, type: str = "estimates", **kwargs) -> pd.DataFrame:
    """Return a long-form DataFrame of estimates, SEs, statistics, and p-values."""
    if type == "estimates":
        estimates = _coef_matrix(x, include_baseline=True)
        se_vals = standard_error(x, type="estimates")
        stat_vals = stats(x, type="estimates")
        p_vals = p_values(x, type="estimates")
        names = coef_names(x, include_baseline=True)
        rows = []
        for coef_idx, name in enumerate(names):
            for voxel in range(estimates.shape[1]):
                rows.append(
                    {
                        "term": name,
                        "voxel": voxel,
                        "estimate": estimates[coef_idx, voxel],
                        "std_error": se_vals[coef_idx, voxel],
                        "stat": stat_vals[coef_idx, voxel],
                        "statistic": stat_vals[coef_idx, voxel],
                        "p_value": p_vals[coef_idx, voxel],
                    }
                )
        return pd.DataFrame(rows)

    if type in {"contrasts", "F"}:
        rows = []
        wanted = "F" if type == "F" else "t"
        for name, result in getattr(x, "contrasts", {}).items():
            if getattr(result, "stat_type", None) != wanted:
                continue
            for voxel, stat in enumerate(np.ravel(result.stat)):
                rows.append(
                    {
                        "term": name,
                        "voxel": voxel,
                        "estimate": np.ravel(result.estimate)[voxel],
                        "std_error": None if result.se is None else np.ravel(result.se)[voxel],
                        "stat": stat,
                        "statistic": stat,
                        "p_value": np.ravel(result.p_value)[voxel],
                    }
                )
        return pd.DataFrame(rows)
    raise ValueError("type must be 'estimates', 'contrasts', or 'F'")


def fitted_hrf(x: Any, sample_at: Sequence[float], **kwargs) -> dict[str, Any]:
    """Best-effort fitted HRF reconstruction from event-term coefficients."""
    sample = _as_array(sample_at)
    if np.any(~np.isfinite(sample)):
        raise ValueError("sample_at must contain finite values")

    model = getattr(x, "model", None)
    event_model = getattr(model, "event_model", None)
    if event_model is None:
        return {}
    terms = getattr(event_model, "terms", [])
    col_indices = getattr(event_model, "column_indices", {}) or {}
    betas = _coef_matrix(x, include_baseline=True)
    out: dict[str, Any] = {}

    for idx, term in enumerate(terms):
        term_name = getattr(term, "name", f"term_{idx}")
        hrf_spec = getattr(term, "hrf", None)
        if hrf_spec is None:
            continue
        hrf_obj = event_model._resolve_hrf(hrf_spec)
        basis = np.atleast_2d(hrf_obj(sample))
        if basis.shape[0] != len(sample):
            basis = basis.T
        nbasis = int(getattr(hrf_obj, "nbasis", basis.shape[1]))
        indices = list(col_indices.get(term_name, []))
        if not indices or len(indices) % nbasis != 0:
            continue
        pred_blocks = []
        condition_names = []
        for block_idx, start in enumerate(range(0, len(indices), nbasis)):
            coef_block = betas[indices[start : start + nbasis], :]
            pred_blocks.append(basis @ coef_block)
            condition_names.append(f"{term_name}_{block_idx}")
        design = pd.DataFrame(
            {
                "condition": np.repeat(condition_names, len(sample)),
                "time": np.tile(sample, len(condition_names)),
            }
        )
        out[term_name] = {"pred": np.vstack(pred_blocks), "design": design}
    return out


def tidy_fitted_hrf(x: Any, sample_at: Sequence[float], **kwargs) -> pd.DataFrame:
    """Return fitted HRF reconstructions in long tabular form."""
    rows = []
    for term, payload in fitted_hrf(x, sample_at=sample_at, **kwargs).items():
        pred = _as_array(payload["pred"])
        design = payload["design"].reset_index(drop=True)
        for row_idx, meta in design.iterrows():
            for voxel in range(pred.shape[1]):
                rows.append(
                    {
                        "term": term,
                        "condition": meta["condition"],
                        "time": meta["time"],
                        "voxel": voxel,
                        "estimate": pred[row_idx, voxel],
                        "value": pred[row_idx, voxel],
                    }
                )
    return pd.DataFrame(rows)
