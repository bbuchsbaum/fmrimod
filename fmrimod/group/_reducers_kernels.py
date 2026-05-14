"""Pure statistical kernels used by :mod:`fmrimod.group.reducers`.

This is the first slice of the policy/kernel/registry split tracked by
bd-01KRHTJ9WFSSBZSDGAN4V7PHGS. Every helper here is a pure compute
primitive that depends only on ``numpy`` (and ``scipy.stats`` for
``_t_p_two_sided``): no dataset, registry, or policy types. Keeping
these isolated lets the reducer file shrink toward orchestration and
makes individual helpers easier to read, test, and reuse.

Nothing here is part of the public API.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


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


def _pack_upper_tri(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    idx = np.triu_indices(matrix.shape[0])
    return matrix[idx]


def _flat_lmm(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return values.reshape(values.shape[0], 1, 1)


__all__ = [
    "_flatten_feature_axis",
    "_unflatten_feature_axis",
    "_safe_inverse",
    "_two_sided_perm_count",
    "_max_abs_null",
    "_t_p_two_sided",
    "_clamp_cpp_p_values",
    "_fe_weights_and_q",
    "_pack_upper_tri",
    "_flat_lmm",
]
