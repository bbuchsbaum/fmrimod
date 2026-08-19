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
    return _perm_tail_count(null_stats, observed, "two.sided")


def _perm_tail_score(
    statistic: NDArray[np.float64] | float,
    alternative: str,
) -> NDArray[np.float64] | float:
    """Score used for tail-matched permutation / FWER counts.

    ``less`` uses ``-t``, ``greater`` uses ``t``, ``two.sided`` uses ``|t|``.
    Cheap-pass disqualifier for fmrigds #22: accepting ``alternative``
    while still ranking on ``|t|``.
    """

    if alternative == "less":
        return np.negative(statistic)
    if alternative == "greater":
        return statistic
    return np.abs(statistic)


def _perm_tail_count(
    null_stats: NDArray[np.float64],
    observed: float,
    alternative: str,
) -> float:
    if not np.isfinite(observed):
        return np.nan
    if alternative == "less":
        return float(np.sum(np.isfinite(null_stats) & (null_stats <= observed)))
    if alternative == "greater":
        return float(np.sum(np.isfinite(null_stats) & (null_stats >= observed)))
    return float(
        np.sum(np.isfinite(null_stats) & (np.abs(null_stats) >= abs(observed)))
    )


def _max_abs_null(null_stats: NDArray[np.float64]) -> NDArray[np.float64]:
    return _max_tail_null(null_stats, "two.sided")


def _max_tail_null(
    null_stats: NDArray[np.float64],
    alternative: str,
) -> NDArray[np.float64]:
    scores = np.asarray(_perm_tail_score(null_stats, alternative), dtype=np.float64)
    safe = np.where(np.isfinite(scores), scores, -np.inf)
    max_score = np.max(safe, axis=1)
    max_score[~np.isfinite(max_score)] = np.nan
    return cast(NDArray[np.float64], max_score)


def _t_p_two_sided(t_value: float, df: float) -> float:
    return _t_p(t_value, df, "two.sided")


def _t_p(t_value: float, df: float, alternative: str) -> float:
    if not np.isfinite(t_value) or not np.isfinite(df) or df <= 0:
        return np.nan
    if alternative == "greater":
        return float(sp_stats.t.sf(t_value, df))
    if alternative == "less":
        return float(sp_stats.t.cdf(t_value, df))
    return float(2.0 * sp_stats.t.sf(abs(t_value), df))


def _weighted_onesample_stats(
    y: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    min_subjects: int,
) -> tuple[float, float, float, float]:
    """Reliability-weighted one-sample mean / SE / t / df (fmrigds #21)."""

    ok = np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    n_ok = int(np.sum(ok))
    if n_ok < min_subjects:
        return np.nan, np.nan, np.nan, np.nan
    y_ok = y[ok]
    w = weights[ok]
    scale = float(np.max(w))
    if not np.isfinite(scale) or scale <= 0:
        return np.nan, np.nan, np.nan, np.nan
    w = w / scale
    sum_w = float(np.sum(w))
    sum_w2 = float(np.sum(w * w))
    if sum_w <= 0 or sum_w2 <= 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.sum(w * y_ok) / sum_w)
    variance_denom = sum_w - sum_w2 / sum_w
    if variance_denom <= 0 or not np.isfinite(variance_denom):
        return np.nan, np.nan, np.nan, np.nan
    centered = float(np.sum(w * y_ok * y_ok) - sum_w * mean * mean)
    if centered < 0.0 and centered > -1e-10:
        centered = 0.0
    variance = centered / variance_denom
    effective_n = (sum_w * sum_w) / sum_w2
    if variance <= 0 or not np.isfinite(variance) or effective_n <= 1.0:
        return np.nan, np.nan, np.nan, np.nan
    se = float(np.sqrt(variance / effective_n))
    if se <= 0 or not np.isfinite(se):
        return np.nan, np.nan, np.nan, np.nan
    return mean, se, mean / se, effective_n - 1.0


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
    "_perm_tail_score",
    "_perm_tail_count",
    "_max_abs_null",
    "_max_tail_null",
    "_t_p_two_sided",
    "_t_p",
    "_weighted_onesample_stats",
    "_clamp_cpp_p_values",
    "_fe_weights_and_q",
    "_pack_upper_tri",
    "_flat_lmm",
]
