"""Statistical inference utilities: p-values, z-scores, FDR."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Optional, Sequence, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as sp_stats

from ..dataset.group_data import GroupData
from .backends import resolve_second_level_backend
from .interfaces import GroupFitRequest, GroupFitResult
from .meta import _coerce_group_data
from .normalize import normalize_group_fit_request
from .spatial_fdr import spatial_fdr


def p_to_z(p: NDArray[np.float64], two_sided: bool = True) -> NDArray[np.float64]:
    """Convert p-values to z-scores.

    Parameters
    ----------
    p : NDArray
        P-values.
    two_sided : bool
        If ``True``, use two-sided conversion.

    Returns
    -------
    NDArray
        Z-scores.
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-300, 1.0 - 1e-15)
    if two_sided:
        return cast("NDArray[np.float64]", sp_stats.norm.isf(p / 2.0))
    return cast("NDArray[np.float64]", sp_stats.norm.isf(p))


def z_to_p(z: NDArray[np.float64], two_sided: bool = True) -> NDArray[np.float64]:
    """Convert z-scores to p-values.

    Parameters
    ----------
    z : NDArray
        Z-scores.
    two_sided : bool
        If ``True``, compute two-sided p-values.

    Returns
    -------
    NDArray
        P-values.
    """
    z = np.asarray(z, dtype=np.float64)
    if two_sided:
        return cast("NDArray[np.float64]", 2.0 * sp_stats.norm.sf(np.abs(z)))
    return cast("NDArray[np.float64]", sp_stats.norm.sf(z))


def t_to_d(
    t: NDArray[np.float64] | float,
    df: NDArray[np.float64] | float,
    n: NDArray[np.float64] | float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert t-statistics to standardized effect sizes and variances.

    Mirrors ``fmrireg::t_to_d``:
    - if ``n`` is ``None``: one-sample/paired formula
    - else: two-sample formula
    """
    t_arr = np.asarray(t, dtype=np.float64)
    df_arr = np.asarray(df, dtype=np.float64)
    if np.any(df_arr <= 0):
        raise ValueError("df must be > 0")

    if n is None:
        n_arr = df_arr + 1.0
        d = t_arr * np.sqrt(1.0 / n_arr)
        v = 1.0 / n_arr + (d ** 2) / (2.0 * n_arr)
    else:
        n_arr = np.asarray(n, dtype=np.float64)
        if np.any(n_arr <= 0):
            raise ValueError("n must be > 0")
        d = 2.0 * t_arr / np.sqrt(df_arr)
        v = 4.0 / n_arr + (d ** 2) / (2.0 * df_arr)

    return np.asarray(d, dtype=np.float64), np.asarray(v, dtype=np.float64)


def r_to_z(
    r: NDArray[np.float64] | float,
    n: NDArray[np.float64] | float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert correlations to Fisher Z and sampling variance."""
    r_arr = np.asarray(r, dtype=np.float64)
    n_arr = np.asarray(n, dtype=np.float64)
    if np.any(np.abs(r_arr) >= 1):
        raise ValueError("r must be strictly between -1 and 1")
    if np.any(n_arr <= 3):
        raise ValueError("n must be > 3 for Fisher Z variance")

    z = 0.5 * np.log((1.0 + r_arr) / (1.0 - r_arr))
    v = 1.0 / (n_arr - 3.0)
    return np.asarray(z, dtype=np.float64), np.asarray(v, dtype=np.float64)


def z_to_r(z: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Back-transform Fisher Z to correlation."""
    z_arr = np.asarray(z, dtype=np.float64)
    return cast(
        "NDArray[np.float64]",
        (np.exp(2.0 * z_arr) - 1.0) / (np.exp(2.0 * z_arr) + 1.0),
    )


def fdr_correction(
    p_values: NDArray[np.float64],
    alpha: float = 0.05,
    method: str = "bh",
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : NDArray
        Raw p-values, shape ``(V,)``.
    alpha : float
        Desired FDR level.
    method : str
        ``"bh"`` for Benjamini-Hochberg, ``"by"`` for
        Benjamini-Yekutieli.

    Returns
    -------
    reject : NDArray[bool]
        Boolean mask of rejected hypotheses.
    p_adjusted : NDArray
        Adjusted p-values.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float64)

    # Sort
    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]
    ranks = np.arange(1, n + 1, dtype=np.float64)

    if method == "bh":
        # BH: p_adj = min(p * n / rank, 1)
        adjusted = sorted_p * n / ranks
    elif method == "by":
        # BY: p_adj = min(p * n * c_n / rank, 1) where c_n = sum(1/k)
        c_n: float = float(np.sum(1.0 / ranks))
        adjusted = sorted_p * n * c_n / ranks
    else:
        raise ValueError(f"Unknown method: {method}")

    # Enforce monotonicity (from right to left)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Unsort
    p_adjusted = np.empty(n)
    p_adjusted[sorted_idx] = adjusted

    reject = p_adjusted <= alpha

    return reject, p_adjusted


def _as_2d(x: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    if arr.ndim == 2:
        return arr
    raise ValueError("Expected 1-D or 2-D p-value array")


def _coerce_group_ids(group_ids: object, n_features: int) -> NDArray[np.intp]:
    if group_ids is None:
        raise ValueError("correction='spatial' requires group_ids")
    arr = np.asarray(group_ids)
    if arr.ndim != 1:
        raise ValueError("group_ids must be 1-D")
    if arr.shape[0] != n_features:
        raise ValueError("group_ids length must match number of features")
    if arr.dtype.kind in ("f", "c"):
        if not np.all(np.isfinite(arr)):
            raise ValueError("group_ids must contain finite integer labels")
        if not np.allclose(arr, np.round(arr)):
            raise ValueError("group_ids must contain integer labels")
    elif arr.dtype.kind not in ("i", "u", "b"):
        try:
            arr = arr.astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError("group_ids must contain integer labels") from exc
    return arr.astype(np.intp, copy=False)


def _apply_group_correction(
    result: GroupFitResult,
    request: GroupFitRequest,
) -> GroupFitResult:
    if request.correction is None:
        return result

    p2 = _as_2d(result.p)
    q2 = np.empty_like(p2)
    corr = str(request.correction)

    if corr in ("bh", "by"):
        for j in range(p2.shape[1]):
            _, q_col = fdr_correction(p2[:, j], alpha=request.alpha, method=corr)
            q2[:, j] = q_col
    elif corr == "spatial":
        gids = _coerce_group_ids(request.group_ids, n_features=p2.shape[0])
        for j in range(p2.shape[1]):
            q2[:, j] = spatial_fdr(p2[:, j], group_ids=gids, alpha=request.alpha).qvalues
    else:
        raise ValueError("correction must be one of: bh, by, spatial")

    md = dict(result.metadata)
    md["correction"] = corr
    md["alpha"] = float(request.alpha)
    return replace(result, q=q2, metadata=md)


def group_fit(
    request: "GroupFitRequest | GroupData | pd.DataFrame",
    *,
    effect_cols: Optional[Mapping[str, str] | Sequence[str]] = None,
) -> GroupFitResult:
    """Canonical second-level interface with parity-oriented normalization.

    ``request`` may be a :class:`GroupFitRequest`, or -- for the common
    default fit -- a :class:`GroupData` or a pandas DataFrame (validated
    to a frozen ``GroupData`` via the typed ``group_data_from_csv``;
    pass its ``effect_cols`` schema). ``GroupFitRequest`` stays the way
    to customize a non-default fit (formula/model/backend/...); it is
    no longer the *only* way in.
    """
    if not isinstance(request, (GroupFitRequest, GroupData, pd.DataFrame)):
        raise TypeError(
            "'request' must be a GroupFitRequest, a GroupData, or a "
            "pandas DataFrame"
        )
    if isinstance(request, GroupFitRequest):
        if effect_cols is not None:
            raise ValueError(
                "effect_cols only applies when 'request' is a GroupData "
                "or DataFrame; a GroupFitRequest already carries its schema"
            )
        request_obj = request
    else:
        request_obj = GroupFitRequest(
            data=_coerce_group_data(request, effect_cols)
        )

    req = normalize_group_fit_request(request_obj)
    backend = resolve_second_level_backend(str(req.backend))
    out = backend.fit(req)
    return _apply_group_correction(out, req)
