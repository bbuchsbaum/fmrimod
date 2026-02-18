"""Statistical inference utilities: p-values, z-scores, FDR."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


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
        return sp_stats.norm.isf(p / 2.0)
    return sp_stats.norm.isf(p)


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
        return 2.0 * sp_stats.norm.sf(np.abs(z))
    return sp_stats.norm.sf(z)


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
    return (np.exp(2.0 * z_arr) - 1.0) / (np.exp(2.0 * z_arr) + 1.0)


def fdr_correction(
    p_values: NDArray[np.float64],
    alpha: float = 0.05,
    method: str = "bh",
) -> tuple:
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
        c_n = np.sum(1.0 / ranks)
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
