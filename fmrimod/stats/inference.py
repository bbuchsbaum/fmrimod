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
