"""Effective degrees of freedom calculation.

Adjusts residual degrees of freedom to account for preprocessing
steps that consume additional degrees of freedom (soft projection,
already-applied censoring). AR whitening is *not* subtracted here:
the live GLM path reports ``n - rank`` (fmrireg 0.2.0).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def effective_df(
    n: int,
    rank: int,
    ar_order: int = 0,
    n_censored: int = 0,
    soft_subspace_rank: int = 0,
) -> float:
    """Compute effective residual degrees of freedom.

    Parameters
    ----------
    n : int
        Number of observations (after censoring).
    rank : int
        Rank of the design matrix.
    ar_order : int
        Accepted for compatibility. Not subtracted: AR whitening does
        not deflate residual df on the live path.
    n_censored : int
        Number of censored volumes already removed from *n*. Not
        subtracted again.
    soft_subspace_rank : int
        Rank of the soft subspace projection (reduces df).

    Returns
    -------
    float
        Effective residual degrees of freedom (``n - rank -
        soft_subspace_rank``, floored at 1).
    """
    del ar_order, n_censored
    df = float(n - rank - soft_subspace_rank)
    return max(df, 1.0)


def satterthwaite_df(
    XtXinv: NDArray[np.float64],
    con_vec: NDArray[np.float64],
    run_dfres: list[float],
    run_XtXinv: list[NDArray[np.float64]],
) -> float:
    """Satterthwaite approximation for effective df in multi-run contrasts.

    Used when runs have different residual variances and the pooled
    df should account for heteroscedasticity.

    Parameters
    ----------
    XtXinv : NDArray
        Pooled ``(X'X)^{-1}``, shape ``(p, p)``.
    con_vec : NDArray
        Contrast vector, shape ``(p,)``.
    run_dfres : list of float
        Residual df per run.
    run_XtXinv : list of NDArray
        ``(X'X)^{-1}`` per run.

    Returns
    -------
    float
        Satterthwaite-adjusted degrees of freedom.
    """
    con_vec = np.asarray(con_vec, dtype=np.float64).ravel()

    # Variance contribution from each run
    v_parts = []
    for _dfr, xtxi in zip(run_dfres, run_XtXinv, strict=False):
        v_r = con_vec @ xtxi @ con_vec
        v_parts.append(v_r)

    v_total = float(sum(v_parts))
    if v_total < 1e-15:
        return float(sum(run_dfres))

    # Satterthwaite: df = v_total^2 / sum(v_r^2 / df_r)
    denom = float(
        sum(
            v_r**2 / dfr
            for v_r, dfr in zip(v_parts, run_dfres, strict=False)
            if dfr > 0
        )
    )
    if denom < 1e-15:
        return float(sum(run_dfres))

    return v_total**2 / denom
