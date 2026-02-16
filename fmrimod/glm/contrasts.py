"""Contrast computation on GLM results.

Computes t-statistics, F-statistics, standard errors, and p-values
for linear contrasts of regression coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


@dataclass
class ContrastResult:
    """Result of evaluating a contrast on GLM output.

    Attributes
    ----------
    name : str
        Contrast name.
    estimate : NDArray
        Contrast estimate ``c' @ B``, shape ``(k, V)`` for F-tests
        or ``(V,)`` for t-tests.
    stat : NDArray
        Test statistic (t or F), shape ``(V,)``.
    se : NDArray or None
        Standard error, shape ``(V,)`` (t-tests only).
    p_value : NDArray
        Two-sided p-values, shape ``(V,)``.
    df : float or tuple
        Degrees of freedom.  Scalar for t-tests, ``(df1, df2)`` for F-tests.
    stat_type : str
        ``"t"`` or ``"F"``.
    """

    name: str
    estimate: NDArray[np.float64]
    stat: NDArray[np.float64]
    se: Optional[NDArray[np.float64]]
    p_value: NDArray[np.float64]
    df: Union[float, tuple]
    stat_type: str


def contrast_t(
    con_vec: NDArray[np.float64],
    betas: NDArray[np.float64],
    XtXinv: NDArray[np.float64],
    sigma: NDArray[np.float64],
    dfres: float,
    name: str = "t-contrast",
) -> ContrastResult:
    """Compute a t-contrast.

    Parameters
    ----------
    con_vec : NDArray
        Contrast vector, shape ``(p,)``.
    betas : NDArray
        Coefficient matrix, shape ``(p, V)``.
    XtXinv : NDArray
        ``(X'X)^{-1}``, shape ``(p, p)``.
    sigma : NDArray
        Residual standard deviation, shape ``(V,)``.
    dfres : float
        Residual degrees of freedom.
    name : str
        Contrast name.

    Returns
    -------
    ContrastResult
    """
    con_vec = np.asarray(con_vec, dtype=np.float64).ravel()
    p = betas.shape[0]
    if len(con_vec) != p:
        raise ValueError(f"Contrast vector length {len(con_vec)} != {p} coefficients")

    # Estimate: c' B  →  (V,)
    estimate = con_vec @ betas

    # Variance of contrast: c' (X'X)^{-1} c * sigma^2
    var_factor = con_vec @ XtXinv @ con_vec  # scalar
    se = sigma * np.sqrt(np.maximum(var_factor, 0.0))

    # t-statistic
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 1e-15, estimate / se, 0.0)

    # Two-sided p-value
    p_value = 2.0 * sp_stats.t.sf(np.abs(tstat), dfres)

    return ContrastResult(
        name=name,
        estimate=estimate,
        stat=tstat,
        se=se,
        p_value=p_value,
        df=dfres,
        stat_type="t",
    )


def contrast_f(
    con_mat: NDArray[np.float64],
    betas: NDArray[np.float64],
    XtXinv: NDArray[np.float64],
    sigma: NDArray[np.float64],
    dfres: float,
    name: str = "F-contrast",
) -> ContrastResult:
    """Compute an F-contrast.

    Parameters
    ----------
    con_mat : NDArray
        Contrast matrix, shape ``(k, p)`` where ``k`` is the number
        of linear constraints.
    betas : NDArray
        Coefficient matrix, shape ``(p, V)``.
    XtXinv : NDArray
        ``(X'X)^{-1}``, shape ``(p, p)``.
    sigma : NDArray
        Residual standard deviation, shape ``(V,)``.
    dfres : float
        Residual degrees of freedom.
    name : str
        Contrast name.

    Returns
    -------
    ContrastResult
    """
    con_mat = np.atleast_2d(np.asarray(con_mat, dtype=np.float64))
    k = con_mat.shape[0]

    # C @ B  →  (k, V)
    CB = con_mat @ betas

    # (C (X'X)^{-1} C')^{-1}  →  (k, k)
    CXtXinvC = con_mat @ XtXinv @ con_mat.T
    try:
        CXtXinvC_inv = np.linalg.inv(CXtXinvC)
    except np.linalg.LinAlgError:
        CXtXinvC_inv = np.linalg.pinv(CXtXinvC)

    # F = (CB)' (C (X'X)^{-1} C')^{-1} (CB) / (k * sigma^2)
    # For each voxel v:
    #   F_v = CB[:,v]' @ CXtXinvC_inv @ CB[:,v] / (k * sigma_v^2)
    sigma2 = sigma ** 2
    V = betas.shape[1]
    fstat = np.zeros(V)

    for v in range(V):
        cb_v = CB[:, v]
        if sigma2[v] > 1e-15:
            fstat[v] = (cb_v @ CXtXinvC_inv @ cb_v) / (k * sigma2[v])
        else:
            fstat[v] = 0.0

    p_value = sp_stats.f.sf(fstat, k, dfres)

    return ContrastResult(
        name=name,
        estimate=CB,
        stat=fstat,
        se=None,
        p_value=p_value,
        df=(float(k), dfres),
        stat_type="F",
    )


def contrast_f_vectorized(
    con_mat: NDArray[np.float64],
    betas: NDArray[np.float64],
    XtXinv: NDArray[np.float64],
    sigma: NDArray[np.float64],
    dfres: float,
    name: str = "F-contrast",
) -> ContrastResult:
    """Vectorised F-contrast (faster for large V).

    Same interface as :func:`contrast_f` but avoids Python loops.
    """
    con_mat = np.atleast_2d(np.asarray(con_mat, dtype=np.float64))
    k = con_mat.shape[0]

    CB = con_mat @ betas  # (k, V)

    CXtXinvC = con_mat @ XtXinv @ con_mat.T
    try:
        L = np.linalg.cholesky(CXtXinvC)
        # Solve L @ Z = CB for each voxel
        Z = np.linalg.solve(L, CB)  # (k, V)
    except np.linalg.LinAlgError:
        CXtXinvC_inv = np.linalg.pinv(CXtXinvC)
        Z = CXtXinvC_inv @ CB
        # fstat = sum(CB * Z, axis=0) / (k * sigma^2)
        sigma2 = sigma ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            fstat = np.where(
                sigma2 > 1e-15,
                np.sum(CB * Z, axis=0) / (k * sigma2),
                0.0,
            )
        p_value = sp_stats.f.sf(fstat, k, dfres)
        return ContrastResult(
            name=name, estimate=CB, stat=fstat, se=None,
            p_value=p_value, df=(float(k), dfres), stat_type="F",
        )

    # F = ||Z||^2 / (k * sigma^2)
    sigma2 = sigma ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        fstat = np.where(
            sigma2 > 1e-15,
            np.sum(Z ** 2, axis=0) / (k * sigma2),
            0.0,
        )

    p_value = sp_stats.f.sf(fstat, k, dfres)

    return ContrastResult(
        name=name, estimate=CB, stat=fstat, se=None,
        p_value=p_value, df=(float(k), dfres), stat_type="F",
    )
