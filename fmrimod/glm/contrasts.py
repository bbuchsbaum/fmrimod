"""Contrast computation on GLM results.

Computes t-statistics, F-statistics, standard errors, and p-values
for linear contrasts of regression coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import special as sp_special


def _validate_f_contrast_matrix(
    con_mat: NDArray[np.float64], n_coefficients: int
) -> NDArray[np.float64]:
    """Validate and normalize an F-contrast matrix."""
    con_mat = np.atleast_2d(np.asarray(con_mat, dtype=np.float64))
    if con_mat.shape[0] == 0:
        raise ValueError("F-contrast matrix must have at least one contrast row")
    if con_mat.shape[1] != n_coefficients:
        raise ValueError(
            f"F-contrast matrix has {con_mat.shape[1]} columns but model has "
            f"{n_coefficients} coefficients"
        )
    return con_mat


def _f_quadratic_form_terms(
    con_mat: NDArray[np.float64],
    betas: NDArray[np.float64],
    XtXinv: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Compute stable F-test quadratic-form numerators."""
    CB = con_mat @ betas  # (k, V)
    cov = con_mat @ XtXinv @ con_mat.T  # (k, k)

    df1 = int(np.linalg.matrix_rank(cov))
    if df1 <= 0:
        return CB, np.zeros(betas.shape[1], dtype=np.float64), 1

    # Use pseudoinverse for rank-deficient or highly ill-conditioned covariance
    # to keep loop and vectorized paths numerically aligned.
    cond = np.linalg.cond(cov)
    use_pinv = (
        df1 < cov.shape[0]
        or not np.isfinite(cond)
        or cond > 1.0 / np.sqrt(np.finfo(np.float64).eps)
    )

    if use_pinv:
        warnings.warn(
            "F-contrast covariance is singular; using pseudoinverse fallback",
            RuntimeWarning,
            stacklevel=3,
        )
        cov_inv = np.linalg.pinv(cov)
        numer = np.sum(CB * (cov_inv @ CB), axis=0)
    else:
        L = np.linalg.cholesky(cov)
        Z = np.linalg.solve(L, CB)
        numer = np.sum(Z ** 2, axis=0)

    return CB, np.maximum(numer, 0.0), df1


def _t_two_sided_pvalue(
    tstat: NDArray[np.float64],
    dfres: float,
) -> NDArray[np.float64]:
    """Compute two-sided p-values for t-statistics."""
    return 2.0 * sp_special.stdtr(dfres, -np.abs(tstat))


def _f_upper_tail_pvalue(
    fstat: NDArray[np.float64],
    df1: float,
    df2: float,
) -> NDArray[np.float64]:
    """Compute upper-tail p-values for F-statistics."""
    return sp_special.fdtrc(df1, df2, fstat)


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
    p_value = _t_two_sided_pvalue(tstat, dfres)

    return ContrastResult(
        name=name,
        estimate=estimate,
        stat=tstat,
        se=se,
        p_value=p_value,
        df=dfres,
        stat_type="t",
    )


def contrast_t_batch(
    con_mat: NDArray[np.float64],
    betas: NDArray[np.float64],
    XtXinv: NDArray[np.float64],
    sigma: NDArray[np.float64],
    dfres: float,
    names: Optional[list[str]] = None,
    name_prefix: str = "t-contrast",
) -> list[ContrastResult]:
    """Compute multiple t-contrasts in one vectorized pass.

    Parameters
    ----------
    con_mat : NDArray
        Contrast matrix, shape ``(k, p)``.
    betas : NDArray
        Coefficient matrix, shape ``(p, V)``.
    XtXinv : NDArray
        ``(X'X)^{-1}``, shape ``(p, p)``.
    sigma : NDArray
        Residual standard deviation, shape ``(V,)``.
    dfres : float
        Residual degrees of freedom.
    names : list[str], optional
        Names for each contrast row.
    name_prefix : str
        Prefix used when ``names`` is not provided.

    Returns
    -------
    list[ContrastResult]
        One result per contrast row.
    """
    con_mat = np.atleast_2d(np.asarray(con_mat, dtype=np.float64))
    if con_mat.shape[0] == 0:
        return []
    p = betas.shape[0]
    if con_mat.shape[1] != p:
        raise ValueError(
            f"Contrast matrix has {con_mat.shape[1]} columns but model has "
            f"{p} coefficients"
        )
    if names is not None and len(names) != con_mat.shape[0]:
        raise ValueError("names length must match number of contrast rows")

    # Estimates: C B -> (k, V)
    estimates = con_mat @ betas

    # Per-contrast variance factors: diag(C XtXinv C')
    c_xtx = con_mat @ XtXinv
    var_factors = np.sum(c_xtx * con_mat, axis=1)
    se = np.sqrt(np.maximum(var_factors, 0.0))[:, np.newaxis] * sigma[np.newaxis, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 1e-15, estimates / se, 0.0)
    p_value = _t_two_sided_pvalue(tstat, dfres)

    if names is None:
        names = [f"{name_prefix}[{i}]" for i in range(con_mat.shape[0])]

    return [
        ContrastResult(
            name=names[i],
            estimate=estimates[i],
            stat=tstat[i],
            se=se[i],
            p_value=p_value[i],
            df=dfres,
            stat_type="t",
        )
        for i in range(con_mat.shape[0])
    ]


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
    con_mat = _validate_f_contrast_matrix(con_mat, betas.shape[0])

    # Fast-path: one-row F-contrast is equivalent to squared t-statistic.
    if con_mat.shape[0] == 1:
        con_vec = con_mat[0]
        estimate = con_vec @ betas  # (V,)
        var_factor = con_vec @ XtXinv @ con_vec
        se = sigma * np.sqrt(np.maximum(var_factor, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = np.where(se > 1e-15, estimate / se, 0.0)
        fstat = np.maximum(tstat * tstat, 0.0)
        p_value = _t_two_sided_pvalue(tstat, dfres)
        return ContrastResult(
            name=name,
            estimate=estimate[np.newaxis, :],
            stat=fstat,
            se=None,
            p_value=p_value,
            df=(1.0, dfres),
            stat_type="F",
        )

    CB, numer, df1 = _f_quadratic_form_terms(con_mat, betas, XtXinv)
    sigma2 = np.square(sigma, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        fstat = np.where(
            sigma2 > 1e-15,
            numer / (df1 * sigma2),
            0.0,
        )

    p_value = _f_upper_tail_pvalue(fstat, float(df1), float(dfres))

    return ContrastResult(
        name=name,
        estimate=CB,
        stat=fstat,
        se=None,
        p_value=p_value,
        df=(float(df1), dfres),
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
    """Compatibility wrapper for F-contrasts.

    Delegates to :func:`contrast_f` to keep numerics identical across
    near-singular and rank-deficient paths.
    """
    return contrast_f(con_mat, betas, XtXinv, sigma, dfres, name=name)
