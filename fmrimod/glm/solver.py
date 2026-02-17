"""Core OLS/WLS solver for fMRI GLM fitting.

Ports R's ``.fast_preproject()`` and ``solve_glm_core()`` to numpy/scipy.
All operations work on ``(time, voxels)`` matrices for vectorised computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


@dataclass
class Projection:
    """Pre-computed projection components for fast least squares.

    Attributes
    ----------
    Pinv : NDArray
        Pseudoinverse-like matrix, shape ``(p, n)``, such that
        ``betas = Pinv @ Y``.
    XtXinv : NDArray
        ``(X'X)^{-1}`` matrix, shape ``(p, p)``, for variance computation.
    dfres : float
        Residual degrees of freedom (``n - rank``).
    rank : int
        Numerical rank of the design matrix.
    is_full_rank : bool
        Whether ``rank == p``.
    """

    Pinv: NDArray[np.float64]
    XtXinv: NDArray[np.float64]
    dfres: float
    rank: int
    is_full_rank: bool


def fast_preproject(X: NDArray[np.float64]) -> Projection:
    """Pre-compute projection matrices for fast OLS.

    This computes the components needed to solve ``Y = X @ B + E``
    efficiently for many columns of ``Y`` simultaneously.

    For full-rank X, uses Cholesky decomposition of X'X.
    For rank-deficient X, falls back to SVD-based pseudoinverse.

    Parameters
    ----------
    X : NDArray
        Design matrix, shape ``(n, p)``.

    Returns
    -------
    Projection
        Pre-computed projection components.

    Raises
    ------
    ValueError
        If *X* contains NaN or Inf values.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Design matrix must have at least one row and one column")
    if not np.all(np.isfinite(X)):
        raise ValueError("Design matrix contains NA/Inf values")

    n, p = X.shape

    # QR decomposition for rank detection
    Q, R, pivot = linalg.qr(X, pivoting=True, mode="economic")
    # Determine rank from R diagonal
    diag_R = np.abs(np.diag(R))
    tol = max(n, p) * np.finfo(np.float64).eps * (diag_R[0] if len(diag_R) > 0 else 1.0)
    rank = int(np.sum(diag_R > tol))

    cond_threshold = 1.0 / np.sqrt(np.finfo(np.float64).eps)
    use_svd = rank != p
    if rank == p:
        try:
            cond_est = np.linalg.cond(R[:p, :p]) if p > 0 else 1.0
        except np.linalg.LinAlgError:
            cond_est = np.inf
        if not np.isfinite(cond_est) or cond_est > cond_threshold:
            use_svd = True

    if use_svd:
        # Rank-deficient or ill-conditioned: SVD-based pseudoinverse
        U, s, Vt = linalg.svd(X, full_matrices=False)
        tol_svd = max(n, p) * np.finfo(np.float64).eps * s[0]
        pos = s > tol_svd
        rank = int(np.sum(pos))
        s_inv = np.zeros_like(s)
        s_inv[pos] = 1.0 / s[pos]

        # Pinv = V @ diag(1/s) @ U'
        Pinv = (Vt.T * s_inv[np.newaxis, :]) @ U.T
        # XtXinv = V @ diag(1/s^2) @ V'
        XtXinv = (Vt.T * (s_inv ** 2)[np.newaxis, :]) @ Vt
    else:
        # Well-conditioned full rank: Cholesky of X'X for efficiency
        XtX = X.T @ X
        L = linalg.cholesky(XtX, lower=True)
        XtXinv = linalg.cho_solve((L, True), np.eye(p))
        Pinv = XtXinv @ X.T

    return Projection(
        Pinv=Pinv,
        XtXinv=XtXinv,
        dfres=float(n - rank),
        rank=rank,
        is_full_rank=(rank == p),
    )


@dataclass
class LmResult:
    """Result of a single least-squares fit.

    Attributes
    ----------
    betas : NDArray
        Coefficient matrix, shape ``(p, V)``.
    rss : NDArray
        Residual sum of squares, shape ``(V,)``.
    sigma2 : NDArray
        Residual variance per voxel, shape ``(V,)``.
    dfres : float
        Residual degrees of freedom.
    rank : int
        Rank of the design matrix.
    fitted : NDArray or None
        Fitted values ``X @ B``, shape ``(n, V)``.  Only present when
        ``return_fitted=True``.
    """

    betas: NDArray[np.float64]
    rss: NDArray[np.float64]
    sigma2: NDArray[np.float64]
    dfres: float
    rank: int
    fitted: Optional[NDArray[np.float64]] = None


def fast_lm_matrix(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    proj: Projection,
    return_fitted: bool = False,
) -> LmResult:
    """Fast matrix-based OLS fit.

    Solves ``Y = X @ B + E`` using pre-computed projection from
    :func:`fast_preproject`, returning coefficients, RSS, and sigma^2
    for all voxels simultaneously.

    Parameters
    ----------
    X : NDArray
        Design matrix, shape ``(n, p)``.
    Y : NDArray
        Data matrix, shape ``(n, V)`` where ``V`` is the number of
        voxels.
    proj : Projection
        Pre-computed projection from :func:`fast_preproject`.
    return_fitted : bool
        If ``True``, also return fitted values ``X @ B``.

    Returns
    -------
    LmResult
        Regression results.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    elif Y.ndim != 2:
        raise ValueError(f"Y must be 1-D or 2-D, got shape {Y.shape}")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Response matrix contains NA/Inf values")

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y dimensions do not match")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Response matrix contains NA/Inf values")

    if proj.Pinv.ndim != 2:
        raise ValueError("Projection Pinv must be 2-D")
    if proj.Pinv.shape != (X.shape[1], X.shape[0]):
        raise ValueError("X and projection dimensions do not match")

    # B = Pinv @ Y   →  (p, V)
    betas = proj.Pinv @ Y

    if return_fitted:
        fitted = X @ betas
        residuals = Y - fitted
        rss = np.sum(residuals ** 2, axis=0)
    else:
        # Memory-efficient: compute RSS without materialising residuals
        # RSS = Y'Y - B' X'Y
        XtY = X.T @ Y
        yTy = np.sum(Y * Y, axis=0)
        rss = yTy - np.sum(betas * XtY, axis=0)
        rss = np.maximum(rss, 0.0)
        fitted = None

    sigma2 = rss / proj.dfres if proj.dfres > 0 else np.full_like(rss, np.nan)

    return LmResult(
        betas=betas,
        rss=rss,
        sigma2=sigma2,
        dfres=proj.dfres,
        rank=proj.rank,
        fitted=fitted,
    )
