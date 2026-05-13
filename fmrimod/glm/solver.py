"""Core OLS/WLS solver for fMRI GLM fitting.

Ports R's ``.fast_preproject()`` and ``solve_glm_core()`` to numpy/scipy.
All operations work on ``(time, voxels)`` matrices for vectorised computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def _rss_needs_direct_residual(
    rss: NDArray[np.float64],
    yTy: NDArray[np.float64],
    fitted_ss: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Flag voxels where sufficient-stat RSS is dominated by cancellation."""
    scale = np.maximum(yTy, np.abs(fitted_ss))
    threshold = np.sqrt(np.finfo(np.float64).eps) * np.maximum(scale, 1.0)
    return rss <= threshold


def _resolve_compute_dtype(dtype: object) -> np.dtype:
    """Normalize and validate solver compute dtype."""
    dt = np.dtype(dtype)
    if dt.kind != "f":
        raise ValueError(f"compute dtype must be floating-point, got {dt}")
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(
            f"unsupported compute dtype {dt}; expected float32 or float64"
        )
    return dt


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
    ill_conditioned : bool
        Whether the design was routed through the SVD path due rank
        deficiency or poor conditioning.
    XtX : NDArray or None
        Optional cached Gram matrix ``X'X`` used by fast RSS evaluation.
        Present for stable full-rank projections.
    """

    Pinv: NDArray[np.float64]
    XtXinv: NDArray[np.float64]
    dfres: float
    rank: int
    is_full_rank: bool
    ill_conditioned: bool
    XtX: Optional[NDArray[np.float64]] = None
    aliased_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class RunConditionReport:
    """Per-run rank/conditioning summary.

    Attributes
    ----------
    run
        Zero-based run index.
    n_columns
        Number of design-matrix columns (``p``) for this run.
    rank
        Numerical rank of the design matrix.
    is_full_rank
        Whether ``rank == n_columns``.
    ill_conditioned
        Whether the solver routed through the SVD pseudoinverse path
        because of rank deficiency or poor conditioning.
    dfres
        Residual degrees of freedom (``n - rank``).
    aliased_columns
        Best-effort tuple of column names that are linearly dependent on
        earlier columns and therefore not individually identifiable. Empty
        for full-rank designs.
    """

    run: int
    n_columns: int
    rank: int
    is_full_rank: bool
    ill_conditioned: bool
    dfres: float
    aliased_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConditionReport:
    """Rank/conditioning summary aggregated over all runs of a fit."""

    runs: tuple[RunConditionReport, ...]

    @property
    def is_full_rank(self) -> bool:
        return all(run.is_full_rank for run in self.runs)

    @property
    def ill_conditioned(self) -> bool:
        return any(run.ill_conditioned for run in self.runs)

    @property
    def aliased_columns(self) -> tuple[str, ...]:
        seen: list[str] = []
        for run in self.runs:
            for name in run.aliased_columns:
                if name not in seen:
                    seen.append(name)
        return tuple(seen)


def fast_preproject(
    X: NDArray[np.float64],
    compute_dtype: object = np.float64,
    check_finite: bool = True,
    method: Literal["auto", "pinv"] = "auto",
) -> Projection:
    """Pre-compute projection matrices for fast OLS.

    This computes the components needed to solve ``Y = X @ B + E``
    efficiently for many columns of ``Y`` simultaneously.

    For full-rank X, uses Cholesky decomposition of X'X.
    For rank-deficient X, falls back to SVD-based pseudoinverse.

    Parameters
    ----------
    X : NDArray
        Design matrix, shape ``(n, p)``.
    compute_dtype : numpy dtype-like
        Internal compute dtype (``float64`` default, optional ``float32``).
    method : {"auto", "pinv"}
        Projection backend. ``"auto"`` uses the fast stable path and falls
        back for rank-deficient or ill-conditioned designs. ``"pinv"`` always
        uses a Moore-Penrose projection.

    Returns
    -------
    Projection
        Pre-computed projection components.

    Raises
    ------
    ValueError
        If *X* contains NaN or Inf values.
    """
    dtype = _resolve_compute_dtype(compute_dtype)
    eps = np.finfo(dtype).eps

    X = np.asarray(X, dtype=dtype)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Design matrix must have at least one row and one column")
    if method not in ("auto", "pinv"):
        raise ValueError("projection method must be 'auto' or 'pinv'")
    if check_finite and not np.all(np.isfinite(X)):
        raise ValueError("Design matrix contains NA/Inf values")

    n, p = X.shape

    if method == "pinv":
        Pinv, rank = linalg.pinv(X, return_rank=True, check_finite=False)
        XtXinv = Pinv @ Pinv.T
        return Projection(
            Pinv=Pinv,
            XtXinv=XtXinv,
            dfres=float(n - rank),
            rank=int(rank),
            is_full_rank=(int(rank) == p),
            ill_conditioned=(int(rank) != p),
            XtX=None,
        )

    cond_threshold = 1.0 / np.sqrt(eps)
    used_cholesky = False
    ill_conditioned = False
    rank = p
    XtX_cached: Optional[NDArray[np.float64]] = None

    # Fast path: attempt full-rank Cholesky first.
    try:
        XtX = X.T @ X
        L = linalg.cholesky(XtX, lower=True, check_finite=False)
        cond_xtx = np.linalg.cond(XtX) if p > 0 else 1.0
        cond_est = np.sqrt(cond_xtx) if np.isfinite(cond_xtx) else np.inf
        if np.isfinite(cond_est) and cond_est <= cond_threshold:
            XtXinv = linalg.cho_solve(
                (L, True),
                np.eye(p, dtype=dtype),
                check_finite=False,
            )
            Pinv = XtXinv @ X.T
            used_cholesky = True
            XtX_cached = XtX
    except np.linalg.LinAlgError:
        used_cholesky = False

    aliased_indices: tuple[int, ...] = ()
    if not used_cholesky:
        # Fallback: rank-revealing QR + SVD when needed. The QR pivot tells
        # us which columns the decomposition treats as redundant — used
        # below to surface aliased-column names in the rank diagnostic.
        _Q, R, pivot = linalg.qr(X, pivoting=True, mode="economic", check_finite=False)

        diag_R = np.abs(np.diag(R))
        tol = max(n, p) * eps * (
            diag_R[0] if len(diag_R) > 0 else dtype.type(1.0)
        )
        rank = int(np.sum(diag_R > tol))
        if rank < p:
            aliased_indices = tuple(int(idx) for idx in pivot[rank:])

        use_svd = rank != p
        if rank == p:
            try:
                cond_est = np.linalg.cond(R[:p, :p]) if p > 0 else 1.0
            except np.linalg.LinAlgError:
                cond_est = np.inf
            if not np.isfinite(cond_est) or cond_est > cond_threshold:
                use_svd = True

        if use_svd:
            ill_conditioned = True
            U, s, Vt = linalg.svd(X, full_matrices=False, check_finite=False)
            tol_svd = max(n, p) * eps * s[0]
            pos = s > tol_svd
            rank = int(np.sum(pos))
            s_inv = np.zeros_like(s)
            s_inv[pos] = 1.0 / s[pos]

            # Pinv = V @ diag(1/s) @ U'
            Pinv = (Vt.T * s_inv[np.newaxis, :]) @ U.T
            # XtXinv = V @ diag(1/s^2) @ V'
            XtXinv = (Vt.T * (s_inv ** 2)[np.newaxis, :]) @ Vt
        else:
            ill_conditioned = False
            XtX = X.T @ X
            L = linalg.cholesky(XtX, lower=True, check_finite=False)
            XtXinv = linalg.cho_solve(
                (L, True),
                np.eye(p, dtype=dtype),
                check_finite=False,
            )
            Pinv = XtXinv @ X.T
            XtX_cached = XtX

    return Projection(
        Pinv=Pinv,
        XtXinv=XtXinv,
        # dfres = n - rank, not n - p. This is the textbook residual-DoF
        # choice for a rank-deficient design: only `rank` linearly
        # independent columns subtract a DoF from the residual. Nilearn's
        # run_glm uses n - p regardless of rank, which silently inflates
        # t/F denominators when X is rank-deficient; we intentionally do
        # not match that.
        dfres=float(n - rank),
        rank=rank,
        is_full_rank=(rank == p),
        ill_conditioned=ill_conditioned,
        XtX=XtX_cached,
        aliased_indices=aliased_indices,
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
    compute_dtype: object = np.float64,
    check_finite: bool = True,
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
    dtype = _resolve_compute_dtype(compute_dtype)
    X = np.asarray(X, dtype=dtype)
    Y = np.asarray(Y, dtype=dtype)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    elif Y.ndim != 2:
        raise ValueError(f"Y must be 1-D or 2-D, got shape {Y.shape}")
    if check_finite and not np.all(np.isfinite(X)):
        raise ValueError("Design matrix contains NA/Inf values")
    if check_finite and not np.all(np.isfinite(Y)):
        raise ValueError("Response matrix contains NA/Inf values")

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y dimensions do not match")

    if proj.Pinv.ndim != 2:
        raise ValueError("Projection Pinv must be 2-D")
    if proj.Pinv.shape != (X.shape[1], X.shape[0]):
        raise ValueError("X and projection dimensions do not match")

    # B = Pinv @ Y   →  (p, V)
    betas = proj.Pinv @ Y

    use_fast_rss = (
        not return_fitted
        and proj.is_full_rank
        and not proj.ill_conditioned
    )

    if not use_fast_rss:
        fitted = X @ betas
        residuals = Y - fitted
        rss = np.sum(residuals ** 2, axis=0)
        if not return_fitted:
            fitted = None
    else:
        # Memory-efficient: compute RSS without materialising residuals.
        # RSS = Y'Y - B' X'Y
        if proj.XtX is not None:
            XtY = proj.XtX @ betas
        else:
            XtY = X.T @ Y
        yTy = np.sum(Y * Y, axis=0)
        fitted_ss = np.einsum("ij,ij->j", betas, XtY)
        rss = yTy - fitted_ss
        unstable = _rss_needs_direct_residual(rss, yTy, fitted_ss)
        if np.any(unstable):
            residuals = Y[:, unstable] - X @ betas[:, unstable]
            rss = rss.copy()
            rss[unstable] = np.sum(residuals ** 2, axis=0)
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
