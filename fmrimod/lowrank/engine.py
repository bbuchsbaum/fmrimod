"""Sketch-based low-rank GLM solver.

Solves the fMRI GLM ``Y = X B + E`` using randomised sketching to
reduce either the temporal dimension (row sketch) or the spatial
dimension (landmark Nyström extension), or both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from ..glm.solver import LmResult, fast_lm_matrix, fast_preproject
from .nystrom import (
    LandmarkWeights,
    build_landmark_weights,
    extend_betas,
    select_landmarks,
)
from .sketch import make_sketch, sketch_data


@dataclass
class LowRankConfig:
    """Configuration for the low-rank GLM solver.

    Parameters
    ----------
    sketch_kind : str
        Sketch type (``"gaussian"``, ``"srht"``, ``"countsketch"``).
    sketch_ratio : float
        Fraction of rows to keep when sketching temporally.
        E.g. ``0.5`` halves the time dimension.
    sketch_size : int or None
        Exact temporal sketch size. When provided, this takes precedence over
        ``sketch_ratio``.
    use_landmarks : bool
        Whether to use Nyström landmark extension for spatial
        dimension reduction.
    n_landmarks : int
        Number of landmark voxels (ignored if ``use_landmarks=False``).
    landmark_k : int
        k-NN parameter for landmark weight matrix.
    landmark_method : str
        Landmark selection method (``"kmeans"`` or ``"random"``).
    ridge : float
        Ridge penalty for numerical stability.
    seed : int or None
        Random seed for reproducibility.
    """

    sketch_kind: str = "gaussian"
    sketch_ratio: float = 0.5
    sketch_size: Optional[int] = None
    use_landmarks: bool = False
    n_landmarks: int = 500
    landmark_k: int = 6
    landmark_method: str = "kmeans"
    ridge: float = 0.0
    seed: Optional[int] = None


@dataclass
class SketchLmResult(LmResult):
    """:class:`LmResult` of a sketch-and-solve fit, with its inference.

    Attributes
    ----------
    cov_unscaled : NDArray or None
        Unscaled coefficient covariance ``(p, p)`` of the estimator that was
        computed: ``G^-1 Xs' K Xs G^-1`` for a temporal sketch (``Xs = S X``,
        ``G = Xs'Xs``, ``K = S S'``), ``(X'X)^-1`` without one. ``None`` for a
        ridge solve, whose inference is not derived here.
    kappa : float or None
        ``tr(P K)``, the divisor that makes ``sigma2 = rss / kappa``
        unbiased (``n - rank`` without a sketch).
    sketch_rows : int or None
        Rows ``m`` of the temporal sketch, ``None`` when none was applied.
    """

    cov_unscaled: NDArray[np.float64] | None = None
    kappa: float | None = None
    sketch_rows: int | None = None


def fit_sketched(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    config: LowRankConfig,
    coords: Optional[NDArray[np.float64]] = None,
) -> SketchLmResult:
    """Fit a GLM using sketch-based dimensionality reduction.

    This sketches the temporal dimension of ``(X, Y)`` before solving,
    which trades a small amount of accuracy for large speed gains when
    *n* (time points) is big relative to *p* (parameters).

    Optionally, if ``config.use_landmarks`` is True and *coords* are
    provided, it first solves at landmark voxels then extends via
    Nyström interpolation.

    Parameters
    ----------
    X : NDArray, shape ``(n, p)``
        Design matrix.
    Y : NDArray, shape ``(n, V)``
        Data matrix.
    config : LowRankConfig
        Solver configuration.
    coords : NDArray, shape ``(V, d)``, optional
        Voxel coordinates for landmark extension.

    Returns
    -------
    SketchLmResult
        Regression results with sketched estimates. A temporal sketch
        reports inference for the sketched estimator conditional on the
        sketch (fmrireg cdb5751): ``cov_unscaled`` as above,
        ``sigma2 = ||r_s||^2 / tr(PK)`` and the Satterthwaite ``dfres =
        tr(PK)^2 / tr(PKPK)`` (``m - p`` when ``K`` is a multiple of the
        identity).

    Raises
    ------
    ValueError
        If the sketched design has lower rank than the design itself (the
        sketch has too few rows for it).
    """
    rng = np.random.default_rng(config.seed)
    n, p = X.shape
    V = Y.shape[1]

    # -- Landmark reduction (spatial) --
    landmark_weights: Optional[LandmarkWeights] = None
    if config.use_landmarks and coords is not None:
        n_lm = min(config.n_landmarks, V)
        lm_idx = select_landmarks(
            coords, n_lm,
            method=config.landmark_method,
            rng=rng,
        )
        landmark_coords = coords[lm_idx]
        landmark_weights = build_landmark_weights(
            coords, landmark_coords, k=config.landmark_k,
        )
        # Solve only at landmarks
        Y_solve = Y[:, lm_idx]
    else:
        Y_solve = Y

    # -- Temporal sketching --
    if config.sketch_size is not None:
        if isinstance(config.sketch_size, bool) or config.sketch_size < 1:
            raise ValueError("sketch_size must be a positive integer or None")
        k = int(config.sketch_size)
    else:
        if not (0.0 < float(config.sketch_ratio) <= 1.0):
            raise ValueError("sketch_ratio must be in (0, 1]")
        k = int(n * config.sketch_ratio)
    k = max(p + 1, k)
    k = min(k, n)  # can't sketch to more rows than we have

    S = make_sketch(n, k, kind=config.sketch_kind, rng=rng) if k < n else None

    # -- Solve (with optional ridge) --
    cov_unscaled: NDArray[np.float64] | None
    if config.ridge > 0:
        X_sk, Y_sk = sketch_data(S, X, Y_solve) if S is not None else (X, Y_solve)
        # Ridge regression: add sqrt(ridge) * I to X
        X_ridge = np.vstack([X_sk, np.sqrt(config.ridge) * np.eye(p)])
        Y_ridge = np.vstack([Y_sk, np.zeros((p, Y_sk.shape[1]))])
        proj = fast_preproject(X_ridge)
        ridge_fit = fast_lm_matrix(X_ridge, Y_ridge, proj)
        betas, rss, rank = ridge_fit.betas, ridge_fit.rss, proj.rank
        # Correct dfres for the original problem size
        kappa = dfres = float(n - rank)
        sigma2 = rss / max(kappa, 1.0)
        cov_unscaled = None
    elif S is None:
        proj = fast_preproject(X)
        exact = fast_lm_matrix(X, Y_solve, proj)
        betas, rss, rank = exact.betas, exact.rss, proj.rank
        kappa = dfres = float(proj.dfres)
        sigma2 = exact.sigma2
        cov_unscaled = proj.XtXinv
    else:
        betas, rss, cov_unscaled, kappa, dfres, rank = _sketch_solve(
            X, Y_solve, S, sketch_kind=str(config.sketch_kind)
        )
        sigma2 = rss / kappa if kappa > 0 else np.full_like(rss, np.nan)
    result = SketchLmResult(
        betas=betas,
        rss=rss,
        sigma2=sigma2,
        dfres=dfres,
        rank=rank,
        cov_unscaled=cov_unscaled,
        kappa=kappa,
        sketch_rows=None if S is None else k,
    )

    # -- Nyström extension (spatial) --
    if landmark_weights is not None:
        betas_full = extend_betas(result.betas, landmark_weights)
        # Recompute RSS at full resolution; the residual variance then comes
        # from the full data, while cov_unscaled still describes the
        # (possibly sketched) landmark estimator.
        residuals = Y - X @ betas_full
        rss_full = np.sum(residuals**2, axis=0)
        dfres = float(n - result.rank)
        result = SketchLmResult(
            betas=betas_full,
            rss=rss_full,
            sigma2=rss_full / max(dfres, 1),
            dfres=dfres,
            rank=result.rank,
            cov_unscaled=result.cov_unscaled,
            kappa=dfres,
            sketch_rows=result.sketch_rows,
        )

    return result


def _sketch_solve(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    S: NDArray[np.float64],
    *,
    sketch_kind: str,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, float, int
]:
    """Sketch-and-solve with its conditional-on-``S`` inference.

    Port of fmrireg ``.lowrank_sketch_solve`` (R/sketch_time.R:169-247).
    Solves ``b_s = G^-1 Xs' S y`` on the estimable columns of ``X`` through
    the pivoted QR of ``Xs = S X`` (never a Cholesky of ``G``, which can
    succeed on a numerically singular ``G``) and returns ``(betas, rss,
    cov_unscaled, kappa, df, rank)``. Aliased columns of ``X`` keep their
    position with zero coefficients and covariance. There is no ridge or
    pseudo-inverse fallback: either would change the estimator that the
    reported inference describes.
    """
    m = S.shape[0]
    p = X.shape[1]
    aliased = set(fast_preproject(X).aliased_indices)
    est = np.array([j for j in range(p) if j not in aliased], dtype=int)

    Xs = S @ X[:, est]
    Ys = S @ Y
    q = est.size
    Q, R, pivot = linalg.qr(Xs, mode="economic", pivoting=True, check_finite=False)
    diag_R = np.abs(np.diag(R))
    tol = max(Xs.shape) * np.finfo(np.float64).eps * (diag_R[0] if q else 1.0)
    rank = int(np.sum(diag_R > tol))
    if rank < q:
        hint = (
            ", or use sketch_kind 'srht' or 'gaussian': CountSketch maps each "
            "scan to one row, so sparse regressors such as spike (scrubbing) "
            "columns collide"
            if sketch_kind == "countsketch"
            else ""
        )
        raise ValueError(
            f"sketch_size m = {m} is too small for this design: the sketched "
            f"design ({m} rows x {q} estimable columns) has numerical rank "
            f"{rank}. Increase the sketch size{hint}."
        )

    R_inv = linalg.solve_triangular(R, np.eye(q), check_finite=False)
    G_inv = np.empty((q, q))
    G_inv[np.ix_(pivot, pivot)] = R_inv @ R_inv.T
    QtY = Q.T @ Ys
    coef = np.empty((q, Ys.shape[1]))
    coef[pivot] = R_inv @ QtY
    resid = Ys - Q @ QtY
    rss = np.sum(resid * resid, axis=0)

    K = S @ S.T
    cov = G_inv @ (Xs.T @ (K @ Xs)) @ G_inv
    cov = (cov + cov.T) / 2.0
    KQ = K @ Q
    QKQ = Q.T @ KQ
    kappa = float(np.trace(K) - np.trace(QKQ))  # tr(PK)
    tr_PKPK = float(np.sum(K * K) - 2.0 * np.sum(KQ * KQ) + np.sum(QKQ * QKQ))
    df = kappa**2 / tr_PKPK if kappa > 0 and tr_PKPK > 0 else 1.0

    betas = np.zeros((p, Ys.shape[1]))
    betas[est] = coef
    cov_unscaled = np.zeros((p, p))
    cov_unscaled[np.ix_(est, est)] = cov
    return betas, rss, cov_unscaled, kappa, max(df, 1.0), q
