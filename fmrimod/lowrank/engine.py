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
    use_landmarks: bool = False
    n_landmarks: int = 500
    landmark_k: int = 6
    landmark_method: str = "kmeans"
    ridge: float = 0.0
    seed: Optional[int] = None


def fit_sketched(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    config: LowRankConfig,
    coords: Optional[NDArray[np.float64]] = None,
) -> LmResult:
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
    LmResult
        Regression results with sketched estimates.
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
    k = max(p + 1, int(n * config.sketch_ratio))
    k = min(k, n)  # can't sketch to more rows than we have

    if k < n:
        S = make_sketch(n, k, kind=config.sketch_kind, rng=rng)
        X_sk, Y_sk = sketch_data(S, X, Y_solve)
    else:
        X_sk, Y_sk = X, Y_solve

    # -- Solve (with optional ridge) --
    if config.ridge > 0:
        # Ridge regression: add sqrt(ridge) * I to X
        X_ridge = np.vstack([X_sk, np.sqrt(config.ridge) * np.eye(p)])
        Y_ridge = np.vstack([Y_sk, np.zeros((p, Y_sk.shape[1]))])
        proj = fast_preproject(X_ridge)
        result = fast_lm_matrix(X_ridge, Y_ridge, proj)
        # Correct dfres for the original problem size
        result = LmResult(
            betas=result.betas,
            rss=result.rss,
            sigma2=result.rss / max(n - proj.rank, 1),
            dfres=float(n - proj.rank),
            rank=proj.rank,
            fitted=None,
        )
    else:
        proj = fast_preproject(X_sk)
        result = fast_lm_matrix(X_sk, Y_sk, proj)
        # Correct sigma2 for original n
        result = LmResult(
            betas=result.betas,
            rss=result.rss,
            sigma2=result.rss / max(n - proj.rank, 1),
            dfres=float(n - proj.rank),
            rank=proj.rank,
            fitted=None,
        )

    # -- Nyström extension (spatial) --
    if landmark_weights is not None:
        betas_full = extend_betas(result.betas, landmark_weights)
        # Recompute RSS at full resolution
        residuals = Y - X @ betas_full
        rss_full = np.sum(residuals**2, axis=0)
        dfres = float(n - result.rank)
        result = LmResult(
            betas=betas_full,
            rss=rss_full,
            sigma2=rss_full / max(dfres, 1),
            dfres=dfres,
            rank=result.rank,
            fitted=None,
        )

    return result
