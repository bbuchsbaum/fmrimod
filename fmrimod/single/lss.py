"""Vectorized Least Squares Separate (LSS) estimation.

Ports the vectorized LSS algorithm from R's ``fmrilss`` package,
replacing the naive per-trial loop with closed-form 2x2 normal
equation solves using precomputed cross-products.

References
----------
Mumford, J. A., Turner, B. O., Ashby, F. G., & Poldrack, R. A. (2012).
Deconvolving BOLD activation in event-related designs for multivoxel
pattern classification analyses. *NeuroImage*, 59(3), 2636-2643.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ._project import NuisanceProjector, build_nuisance_projector
from ._types import SingleTrialResult


def _auto_chunk_size(
    n_trials: int,
    n_voxels: int,
    target_working_mb: int = 96,
) -> int:
    """Heuristic voxel chunk size for beta-only LSS.

    Targets a bounded temporary working set for the `(T, V)` intermediates.
    """
    if n_voxels < 1:
        return 1
    # For moderate shapes, full-matrix path is often faster.
    if n_trials * n_voxels <= 2_000_000:
        return n_voxels
    # Also keep full-matrix when the core trial-by-voxel block is still
    # modest in memory footprint (~64 MB).
    core_block_mb = (n_trials * n_voxels * 8) / (1024.0 * 1024.0)
    if core_block_mb <= 64.0:
        return n_voxels

    bytes_budget = int(target_working_mb * 1024 * 1024)
    # Roughly 3 trial-by-voxel temporaries per chunk (CtY, BtY, num).
    bytes_per_voxel = max(1, n_trials * 8 * 3)
    chunk = max(512, bytes_budget // bytes_per_voxel)
    return int(min(n_voxels, chunk))


def _lss_beta_vec(
    C: NDArray[np.float64],
    Y: NDArray[np.float64],
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Vectorized LSS beta computation (no per-trial loop).

    For each trial *j*, the LSS model is::

        Y = c_j * beta_j  +  b_j * gamma_j  +  E

    where ``c_j`` is trial *j*'s regressor and ``b_j = sum_{i!=j} c_i``
    is the aggregator.  This function solves all trials simultaneously
    using the 2x2 normal equation structure.

    Parameters
    ----------
    C : NDArray, shape ``(n, T)``
        Trial regressor matrix (already projected free of nuisance).
    Y : NDArray, shape ``(n, V)``
        Data matrix (already projected free of nuisance).
    eps : float
        Numerical tolerance for near-zero denominators.

    Returns
    -------
    NDArray, shape ``(T, V)``
        Trial-wise beta estimates.
    """
    T = C.shape[1]

    # Shared cross-products (computed once) ----------------------------------
    total = C.sum(axis=1)               # (n,) sum of all trial columns
    ss_tot = float(total @ total)       # scalar: ||total||^2

    CtY = C.T @ Y                       # (T, V)
    CtC = np.einsum("ij,ij->j", C, C)   # (T,) per-trial ||c_j||^2
    CtT = C.T @ total                    # (T,) c_j' total
    total_Y = total @ Y                  # (V,)

    # Per-trial aggregator cross-products ------------------------------------
    # b_j = total - c_j
    # b_j'Y = total'Y - c_j'Y
    BtY = total_Y[np.newaxis, :] - CtY  # (T, V)

    # ||b_j||^2 = ||total||^2 - 2*(c_j'total) + ||c_j||^2
    bt2 = ss_tot - 2.0 * CtT + CtC      # (T,)

    # c_j'b_j = c_j'total - ||c_j||^2
    ctb = CtT - CtC                      # (T,)

    # Guard against near-zero aggregators (e.g. single trial)
    bt2_safe = bt2.copy()
    bt2_safe[bt2_safe < eps] = np.inf

    # Closed-form 2x2 solve --------------------------------------------------
    # beta_j = (c_j'y - (c_j'b_j / ||b_j||^2) * b_j'y)
    #        / (||c_j||^2 - (c_j'b_j)^2 / ||b_j||^2)
    ctb_bt2 = ctb / bt2_safe             # (T,)
    num = CtY - ctb_bt2[:, np.newaxis] * BtY   # (T, V)
    den = CtC - ctb ** 2 / bt2_safe      # (T,)

    return num / np.maximum(den, eps)[:, np.newaxis]  # (T, V)


def _lss_beta_vec_chunked(
    C: NDArray[np.float64],
    Y: NDArray[np.float64],
    chunk_size: int = 5000,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Chunked vectorized LSS beta computation across voxels.

    Useful for large ``V`` where the fully materialized ``(T, V)``
    intermediates are less cache-friendly.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    T = C.shape[1]
    V = Y.shape[1]
    out = np.empty((T, V), dtype=np.float64)

    total = C.sum(axis=1)               # (n,)
    ss_tot = float(total @ total)       # scalar
    CtC = np.einsum("ij,ij->j", C, C)   # (T,)
    CtT = C.T @ total                   # (T,)
    bt2 = ss_tot - 2.0 * CtT + CtC      # (T,)
    ctb = CtT - CtC                      # (T,)

    bt2_safe = bt2.copy()
    bt2_safe[bt2_safe < eps] = np.inf

    ctb_bt2 = ctb / bt2_safe             # (T,)
    den = CtC - ctb ** 2 / bt2_safe      # (T,)
    den_safe = np.maximum(den, eps)

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        Y_chunk = Y[:, start:end]                        # (n, vc)
        CtY = C.T @ Y_chunk                               # (T, vc)
        total_Y = total @ Y_chunk                         # (vc,)
        BtY = total_Y[np.newaxis, :] - CtY                # (T, vc)
        num = CtY - ctb_bt2[:, np.newaxis] * BtY          # (T, vc)
        out[:, start:end] = num / den_safe[:, np.newaxis]

    return out


def _lss_beta_vec_with_se(
    C: NDArray[np.float64],
    Y: NDArray[np.float64],
    eps: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Vectorized LSS with standard errors.

    Returns
    -------
    betas : NDArray, shape ``(T, V)``
    se : NDArray, shape ``(T, V)``
    residual_var : NDArray, shape ``(T, V)``
        Per-trial residual variance estimate.
    """
    n, T = C.shape
    V = Y.shape[1]

    total = C.sum(axis=1)
    ss_tot = float(total @ total)

    CtY = C.T @ Y
    CtC = np.einsum("ij,ij->j", C, C)
    CtT = C.T @ total
    total_Y = total @ Y

    BtY = total_Y[np.newaxis, :] - CtY
    bt2 = ss_tot - 2.0 * CtT + CtC
    ctb = CtT - CtC

    bt2_safe = bt2.copy()
    bt2_safe[bt2_safe < eps] = np.inf

    ctb_bt2 = ctb / bt2_safe
    num = CtY - ctb_bt2[:, np.newaxis] * BtY
    den = CtC - ctb ** 2 / bt2_safe
    den_safe = np.maximum(den, eps)

    betas = num / den_safe[:, np.newaxis]

    # Gamma (aggregator coefficient) for residual computation
    gammas = (BtY - ctb[:, np.newaxis] * betas) / bt2_safe[:, np.newaxis]

    # Residual variance: ||Y - c_j*beta_j - b_j*gamma_j||^2 / (n - 2)
    # Computed efficiently from precomputed products
    YtY = np.einsum("ij,ij->j", Y, Y)  # (V,)
    dfres = max(n - 2, 1)
    rss = (
        YtY[np.newaxis, :]
        - 2.0 * betas * CtY
        - 2.0 * gammas * BtY
        + betas ** 2 * CtC[:, np.newaxis]
        + 2.0 * betas * gammas * ctb[:, np.newaxis]
        + gammas ** 2 * bt2[:, np.newaxis]
    )
    rss = np.maximum(rss, 0.0)
    sigma2 = rss / dfres

    # SE = sqrt(sigma2 / den)
    se = np.sqrt(sigma2 / den_safe[:, np.newaxis])

    return betas, se, sigma2


def lss_single_trial(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    confounds: Optional[NDArray[np.float64]] = None,
    nuisance_projector: Optional[NuisanceProjector] = None,
    chunk_size: Optional[int] = None,
    return_se: bool = False,
    trial_labels: Optional[list] = None,
) -> SingleTrialResult:
    """Vectorized LSS estimation.

    Parameters
    ----------
    Y : NDArray, shape ``(n, V)``
        Data matrix (time x voxels).
    X : NDArray, shape ``(n, n_trials)``
        Trial regressor matrix (already convolved with HRF).
    confounds : NDArray, shape ``(n, q)``, optional
        Nuisance regressors (motion, drift, etc.).
    nuisance_projector : NuisanceProjector, optional
        Precomputed nuisance projector returned by
        :func:`fmrimod.single._project.build_nuisance_projector`.
        When provided, ``confounds`` is ignored.
    chunk_size : int, optional
        Voxel chunk size for beta-only LSS computation.  If ``None``,
        a heuristic is used to choose full-matrix vs chunked solve.
    return_se : bool
        If ``True``, compute standard errors.
    trial_labels : list of str, optional
        Labels for each trial.

    Returns
    -------
    SingleTrialResult
    """
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    n, n_trials = X.shape
    if Y.shape[0] != n:
        raise ValueError(
            f"Y has {Y.shape[0]} timepoints, X has {n}."
        )

    # Project out nuisance
    n_nuisance = 0
    if nuisance_projector is not None:
        if nuisance_projector.n_rows != n:
            raise ValueError(
                f"nuisance_projector has {nuisance_projector.n_rows} rows, expected {n}."
            )
        Y_clean, X_clean = nuisance_projector.project(Y, X)
        n_nuisance = nuisance_projector.n_cols
    elif confounds is not None:
        confounds = np.asarray(confounds, dtype=np.float64)
        if confounds.shape[0] != n:
            raise ValueError(
                f"confounds has {confounds.shape[0]} rows, expected {n}."
            )
        projector = build_nuisance_projector(confounds)
        if projector is None:
            Y_clean, X_clean = Y, X
        else:
            Y_clean, X_clean = projector.project(Y, X)
            n_nuisance = projector.n_cols
    else:
        Y_clean, X_clean = Y, X

    if return_se:
        betas, se, _ = _lss_beta_vec_with_se(X_clean, Y_clean)
    else:
        effective_chunk_size = (
            _auto_chunk_size(X_clean.shape[1], Y_clean.shape[1])
            if chunk_size is None
            else chunk_size
        )
        if effective_chunk_size >= Y_clean.shape[1]:
            betas = _lss_beta_vec(X_clean, Y_clean)
        else:
            betas = _lss_beta_vec_chunked(
                X_clean, Y_clean, chunk_size=effective_chunk_size
            )
        se = None
    dfres = float(n - 2 - n_nuisance)

    return SingleTrialResult(
        betas=betas,
        method="lss",
        trial_labels=list(trial_labels) if trial_labels is not None else None,
        residual_df=dfres,
        se=se,
    )
