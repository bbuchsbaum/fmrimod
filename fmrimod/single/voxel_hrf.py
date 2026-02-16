"""Per-voxel HRF estimation and HRF-aware LSS.

Ports R ``fmrilss`` voxel_hrf.R: estimate voxel-specific HRF shapes
from a multi-basis GLM, then run LSS with per-voxel trial regressors
reconstructed from the estimated shapes.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ._project import project_nuisance
from ._types import SingleTrialResult, VoxelHrfResult
from .lss import _lss_beta_vec


def estimate_voxel_hrf(
    Y: NDArray[np.float64],
    X_trials: NDArray[np.float64],
    basis: Any,
    confounds: Optional[NDArray[np.float64]] = None,
    K: Optional[int] = None,
) -> VoxelHrfResult:
    """Estimate per-voxel HRF via multi-basis aggregate GLM.

    Parameters
    ----------
    Y : NDArray, shape ``(T, V)``
        fMRI data.
    X_trials : NDArray, shape ``(T, N*K)``
        Trial design with K basis functions per trial, interleaved:
        ``[t1_b1, t1_b2, ..., t1_bK, t2_b1, ...]``.
    basis : object
        HRF basis object (stored in result for reference).
    confounds : NDArray, shape ``(T, q)``, optional
        Nuisance regressors to project out.
    K : int, optional
        Number of basis functions per trial.  Inferred from *basis*
        shape if possible.

    Returns
    -------
    VoxelHrfResult
        ``coefficients`` has shape ``(K, V)``, normalised to unit L2
        norm per voxel.
    """
    Y = np.asarray(Y, dtype=np.float64)
    X_trials = np.asarray(X_trials, dtype=np.float64)
    T, V = Y.shape

    if K is None:
        b = np.asarray(basis)
        if b.ndim == 2:
            K = b.shape[1]
        else:
            raise ValueError("Cannot infer K; pass K explicitly")

    NK = X_trials.shape[1]
    if NK % K != 0:
        raise ValueError(f"ncol(X_trials)={NK} not divisible by K={K}")
    N = NK // K

    # Aggregate per-basis regressors: A[:, k] = sum of all trial-k columns
    A = np.zeros((T, K), dtype=np.float64)
    for k in range(K):
        A[:, k] = X_trials[:, k::K].sum(axis=1)

    # Project out confounds
    if confounds is not None:
        confounds = np.asarray(confounds, dtype=np.float64)
        Y_c, A_c = project_nuisance(confounds, Y, A)
    else:
        Y_c, A_c = Y, A

    # Solve A'A @ W = A'Y via least squares → W (K, V)
    W, _, _, _ = np.linalg.lstsq(A_c, Y_c, rcond=None)

    # Normalise to unit L2 norm per voxel
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    W = W / norms

    return VoxelHrfResult(coefficients=W, basis=basis, conditions=[])


def lss_with_voxel_hrf(
    Y: NDArray[np.float64],
    X_trials: NDArray[np.float64],
    hrf_result: VoxelHrfResult,
    confounds: Optional[NDArray[np.float64]] = None,
    chunk_size: int = 5000,
) -> SingleTrialResult:
    """LSS with per-voxel HRF shapes.

    For each voxel, reconstructs scalar trial regressors from the
    multi-basis columns weighted by the estimated HRF coefficients,
    then runs vectorised LSS.

    Parameters
    ----------
    Y : NDArray, shape ``(T, V)``
    X_trials : NDArray, shape ``(T, N*K)``
        Same interleaved multi-basis format as :func:`estimate_voxel_hrf`.
    hrf_result : VoxelHrfResult
        From :func:`estimate_voxel_hrf`.
    confounds : NDArray, shape ``(T, q)``, optional
    chunk_size : int
        Voxels per chunk.

    Returns
    -------
    SingleTrialResult
    """
    Y = np.asarray(Y, dtype=np.float64)
    X_trials = np.asarray(X_trials, dtype=np.float64)
    T, V = Y.shape
    K = hrf_result.coefficients.shape[0]
    NK = X_trials.shape[1]
    N = NK // K

    if hrf_result.coefficients.shape[1] != V:
        raise ValueError(
            f"HRF result has {hrf_result.coefficients.shape[1]} voxels, "
            f"Y has {V}"
        )

    # Project out confounds
    if confounds is not None:
        confounds = np.asarray(confounds, dtype=np.float64)
        Y_c, X_c = project_nuisance(confounds, Y, X_trials)
    else:
        Y_c, X_c = Y, X_trials

    betas = np.empty((N, V), dtype=np.float64)

    # Process voxels in chunks
    for start_v in range(0, V, chunk_size):
        end_v = min(start_v + chunk_size, V)
        n_v = end_v - start_v
        W_chunk = hrf_result.coefficients[:, start_v:end_v]  # (K, n_v)

        # Build per-voxel scalar trial regressors: (T, N, n_v)
        # For trial n, voxel v: x_nv = sum_k W[k,v] * X_c[:, n*K+k]
        X_vox = np.zeros((T, N, n_v), dtype=np.float64)
        for n in range(N):
            trial_cols = X_c[:, n * K : (n + 1) * K]  # (T, K)
            X_vox[:, n, :] = trial_cols @ W_chunk      # (T, n_v)

        Y_chunk = Y_c[:, start_v:end_v]                # (T, n_v)

        # Run vectorised LSS per voxel (each has different X)
        for v_local in range(n_v):
            C_v = X_vox[:, :, v_local]                  # (T, N)
            y_v = Y_chunk[:, v_local : v_local + 1]     # (T, 1)
            b_v = _lss_beta_vec(C_v, y_v)               # (N, 1)
            betas[:, start_v + v_local] = b_v.ravel()

    return SingleTrialResult(
        betas=betas,
        method="lss_voxel_hrf",
        residual_df=float(T - 2),
    )
