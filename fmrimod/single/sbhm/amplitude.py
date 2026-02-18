"""SBHM amplitude estimation: single-trial betas with matched HRF shapes.

After matching per-voxel HRF shapes, this module reconstructs per-voxel
single-trial regressors and estimates amplitudes using one of three methods:
- global_ls: Standard OLS with all trials
- lss1: Per-trial LSS (2x2 system)
- oasis_voxel: OASIS K=1 per voxel
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .._types import OasisConfig
from ..oasis import oasis_single_trial


@contextmanager
def _maybe_limit_blas_threads(blas_threads: Optional[int]):
    """Limit BLAS threads within a context when threadpoolctl is available."""
    if blas_threads is None:
        yield
        return
    try:
        from threadpoolctl import threadpool_limits  # type: ignore[import-not-found]
    except Exception:
        yield
        return
    with threadpool_limits(limits=int(blas_threads)):
        yield


def _reconstruct_voxel_regressors(
    X_trials: NDArray[np.float64],
    alpha_coords: NDArray[np.float64],
    K: int,
) -> NDArray[np.float64]:
    """Reconstruct per-voxel single-trial regressors from basis and coordinates.

    Parameters
    ----------
    X_trials : NDArray, shape ``(T, N*K)``
        Interleaved trial regressors: [t1_b1, t1_b2, ..., t1_bK, t2_b1, ...].
    alpha_coords : NDArray, shape ``(r, V)`` or ``(K, V)``
        Per-voxel HRF shape coordinates.
    K : int
        Basis dimension (should match alpha_coords.shape[0]).

    Returns
    -------
    NDArray, shape ``(T, N, V)``
        Per-voxel single-trial regressors.
    """
    T, NK = X_trials.shape
    N = NK // K
    r, V = alpha_coords.shape

    if r != K:
        raise ValueError(f"alpha_coords has {r} rows, expected K={K}")

    # Reshape X_trials to (T, N, K)
    X_reshaped = np.zeros((T, N, K), dtype=np.float64)
    for k in range(K):
        X_reshaped[:, :, k] = X_trials[:, k::K]

    # Reconstruct: X_v[:, n] = sum_k alpha_k[v] * X[:, n, k]
    X_voxel = np.zeros((T, N, V), dtype=np.float64)
    for v in range(V):
        X_voxel[:, :, v] = X_reshaped @ alpha_coords[:, v]

    return X_voxel


def _global_ls(
    Y: NDArray[np.float64],
    X_voxel: NDArray[np.float64],
    confounds: Optional[NDArray[np.float64]] = None,
    ridge: float = 0.0,
) -> NDArray[np.float64]:
    """Global OLS per voxel with matched HRF.

    Parameters
    ----------
    Y : NDArray, shape ``(T, V)``
    X_voxel : NDArray, shape ``(T, N, V)``
    confounds : NDArray, shape ``(T, q)``, optional
    ridge : float
        Ridge regularization parameter.

    Returns
    -------
    NDArray, shape ``(N, V)``
        Trial betas.
    """
    T, N, V = X_voxel.shape

    # Project out confounds
    Y_clean = Y.copy()
    X_clean = X_voxel.copy()

    if confounds is not None:
        Q, _ = np.linalg.qr(confounds, mode="reduced")
        Y_clean -= Q @ (Q.T @ Y_clean)
        for v in range(V):
            X_clean[:, :, v] -= Q @ (Q.T @ X_clean[:, :, v])

    # Per-voxel OLS
    betas = np.zeros((N, V), dtype=np.float64)
    for v in range(V):
        Xv = X_clean[:, :, v]  # (T, N)
        yv = Y_clean[:, v]      # (T,)
        G = Xv.T @ Xv           # (N, N)
        if ridge > 0:
            G += ridge * np.eye(N)
        Xty = Xv.T @ yv         # (N,)
        try:
            betas[:, v] = np.linalg.solve(G, Xty)
        except np.linalg.LinAlgError:
            betas[:, v] = np.linalg.lstsq(G, Xty, rcond=None)[0]

    return betas


def _lss1_voxel_chunk(
    Y_chunk: NDArray[np.float64],
    X_chunk: NDArray[np.float64],
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Vectorized LSS1 solve for a voxel chunk.

    Parameters
    ----------
    Y_chunk : NDArray, shape ``(T, Vc)``
        Data for a contiguous chunk of voxels.
    X_chunk : NDArray, shape ``(T, N, Vc)``
        Voxel-specific trial design for the same chunk.
    eps : float
        Numerical guard for near-singular denominators.

    Returns
    -------
    NDArray, shape ``(N, Vc)``
        Trial-wise beta estimates for each voxel in the chunk.
    """
    total = X_chunk.sum(axis=1)                         # (T, Vc)
    ss_tot = np.einsum("tv,tv->v", total, total)        # (Vc,)

    CtY = np.einsum("tnv,tv->nv", X_chunk, Y_chunk)     # (N, Vc)
    CtC = np.einsum("tnv,tnv->nv", X_chunk, X_chunk)    # (N, Vc)
    CtT = np.einsum("tnv,tv->nv", X_chunk, total)       # (N, Vc)
    total_Y = np.einsum("tv,tv->v", total, Y_chunk)     # (Vc,)

    BtY = total_Y[np.newaxis, :] - CtY                  # (N, Vc)
    bt2 = ss_tot[np.newaxis, :] - 2.0 * CtT + CtC       # (N, Vc)
    ctb = CtT - CtC                                      # (N, Vc)

    bt2_safe = bt2.copy()
    bt2_safe[bt2_safe < eps] = np.inf

    ctb_bt2 = ctb / bt2_safe
    num = CtY - ctb_bt2 * BtY
    den = CtC - (ctb * ctb) / bt2_safe
    return num / np.maximum(den, eps)


def _lss1_voxelwise_chunked(
    Y: NDArray[np.float64],
    X_voxel: NDArray[np.float64],
    confounds: Optional[NDArray[np.float64]] = None,
    chunk_size: int = 4096,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Chunked vectorized voxel-wise LSS1 solver."""
    T, N, V = X_voxel.shape
    if Y.shape != (T, V):
        raise ValueError(
            f"Y has shape {Y.shape}, expected ({T}, {V}) to match X_voxel."
        )
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    if confounds is not None:
        confounds = np.asarray(confounds, dtype=np.float64)
        if confounds.shape[0] != T:
            raise ValueError(
                f"confounds has {confounds.shape[0]} rows, expected {T}."
            )
        Q, _ = np.linalg.qr(confounds, mode="reduced")
        Y_clean = Y - Q @ (Q.T @ Y)
    else:
        Q = None
        Y_clean = Y

    betas = np.empty((N, V), dtype=np.float64)
    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        X_chunk = X_voxel[:, :, start:end]                         # (T, N, Vc)
        if Q is not None:
            QtX = np.einsum("tq,tnv->qnv", Q, X_chunk)            # (q, N, Vc)
            X_chunk = X_chunk - np.einsum("tq,qnv->tnv", Q, QtX)  # (T, N, Vc)

        Y_chunk = Y_clean[:, start:end]                            # (T, Vc)
        betas[:, start:end] = _lss1_voxel_chunk(
            Y_chunk,
            X_chunk,
            eps=eps,
        )
    return betas


def sbhm_amplitude(
    Y: NDArray[np.float64],
    X_trials: NDArray[np.float64],
    alpha_coords: NDArray[np.float64],
    confounds: Optional[NDArray[np.float64]] = None,
    method: str = "oasis_voxel",
    ridge_x: float = 0.02,
    ridge_b: float = 0.02,
    K: Optional[int] = None,
) -> NDArray[np.float64]:
    """Estimate single-trial amplitudes with per-voxel matched HRF shapes.

    Parameters
    ----------
    Y : NDArray, shape ``(T, V)``
        Data matrix.
    X_trials : NDArray, shape ``(T, N*K)``
        Interleaved trial regressors (basis columns per trial).
    alpha_coords : NDArray, shape ``(r, V)``
        Per-voxel HRF shape coordinates from matching step.
    confounds : NDArray, shape ``(T, q)``, optional
        Nuisance regressors.
    method : str, default="oasis_voxel"
        Amplitude estimation method:
        - ``"global_ls"``: Standard OLS per voxel
        - ``"lss1"``: Vectorized LSS (2x2 per trial)
        - ``"oasis_voxel"``: OASIS K=1 per voxel
    ridge_x : float, default=0.02
        Ridge parameter for trial regressors (used by oasis_voxel).
    ridge_b : float, default=0.02
        Ridge parameter for aggregator (used by oasis_voxel).
    K : int, optional
        Basis dimension. If not provided, inferred from alpha_coords.

    Returns
    -------
    NDArray, shape ``(N, V)``
        Single-trial beta estimates.

    Notes
    -----
    All methods reconstruct per-voxel single-trial regressors by combining
    the basis columns with the matched coordinates::

        X_v[:, n] = sum_k alpha_k[v] * X_trials[:, n*K + k]

    Then apply the chosen estimation method.

    Examples
    --------
    >>> import numpy as np
    >>> from fmrimod.single.sbhm import sbhm_amplitude
    >>> Y = np.random.randn(100, 500)
    >>> X_trials = np.random.randn(100, 60)  # 20 trials x 3 basis
    >>> alpha_coords = np.random.randn(3, 500)
    >>> betas = sbhm_amplitude(Y, X_trials, alpha_coords, K=3)
    >>> betas.shape
    (20, 500)
    """
    Y = np.asarray(Y, dtype=np.float64)
    X_trials = np.asarray(X_trials, dtype=np.float64)
    alpha_coords = np.asarray(alpha_coords, dtype=np.float64)

    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if alpha_coords.ndim == 1:
        alpha_coords = alpha_coords[:, np.newaxis]

    T, V = Y.shape
    NK = X_trials.shape[1]

    if K is None:
        K = alpha_coords.shape[0]

    if NK % K != 0:
        raise ValueError(
            f"X_trials has {NK} columns, not divisible by K={K}"
        )
    N = NK // K

    if alpha_coords.shape[0] != K:
        raise ValueError(
            f"alpha_coords has {alpha_coords.shape[0]} rows, expected K={K}"
        )
    if alpha_coords.shape[1] != V:
        raise ValueError(
            f"alpha_coords has {alpha_coords.shape[1]} columns, Y has {V} voxels"
        )

    method = method.lower()

    if method == "global_ls":
        # Reconstruct per-voxel regressors, then OLS per voxel
        X_voxel = _reconstruct_voxel_regressors(X_trials, alpha_coords, K)
        betas = _global_ls(Y, X_voxel, confounds=confounds, ridge=0.0)
        return betas

    elif method == "lss1":
        # Reconstruct per-voxel regressors, then vectorized LSS per voxel
        X_voxel = _reconstruct_voxel_regressors(X_trials, alpha_coords, K)
        with _maybe_limit_blas_threads(1):
            return _lss1_voxelwise_chunked(
                Y,
                X_voxel,
                confounds=confounds,
            )

    elif method == "oasis_voxel":
        # Reconstruct per-voxel regressors, then OASIS K=1 per voxel
        X_voxel = _reconstruct_voxel_regressors(X_trials, alpha_coords, K)
        config = OasisConfig(
            K=1,
            ridge_mode="fractional",
            ridge_x=ridge_x,
            ridge_b=ridge_b,
        )
        betas = np.zeros((N, V), dtype=np.float64)
        with _maybe_limit_blas_threads(1):
            for v in range(V):
                result = oasis_single_trial(
                    Y[:, v:v+1],
                    X_voxel[:, :, v],
                    confounds=confounds,
                    config=config,
                )
                betas[:, v] = result.betas[:, 0]
        return betas

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: "
            "global_ls, lss1, oasis_voxel"
        )
