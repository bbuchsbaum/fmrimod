"""Preprocessing steps applied before GLM fitting.

Includes volume weighting, soft subspace projection, and censoring.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def apply_volume_weights(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply per-volume (scan) weights to design and data matrices.

    Multiplies each row by ``sqrt(w_i)`` so that subsequent OLS
    is equivalent to WLS with diagonal weight matrix.

    Parameters
    ----------
    X : NDArray
        Design matrix, shape ``(n, p)``.
    Y : NDArray
        Data matrix, shape ``(n, V)``.
    weights : NDArray
        Non-negative weight vector, shape ``(n,)``.

    Returns
    -------
    X_w, Y_w : tuple of NDArray
        Weighted design and data matrices.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must both be 2-D matrices")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows")
    if weights.ndim != 1:
        raise ValueError("weights must be a 1-D array")
    if weights.shape[0] != X.shape[0]:
        raise ValueError(
            f"weights length {weights.shape[0]} does not match matrix rows {X.shape[0]}"
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    w_sqrt = np.sqrt(np.maximum(weights, 0.0))
    X_w = X * w_sqrt[:, np.newaxis]
    Y_w = Y * w_sqrt[:, np.newaxis]
    return X_w, Y_w


def compute_dvars(
    Y: NDArray[np.float64],
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Compute DVARS (temporal derivative of voxelwise variance).

    Parity with fmrireg:
    - first timepoint is set to median of subsequent DVARS values
    - optional normalization by median DVARS (enabled by default)

    Parameters
    ----------
    Y : NDArray
        Data matrix, shape ``(n, V)``.
    normalize : bool
        If ``True``, divide by median DVARS (fmrireg default).

    Returns
    -------
    NDArray
        DVARS values, shape ``(n,)``.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Y.ndim != 2:
        raise ValueError("Y must be a 1-D or 2-D array")

    n = Y.shape[0]
    if n < 2:
        raise ValueError("DVARS requires at least 2 timepoints")

    # Temporal derivative: Y[t] - Y[t-1]
    dY = np.diff(Y, axis=0)
    # RMS across voxels for each timepoint
    dvars_raw = np.sqrt(np.mean(dY ** 2, axis=1))

    # First timepoint has no derivative; use median of subsequent values
    dvars = np.concatenate([[np.median(dvars_raw)], dvars_raw])

    if normalize:
        med = np.median(dvars)
        if med > 0:
            dvars = dvars / med

    return dvars


def dvars_weights(
    dvars: NDArray[np.float64],
    method: str = "inverse_squared",
    threshold: float = 1.5,
    steepness: float = 2.0,
) -> NDArray[np.float64]:
    """Convert DVARS into per-volume weights.

    Parameters
    ----------
    dvars : NDArray
        DVARS values, shape ``(n,)``.
    method : str
        ``"inverse_squared"``, ``"soft_threshold"``, or ``"tukey"``.
    threshold : float
        Threshold in MAD units.
    steepness : float
        Decay steepness for ``"soft_threshold"`` method.

    Returns
    -------
    NDArray
        Weights, shape ``(n,)``.
    """
    dvars = np.asarray(dvars, dtype=np.float64).ravel()
    if np.any(dvars < 0):
        raise ValueError("DVARS values must be non-negative")
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    if method == "inverse_squared":
        # w = 1 / (1 + dvars^2)
        w = 1.0 / (1.0 + dvars ** 2)
    elif method == "soft_threshold":
        # Sigmoid-like decay above threshold
        w = 1.0 / (
            1.0 + ((np.maximum(dvars, threshold) - threshold) / threshold) ** steepness
        )
    elif method == "tukey":
        c_tukey = threshold * 2.0
        u = dvars / c_tukey
        w = np.where(np.abs(u) <= 1.0, (1.0 - u ** 2) ** 2, 0.0)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure weights are in [0, 1], then normalize mean to ~1
    w = np.clip(w, 0.0, 1.0)
    mean_w = np.mean(w)
    if mean_w > 0:
        w = w / mean_w
    return w


def apply_censoring(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    censor: NDArray[np.bool_],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Remove censored volumes from design and data.

    Parameters
    ----------
    X : NDArray
        Design matrix, shape ``(n, p)``.
    Y : NDArray
        Data matrix, shape ``(n, V)``.
    censor : NDArray[bool]
        Boolean vector, shape ``(n,)``.  ``True`` = censored (excluded).

    Returns
    -------
    X_clean, Y_clean, keep_mask : tuple
        Subsetted matrices and the boolean keep mask.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    censor_arr = np.asarray(censor)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must both be 2-D matrices")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows")
    if censor_arr.ndim != 1 or censor_arr.shape[0] != X.shape[0]:
        raise ValueError(
            f"Censor vector length {censor_arr.shape[0]} does not match matrix rows {X.shape[0]}"
        )

    if np.issubdtype(censor_arr.dtype, np.bool_):
        censor_mask = censor_arr
    elif np.issubdtype(censor_arr.dtype, np.number):
        if not np.all(np.isfinite(censor_arr)):
            raise ValueError("Censor vector must contain finite values")
        if not np.all(np.isin(censor_arr, [0, 1])):
            raise ValueError("Censor vector must be boolean or binary (0/1)")
        censor_mask = censor_arr.astype(bool)
    else:
        raise ValueError("Censor vector must be boolean or binary (0/1)")

    keep = ~censor_mask
    return X[keep], Y[keep], keep


def soft_subspace_projection(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    nuisance: NDArray[np.float64],
    lam: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply soft subspace projection to remove nuisance signals.

    Projects both X and Y onto the space orthogonal to the nuisance
    subspace, with soft (ridge-regularised) shrinkage controlled by *lam*.

    Parameters
    ----------
    X : NDArray
        Design matrix, shape ``(n, p)``.
    Y : NDArray
        Data matrix, shape ``(n, V)``.
    nuisance : NDArray
        Nuisance matrix, shape ``(n, q)``.
    lam : float
        Regularisation parameter.  0 = hard projection, larger = softer.

    Returns
    -------
    X_proj, Y_proj : tuple of NDArray
        Projected matrices.
    """
    n, q = nuisance.shape
    NtN = nuisance.T @ nuisance
    # Regularised projection: P = N @ (N'N + lam*I)^{-1} @ N'
    reg = NtN + lam * np.eye(q)
    try:
        L = np.linalg.cholesky(reg)
        NtN_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(q)))
    except np.linalg.LinAlgError:
        NtN_inv = np.linalg.pinv(reg)

    P = nuisance @ NtN_inv @ nuisance.T
    I_minus_P = np.eye(n) - P

    return I_minus_P @ X, I_minus_P @ Y
