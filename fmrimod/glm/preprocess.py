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
    w_sqrt = np.sqrt(np.maximum(weights, 0.0))
    X_w = X * w_sqrt[:, np.newaxis]
    Y_w = Y * w_sqrt[:, np.newaxis]
    return X_w, Y_w


def compute_dvars(Y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute DVARS (temporal derivative of voxelwise variance).

    Parameters
    ----------
    Y : NDArray
        Data matrix, shape ``(n, V)``.

    Returns
    -------
    NDArray
        DVARS values, shape ``(n,)``.  The first value is 0.
    """
    diff = np.diff(Y, axis=0)
    dvars = np.sqrt(np.mean(diff ** 2, axis=1))
    return np.concatenate([[0.0], dvars])


def dvars_weights(
    dvars: NDArray[np.float64],
    method: str = "inverse_squared",
    threshold: float = 1.5,
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

    Returns
    -------
    NDArray
        Weights in ``[0, 1]``, shape ``(n,)``.
    """
    # Robust scale: MAD
    med = np.median(dvars[dvars > 0]) if np.any(dvars > 0) else 1.0
    mad = np.median(np.abs(dvars - med)) * 1.4826
    if mad < 1e-10:
        return np.ones_like(dvars)

    z = (dvars - med) / mad

    if method == "inverse_squared":
        w = 1.0 / (1.0 + np.maximum(z / threshold, 0.0) ** 2)
    elif method == "soft_threshold":
        w = np.where(z <= threshold, 1.0, threshold / np.maximum(np.abs(z), 1e-10))
    elif method == "tukey":
        # Tukey bisquare
        c = threshold
        w = np.where(np.abs(z) <= c, (1.0 - (z / c) ** 2) ** 2, 0.0)
    else:
        raise ValueError(f"Unknown method: {method}")

    return np.clip(w, 0.0, 1.0)


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
    keep = ~censor
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
