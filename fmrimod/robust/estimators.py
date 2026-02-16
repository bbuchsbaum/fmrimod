"""M-estimator weight functions for robust regression.

Provides Huber and Tukey bisquare weight functions used in the
IRLS algorithm.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mad_scale(residuals: NDArray[np.float64], axis: int = 0) -> NDArray[np.float64]:
    """Compute MAD (median absolute deviation) scale estimate.

    Parameters
    ----------
    residuals : NDArray
        Residual array.
    axis : int
        Axis along which to compute.

    Returns
    -------
    NDArray
        Scale estimate (MAD * 1.4826 for consistency with normal).
    """
    med = np.median(residuals, axis=axis, keepdims=True)
    mad = np.median(np.abs(residuals - med), axis=axis)
    return mad * 1.4826


def huber_weights(
    residuals: NDArray[np.float64],
    scale: NDArray[np.float64],
    k: float = 1.345,
) -> NDArray[np.float64]:
    """Compute Huber weights.

    Parameters
    ----------
    residuals : NDArray
        Residual matrix, shape ``(n, V)``.
    scale : NDArray
        Scale estimate, shape ``(V,)`` or scalar.
    k : float
        Huber tuning constant.

    Returns
    -------
    NDArray
        Weights in ``(0, 1]``, shape ``(n, V)``.
    """
    scale = np.maximum(scale, 1e-10)
    u = residuals / scale[np.newaxis, :]
    w = np.where(np.abs(u) <= k, 1.0, k / np.maximum(np.abs(u), 1e-10))
    return w


def bisquare_weights(
    residuals: NDArray[np.float64],
    scale: NDArray[np.float64],
    c: float = 4.685,
) -> NDArray[np.float64]:
    """Compute Tukey bisquare weights.

    Parameters
    ----------
    residuals : NDArray
        Residual matrix, shape ``(n, V)``.
    scale : NDArray
        Scale estimate, shape ``(V,)`` or scalar.
    c : float
        Tukey bisquare tuning constant.

    Returns
    -------
    NDArray
        Weights in ``[0, 1]``, shape ``(n, V)``.
    """
    scale = np.maximum(scale, 1e-10)
    u = residuals / scale[np.newaxis, :]
    w = np.where(np.abs(u) <= c, (1.0 - (u / c) ** 2) ** 2, 0.0)
    return w
