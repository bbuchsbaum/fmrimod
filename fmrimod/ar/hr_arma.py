"""Hannan-Rissanen ARMA(p,q) estimation.

Ports ``hr_arma.R`` and ``fmriAR_hr.cpp`` into pure NumPy/SciPy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

from .numhelpers import (
    enforce_invertible_ma,
    enforce_stationary_ar,
    levinson_durbin,
)


def _lag_matrix(x: NDArray, k: int) -> NDArray:
    """Build a matrix of lagged values.

    Parameters
    ----------
    x : NDArray
        1-D input, shape ``(n,)``.
    k : int
        Number of lags.

    Returns
    -------
    NDArray
        Matrix shape ``(n, k)`` where column *j* is ``x`` shifted by
        ``j+1`` steps.  Leading rows are filled with 0.
    """
    n = len(x)
    if k <= 0:
        return np.empty((n, 0), dtype=np.float64)
    M = np.zeros((n, k), dtype=np.float64)
    for j in range(k):
        lag = j + 1
        M[lag:, j] = x[: n - lag]
    return M


def _arma_innovations(
    y: NDArray, phi: NDArray, theta: NDArray
) -> NDArray:
    """Compute ARMA innovations (residuals) via filtering.

    Applies the ARMA filter::

        e[t] = y[t] - phi_1*y[t-1] - ... - phi_p*y[t-p]
                     - theta_1*e[t-1] - ... - theta_q*e[t-q]

    Using ``scipy.signal.lfilter``.

    Parameters
    ----------
    y : NDArray
        Input series, shape ``(n,)``.
    phi : NDArray
        AR coefficients, shape ``(p,)``.
    theta : NDArray
        MA coefficients, shape ``(q,)``.

    Returns
    -------
    NDArray
        Innovation series, shape ``(n,)``.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64).ravel()
    theta = np.asarray(theta, dtype=np.float64).ravel()

    # AR polynomial: b = [1, -phi_1, -phi_2, ...]
    b = np.concatenate([[1.0], -phi])
    # MA polynomial: a = [1, theta_1, theta_2, ...]
    a = np.concatenate([[1.0], theta])

    return lfilter(b, a, y)


def hr_arma(
    y: NDArray,
    p: int,
    q: int,
    *,
    p_big: Optional[int] = None,
    step1: str = "yw",
    n_iter: int = 0,
    bound: float = 0.99,
    enforce: bool = True,
) -> dict:
    """Hannan-Rissanen two-step ARMA estimation.

    Parameters
    ----------
    y : NDArray
        Input series, shape ``(n,)``.
    p : int
        AR order.
    q : int
        MA order.
    p_big : int, optional
        High-order AR for preliminary residuals (default: auto).
    step1 : str
        ``"yw"`` (Yule-Walker) or ``"burg"`` for step-1 AR fit.
    n_iter : int
        Number of refinement iterations after the initial fit.
    bound : float
        PACF clipping bound for stationarity enforcement.
    enforce : bool
        Whether to enforce stationarity/invertibility.

    Returns
    -------
    dict
        ``{"phi", "theta", "sigma2", "order", "method", "p_big", "iterations"}``
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y - y.mean()
    n = len(y)
    if n < 10:
        raise ValueError("Series too short for HR estimation")

    if p_big is None:
        p_big = max(8, p + q + 5, int(np.ceil(10 * np.log10(n))))
        p_big = min(p_big, max(2, n - 2), 40)

    # Step 1: high-order AR to get preliminary residuals
    # Compute autocovariance
    gamma = np.zeros(p_big + 1)
    for lag in range(p_big + 1):
        if lag >= n:
            break
        gamma[lag] = np.sum(y[: n - lag] * y[lag:]) / n

    if gamma[0] < 1e-15:
        phi_init = np.zeros(p_big, dtype=np.float64)
    else:
        phi_init, _ = levinson_durbin(gamma[: p_big + 1], p_big)

    # Compute initial residuals from high-order AR
    ehat = _arma_innovations(y, phi_init, np.array([]))

    # Initialize
    phi = np.zeros(p, dtype=np.float64) if p > 0 else np.array([], dtype=np.float64)
    theta = np.zeros(q, dtype=np.float64) if q > 0 else np.array([], dtype=np.float64)

    # Step 2 + refinement iterations
    for _iteration in range(n_iter + 1):
        Ylags = _lag_matrix(y, p)
        Elags = _lag_matrix(ehat, q)
        m = max(p, q)
        idx = slice(m, n)
        Z = np.hstack([Ylags[idx], Elags[idx]])
        z_y = y[idx]

        if Z.shape[0] < Z.shape[1] + 1:
            raise ValueError(
                f"Not enough data for ARMA({p},{q}): "
                f"n_eff={Z.shape[0]}, n_params={Z.shape[1]}"
            )

        coef, _, _, _ = np.linalg.lstsq(Z, z_y, rcond=None)

        if p > 0:
            phi = coef[:p]
        if q > 0:
            theta = coef[p : p + q]

        # Recompute innovations for next iteration
        ehat = _arma_innovations(y, phi, theta)

    # Enforce stationarity/invertibility
    if enforce:
        if len(phi):
            phi = enforce_stationary_ar(phi, bound=bound)
        if len(theta):
            theta = enforce_invertible_ma(theta)

    sigma2 = float(np.mean(ehat ** 2))

    return {
        "phi": phi,
        "theta": theta,
        "sigma2": sigma2,
        "order": (int(p), int(q)),
        "method": "hr",
        "p_big": int(p_big),
        "iterations": int(n_iter),
    }
