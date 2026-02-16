"""Numeric helpers for AR/ARMA estimation.

Ports ``pacf_helpers.R``, ``ma_invertibility.R``, and the C++ Levinson-Durbin
solver into pure NumPy/SciPy.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_toeplitz


# ---------------------------------------------------------------------------
# PACF <-> AR conversion (Durbin-Levinson recursion)
# ---------------------------------------------------------------------------

def pacf_to_ar(kappa: NDArray) -> NDArray:
    """Convert partial autocorrelations to AR coefficients.

    Uses the Durbin-Levinson forward recursion.

    Parameters
    ----------
    kappa : NDArray
        PACF values, shape ``(p,)``.

    Returns
    -------
    NDArray
        AR coefficients, shape ``(p,)``.
    """
    kappa = np.asarray(kappa, dtype=np.float64).ravel()
    p = len(kappa)
    if p == 0:
        return np.array([], dtype=np.float64)

    # Iterative Durbin-Levinson
    phi_prev = np.array([kappa[0]])
    for m in range(1, p):
        km = kappa[m]
        phi_new = np.empty(m + 1)
        phi_new[:m] = phi_prev - km * phi_prev[::-1]
        phi_new[m] = km
        phi_prev = phi_new

    return phi_prev


def ar_to_pacf(phi: NDArray, eps: float = 1e-12) -> NDArray:
    """Convert AR coefficients to partial autocorrelations.

    Inverse of :func:`pacf_to_ar` via backward Levinson recursion.

    Parameters
    ----------
    phi : NDArray
        AR coefficients, shape ``(p,)``.
    eps : float
        Floor for denominator to avoid division by zero.

    Returns
    -------
    NDArray
        PACF values, shape ``(p,)``.
    """
    phi = np.asarray(phi, dtype=np.float64).ravel()
    p = len(phi)
    if p == 0:
        return np.array([], dtype=np.float64)

    a = phi.copy()
    kappa = np.empty(p)
    for m in range(p, 0, -1):
        km = a[m - 1]
        kappa[m - 1] = km
        if m == 1:
            break
        den = 1.0 - km * km
        if den < eps:
            den = eps
        a_new = np.empty(m - 1)
        for j in range(m - 1):
            a_new[j] = (a[j] + km * a[m - 2 - j]) / den
        a = a_new

    return kappa


# ---------------------------------------------------------------------------
# Stationarity / invertibility enforcement
# ---------------------------------------------------------------------------

def enforce_stationary_ar(
    phi: NDArray, bound: float = 0.99
) -> NDArray:
    """Enforce stationarity by clipping PACF values.

    Converts AR coefficients to PACF, clips each to ``[-bound, bound]``,
    then converts back.  This is more principled than the root-shrinkage
    heuristic used in the previous implementation.

    Parameters
    ----------
    phi : NDArray
        AR coefficients, shape ``(p,)``.
    bound : float
        PACF clipping bound (default 0.99).

    Returns
    -------
    NDArray
        Stationary AR coefficients, shape ``(p,)``.
    """
    phi = np.asarray(phi, dtype=np.float64).ravel()
    if len(phi) == 0:
        return phi
    kap = ar_to_pacf(phi)
    kap = np.clip(kap, -bound, bound)
    return pacf_to_ar(kap)


def enforce_invertible_ma(
    theta: NDArray, tol: float = 1e-8
) -> NDArray:
    """Enforce MA invertibility by reflecting roots inside the unit circle.

    Parameters
    ----------
    theta : NDArray
        MA coefficients, shape ``(q,)``.
    tol : float
        Tolerance for unit-circle boundary.

    Returns
    -------
    NDArray
        Invertible MA coefficients, shape ``(q,)``.
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    q = len(theta)
    if q == 0:
        return theta

    # Roots of 1 + theta[0]*z + theta[1]*z^2 + ...
    poly = np.concatenate([[1.0], theta])
    roots = np.roots(poly[::-1])  # np.roots expects descending power order
    if len(roots) == 0:
        return theta

    # Reflect roots inside unit circle
    modified = False
    for i in range(len(roots)):
        if np.abs(roots[i]) <= 1.0 + tol:
            roots[i] = 1.0 / np.conj(roots[i])
            modified = True

    if not modified:
        return theta

    # Reconstruct polynomial from roots: prod_i (1 - z/r_i)
    # But we need coefficients of 1 + c1*z + c2*z^2 + ...
    coeffs = np.array([1.0 + 0j])
    for r in roots:
        coeffs = np.convolve(coeffs, np.array([1.0 + 0j, -1.0 / r]))

    return np.real(coeffs[1:])


# ---------------------------------------------------------------------------
# Levinson-Durbin / Yule-Walker
# ---------------------------------------------------------------------------

def levinson_durbin(
    gamma: NDArray, p: int
) -> Tuple[NDArray, float]:
    """Solve Yule-Walker equations via Levinson-Durbin.

    Uses ``scipy.linalg.solve_toeplitz`` for the symmetric positive-definite
    Toeplitz system.

    Parameters
    ----------
    gamma : NDArray
        Autocovariance values ``gamma[0], gamma[1], ..., gamma[p]``.
    p : int
        AR order.

    Returns
    -------
    phi : NDArray
        AR coefficients, shape ``(p,)``.
    sigma2 : float
        Innovation variance.
    """
    gamma = np.asarray(gamma, dtype=np.float64).ravel()
    if p <= 0 or len(gamma) < p + 1 or gamma[0] < 1e-15:
        return np.zeros(max(p, 0), dtype=np.float64), float(gamma[0]) if len(gamma) else 0.0

    # solve_toeplitz solves T @ x = b where T = toeplitz(gamma[0:p])
    phi = solve_toeplitz(gamma[:p], gamma[1 : p + 1])

    # Innovation variance: gamma[0] - phi . gamma[1:p+1]
    sigma2 = float(gamma[0] - np.dot(phi, gamma[1 : p + 1]))
    sigma2 = max(sigma2, 1e-15)

    return phi, sigma2


# ---------------------------------------------------------------------------
# Autocovariance functions (segment/run-aware)
# ---------------------------------------------------------------------------

def segmented_acvf(
    y: NDArray,
    seg_starts: NDArray,
    max_lag: int,
    unbiased: bool = False,
    center: bool = True,
) -> NDArray:
    """Compute autocovariance from a segmented 1-D series.

    Each segment is treated independently (no cross-segment products).

    Parameters
    ----------
    y : NDArray
        1-D time series, shape ``(n,)``.
    seg_starts : NDArray
        0-based segment start indices (must include 0).
    max_lag : int
        Maximum lag to compute.
    unbiased : bool
        If ``True``, divide by number of lag-*k* pairs.
    center : bool
        If ``True``, center each segment before computing.

    Returns
    -------
    NDArray
        Autocovariance values, shape ``(max_lag + 1,)``.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    seg_starts = np.asarray(seg_starts, dtype=np.intp).ravel()
    seg_starts = np.sort(np.unique(seg_starts))

    # Build segment boundaries
    ends = np.append(seg_starts[1:], n)

    gamma = np.zeros(max_lag + 1)
    counts = np.zeros(max_lag + 1)

    for s_start, s_end in zip(seg_starts, ends):
        seg = y[s_start:s_end].copy()
        seg_len = len(seg)
        if seg_len < 2:
            continue
        if center:
            seg = seg - seg.mean()
        for lag in range(min(max_lag + 1, seg_len)):
            products = seg[: seg_len - lag] * seg[lag:]
            gamma[lag] += products.sum()
            counts[lag] += seg_len - lag

    if unbiased:
        valid = counts > 0
        gamma[valid] /= counts[valid]
        gamma[~valid] = 0.0
    else:
        total = counts[0] if counts[0] > 0 else 1.0
        gamma /= total

    return gamma


def run_avg_acvf(
    mat: NDArray, max_lag: int, run_starts: Optional[NDArray] = None
) -> NDArray:
    """Compute column-pooled autocovariance from a matrix.

    Pools ACVF across columns (voxels) by computing the mean
    cross-product at each lag.

    Parameters
    ----------
    mat : NDArray
        Data matrix, shape ``(n, V)``.
    max_lag : int
        Maximum lag.
    run_starts : NDArray or None
        0-based run start indices.  If ``None``, treats the whole
        series as one segment.

    Returns
    -------
    NDArray
        Pooled autocovariance, shape ``(max_lag + 1,)``.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat[:, np.newaxis]
    n, v = mat.shape

    if run_starts is None:
        run_starts = np.array([0], dtype=np.intp)

    # Center each column
    mat = mat - mat.mean(axis=0, keepdims=True)

    gamma = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag >= n:
            break
        gamma[lag] = np.sum(mat[: n - lag] * mat[lag:]) / (n * v)

    return gamma
