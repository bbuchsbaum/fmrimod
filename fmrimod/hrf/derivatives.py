"""Derivative methods for HRF objects."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import HRF


def _finite_difference_derivative(func, x0, dx=1e-4, n=1):
    """Numerical derivative via Richardson extrapolation.

    Uses Richardson extrapolation (two step sizes) for improved accuracy,
    similar to R's ``numDeriv::grad``.  Falls back to simple central
    differences for orders > 1.
    """
    if n == 0:
        return func(x0)
    if n == 1:
        # Richardson extrapolation: combine two central-difference estimates
        # with step sizes dx and dx/2 to cancel the O(dx^2) error term.
        d1 = (func(x0 + dx) - func(x0 - dx)) / (2 * dx)
        dx2 = dx / 2
        d2 = (func(x0 + dx2) - func(x0 - dx2)) / (2 * dx2)
        # Richardson: (4*d2 - d1) / 3 gives O(dx^4) accuracy
        return (4 * d2 - d1) / 3
    # Higher orders: recurse with central differences
    return (
        _finite_difference_derivative(func, x0 + dx, dx=dx, n=n - 1)
        - _finite_difference_derivative(func, x0 - dx, dx=dx, n=n - 1)
    ) / (2 * dx)


def derivative(func, x0, dx=1e-6, n=1, order=3):
    """Compute the *n*-th derivative of *func* at *x0*.

    Uses an internal recursive central-difference implementation for all
    orders.
    """
    # Keep ``order`` for API compatibility with scipy.misc.derivative.
    _ = order
    return _finite_difference_derivative(func, x0, dx=dx, n=n)


def deriv(hrf: HRF, t: ArrayLike, method: Literal["auto", "numeric", "analytic"] = "auto", order: int = 1) -> NDArray[np.float64]:
    """Compute the *n*-th derivative of an HRF at given time points.

    Args:
        hrf: HRF object to differentiate.
        t: Time points at which to evaluate the derivative.
        method: ``'auto'``, ``'numeric'``, or ``'analytic'``.
        order: Derivative order (default 1).

    Returns:
        Array of derivative values.  Shape is ``(len(t),)`` for single
        basis, or ``(len(t), nbasis)`` for multi-basis HRFs.

    Raises:
        ValueError: If ``method='analytic'`` but HRF has no analytic derivative.
    """
    t = np.asarray(t, dtype=np.float64)

    has_analytic = hasattr(hrf, '_derivative')

    if method == "analytic" and not has_analytic:
        raise ValueError(f"HRF '{hrf.name}' does not have an analytic derivative")

    if method == "auto":
        method = "analytic" if has_analytic else "numeric"

    if method == "analytic":
        if order == 1:
            return hrf._derivative(t)
        elif order == 2 and hasattr(hrf, '_second_derivative'):
            return hrf._second_derivative(t)
        else:
            # Fall back to numeric for higher orders
            return _numeric_derivative(hrf, t, order=order)
    else:
        return _numeric_derivative(hrf, t, order=order)


def _numeric_derivative(hrf: HRF, t: ArrayLike, order: int = 1) -> NDArray[np.float64]:
    """Compute *n*-th derivative using numerical differentiation."""
    t = np.asarray(t, dtype=np.float64)
    n_basis = hrf.nbasis
    result = np.zeros((len(t), n_basis))

    for j in range(n_basis):
        if n_basis > 1:
            def f(time, _j=j):
                val = hrf(np.array([time]))
                return val[0, _j]
        else:
            def f(time):
                val = hrf(np.array([time]))
                return val[0] if val.ndim == 1 else val[0, 0]

        for i, ti in enumerate(t):
            result[i, j] = derivative(f, ti, dx=1e-6, n=order)

    if n_basis == 1:
        return result[:, 0]
    return result


# Analytic derivative functions for specific HRF types

# Shared constant with functions.py
from .functions import _SPM_C


def spmg1_derivative(t: ArrayLike, p1: float = 5, p2: float = 15, a1: float = 0.0833) -> NDArray[np.float64]:
    """Analytic derivative of SPM canonical HRF (SPMG1).

    Args:
        t: Time points
        p1: Shape parameter for positive gamma
        p2: Shape parameter for negative gamma
        a1: Amplitude scaling factor

    Returns:
        Derivative values at time points t
    """
    t = np.asarray(t, dtype=np.float64)
    ret = np.zeros_like(t)
    pos = t >= 0

    if np.any(pos):
        t_pos = t[pos]
        ret[pos] = np.exp(-t_pos) * (
            a1 * t_pos**(p1 - 1) * (p1 - t_pos) -
            _SPM_C * t_pos**(p2 - 1) * (p2 - t_pos)
        )

    return ret


def spmg1_second_derivative(t: ArrayLike, p1: float = 5, p2: float = 15, a1: float = 0.0833) -> NDArray[np.float64]:
    """Analytic second derivative of SPM canonical HRF.

    Args:
        t: Time points
        p1: Shape parameter for positive gamma
        p2: Shape parameter for negative gamma
        a1: Amplitude scaling factor

    Returns:
        Second derivative values at time points t
    """
    t = np.asarray(t, dtype=np.float64)
    ret = np.zeros_like(t)
    pos = t >= 0

    if np.any(pos):
        t_pos = t[pos]
        # Components and their derivatives
        d1 = a1 * t_pos**(p1 - 1) * (p1 - t_pos)
        d2 = _SPM_C * t_pos**(p2 - 1) * (p2 - t_pos)
        d1_prime = a1 * ((p1 - 1) * t_pos**(p1 - 2) * (p1 - t_pos) - t_pos**(p1 - 1))
        d2_prime = _SPM_C * ((p2 - 1) * t_pos**(p2 - 2) * (p2 - t_pos) - t_pos**(p2 - 1))

        ret[pos] = np.exp(-t_pos) * (d1_prime - d2_prime - (d1 - d2))

    return ret
