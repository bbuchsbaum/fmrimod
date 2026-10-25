"""Derivative methods for HRF objects."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import HRF


@runtime_checkable
class HasAnalyticDerivative(Protocol):
    """HRFs that provide a closed-form first derivative.

    Concrete HRF kinds implement ``_derivative(t)`` when an analytic
    form exists (e.g., the SPM canonical family). The ``deriv``
    function checks for this protocol via ``isinstance`` rather than
    ``hasattr`` so static type-checkers see the contract and accidental
    attribute collisions on unrelated objects cannot route to the
    analytic branch.
    """

    def _derivative(self, t: ArrayLike) -> NDArray[np.float64]:
        ...


@runtime_checkable
class HasAnalyticSecondDerivative(Protocol):
    """HRFs that provide a closed-form second derivative.

    Parallel to :class:`HasAnalyticDerivative`. No fmrimod-shipped HRF
    currently implements this; the protocol is the extension seam for
    custom HRFs.
    """

    def _second_derivative(self, t: ArrayLike) -> NDArray[np.float64]:
        ...


def _finite_difference_derivative(
    func: Any, x0: Any, dx: float = 1e-4, n: int = 1
) -> Any:
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


def derivative(
    func: Any, x0: Any, dx: float = 1e-6, n: int = 1, order: int = 3
) -> Any:
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

    has_analytic = isinstance(hrf, HasAnalyticDerivative)

    if method == "analytic" and not has_analytic:
        raise ValueError(f"HRF '{hrf.name}' does not have an analytic derivative")

    if method == "auto":
        method = "analytic" if has_analytic else "numeric"

    if method == "analytic":
        assert isinstance(hrf, HasAnalyticDerivative)
        if order == 1:
            return hrf._derivative(t)
        if order == 2 and isinstance(hrf, HasAnalyticSecondDerivative):
            return hrf._second_derivative(t)
        # Fall back to numeric for higher orders
        return _numeric_derivative(hrf, t, order=order)
    return _numeric_derivative(hrf, t, order=order)


def _numeric_derivative(hrf: HRF, t: ArrayLike, order: int = 1) -> NDArray[np.float64]:
    """Compute *n*-th derivative using numerical differentiation."""
    t = np.asarray(t, dtype=np.float64)
    n_basis = hrf.nbasis
    result = np.zeros((len(t), n_basis))

    for j in range(n_basis):
        f: Any
        if n_basis > 1:
            def f(time: Any, _j: int = j) -> Any:
                val = hrf(np.array([time]))
                return val[0, _j]
        else:
            def f(time: Any, _j: int = j) -> Any:  # noqa: F811
                val = hrf(np.array([time]))
                return val[0] if val.ndim == 1 else val[0, 0]

        for i, ti in enumerate(t):
            result[i, j] = derivative(f, ti, dx=1e-6, n=order)

    if n_basis == 1:
        return result[:, 0]
    return result


# Analytic derivative functions for the SPM canonical family


def _gamma_pdf_and_log_slope(
    t: NDArray[np.float64], shape: float, scale: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gamma pdf and ``d log g / dt = (shape - 1) / t - 1 / scale`` at ``t > 0``."""
    from scipy import stats

    return stats.gamma.pdf(t, shape, scale=scale), (shape - 1.0) / t - 1.0 / scale


def _spm_component_derivative(
    t: NDArray[np.float64], shape: float, scale: float, order: int
) -> NDArray[np.float64]:
    g, slope = _gamma_pdf_and_log_slope(t, shape, scale)
    if order == 1:
        return g * slope
    return g * (slope**2 - (shape - 1.0) / t**2)


def spmg1_derivative(
    t: ArrayLike,
    *,
    delay: float = 6.0,
    undershoot: float = 16.0,
    dispersion: float = 1.0,
    u_dispersion: float = 1.0,
    ratio: float = 1.0 / 6.0,
    order: Literal[1, 2] = 1,
) -> NDArray[np.float64]:
    """Analytic time derivative of :func:`~fmrimod.hrf.functions.spm_canonical`.

    ``order=1`` gives ``dh/dt`` and ``order=2`` gives ``d2h/dt2``. At the
    defaults these are R fmrihrf's ``hrf_spmg1_deriv`` and
    ``hrf_spmg1_second_deriv`` (``P1=5, P2=15, A1=1/120``), and ``order=1``
    is the second column of :class:`~fmrimod.hrf.spm_hrf.SPMG2_HRF`.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    t = np.asarray(t, dtype=np.float64)
    ret = np.zeros_like(t)
    pos = t > 0
    if np.any(pos):
        t_pos = t[pos]
        peak = _spm_component_derivative(t_pos, delay / dispersion, dispersion, order)
        under = _spm_component_derivative(
            t_pos, undershoot / u_dispersion, u_dispersion, order
        )
        ret[pos] = peak - ratio * under
    return ret


def spmg1_second_derivative(
    t: ArrayLike,
    *,
    delay: float = 6.0,
    undershoot: float = 16.0,
    dispersion: float = 1.0,
    u_dispersion: float = 1.0,
    ratio: float = 1.0 / 6.0,
) -> NDArray[np.float64]:
    """Analytic ``d2h/dt2`` of the SPM canonical (R ``hrf_spmg1_second_deriv``)."""
    return spmg1_derivative(
        t, delay=delay, undershoot=undershoot, dispersion=dispersion,
        u_dispersion=u_dispersion, ratio=ratio, order=2,
    )


# SPM ``spm_get_bf.m`` (and Nilearn, and R fmrihrf) use dx = 0.01 for the
# dispersion perturbation.
_SPM_DISP_DX = 0.01


def spmg1_dispersion_derivative(
    t: ArrayLike,
    *,
    delay: float = 6.0,
    undershoot: float = 16.0,
    dispersion: float = 1.0,
    u_dispersion: float = 1.0,
    ratio: float = 1.0 / 6.0,
    dx: float = _SPM_DISP_DX,
) -> NDArray[np.float64]:
    """SPM dispersion basis column ``(h(disp) - h(disp + dx)) / dx``.

    Only the peak gamma's dispersion is perturbed, so the undershoot
    cancels. This is SPM ``spm_get_bf``'s and Nilearn's
    ``spm_dispersion_derivative``, and at the defaults R fmrihrf's
    ``hrf_spmg1_dispersion_deriv`` (the third column of ``HRF_SPMG3``).
    """
    from .functions import spm_canonical as _canon

    params = {
        "delay": delay, "undershoot": undershoot,
        "u_dispersion": u_dispersion, "ratio": ratio,
    }
    h0 = _canon(t, dispersion=dispersion, **params)
    h1 = _canon(t, dispersion=dispersion + dx, **params)
    return (h0 - h1) / dx


def spmg1_dispersion_time_derivative(
    t: ArrayLike,
    *,
    delay: float = 6.0,
    undershoot: float = 16.0,
    dispersion: float = 1.0,
    u_dispersion: float = 1.0,
    ratio: float = 1.0 / 6.0,
    dx: float = _SPM_DISP_DX,
) -> NDArray[np.float64]:
    """Time derivative of :func:`spmg1_dispersion_derivative`.

    R fmrihrf's ``hrf_spmg1_dispersion_time_deriv``: the third column of
    ``deriv(HRF_SPMG3)``.
    """
    params = {
        "delay": delay, "undershoot": undershoot,
        "u_dispersion": u_dispersion, "ratio": ratio,
    }
    d0 = spmg1_derivative(t, dispersion=dispersion, **params)
    d1 = spmg1_derivative(t, dispersion=dispersion + dx, **params)
    return (d0 - d1) / dx
