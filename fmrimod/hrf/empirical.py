"""Empirical HRF functions."""

from __future__ import annotations

from typing import Optional
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import interpolate

from .core import FunctionHRF, HRF


def empirical_hrf(
    t: ArrayLike,
    y: ArrayLike,
    name: str = "empirical_hrf",
    span: Optional[float] = None
) -> HRF:
    """Generate an empirical HRF from data points.
    
    Creates an HRF by interpolating provided time points and values.
    
    Args:
        t: Time points
        y: HRF values at time points
        name: Name for the HRF
        span: Temporal span (if None, uses max(t))
        
    Returns:
        HRF object
        
    Examples:
        >>> t_points = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        >>> y_values = np.array([0, 0.2, 0.8, 1.0, 0.7, 0.3, 0, -0.1, -0.05, 0, 0])
        >>> emp_hrf = empirical_hrf(t_points, y_values)
        >>> 
        >>> # Evaluate at new time points
        >>> new_times = np.linspace(0, 20, 100)
        >>> response = emp_hrf(new_times)
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    if len(t) != len(y):
        raise ValueError("t and y must have the same length")
    
    if len(t) < 2:
        raise ValueError("Need at least 2 points to create empirical HRF")
    
    # Sort by time
    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]
    y_sorted = y[sort_idx]
    
    # Determine span
    if span is None:
        span = float(t_sorted[-1])
    
    # Create interpolation function
    # Use linear interpolation (matching R's approx() default)
    interp_func = interpolate.interp1d(
        t_sorted, y_sorted,
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )
    
    # Create HRF function
    def hrf_func(t_eval):
        t_eval = np.asarray(t_eval, dtype=np.float64)
        return interp_func(t_eval)
    
    # Return as FunctionHRF
    return FunctionHRF(
        func=hrf_func,
        name=name,
        nbasis=1,
        span=span,
        params={"t": t_sorted.tolist(), "y": y_sorted.tolist()},
        param_names=["t", "y"]
    )


def gen_empirical_hrf(
    t: ArrayLike,
    y: ArrayLike,
    name: str = "empirical_hrf",
    span: Optional[float] = None,
) -> HRF:
    """R-compatible alias for :func:`empirical_hrf`."""
    return empirical_hrf(t, y, name=name, span=span)
