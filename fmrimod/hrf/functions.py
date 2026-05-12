"""HRF function implementations."""

from __future__ import annotations

import warnings
from typing import Literal, Optional, Union, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats, special, interpolate

# Constant used in SPM canonical HRF parameterization
_SPM_C = 1.274527e-13


def hrf_ident(t: ArrayLike) -> NDArray[np.float64]:
    """Identity (delta function) HRF.

    Returns 1 when t == 0, 0 elsewhere. Represents a Dirac delta-like
    impulse response (sampled).

    Args:
        t: Time points in seconds

    Returns:
        HRF values at time points t (1 at t=0, 0 elsewhere)
    """
    t = np.asarray(t, dtype=np.float64)
    return np.where(t == 0, 1.0, 0.0)


def spm_canonical(
    t: ArrayLike,
    p1: float = 5.0,
    p2: float = 15.0,
    a1: float = 0.0833,
) -> NDArray[np.float64]:
    """SPM canonical hemodynamic response function.

    Implements the canonical HRF as defined in SPM using the double gamma
    parameterization: exp(-t) * (A1*t^P1 - C*t^P2) where C = 1.274527e-13.

    Args:
        t: Time points in seconds
        p1: First exponent parameter (default: 5)
        p2: Second exponent parameter (default: 15)
        a1: Amplitude scaling factor (default: 0.0833)

    Returns:
        HRF values at time points t
    """
    t = np.asarray(t, dtype=np.float64)
    result = np.zeros_like(t)
    mask = t >= 0
    if np.any(mask):
        t_pos = t[mask]
        result[mask] = np.exp(-t_pos) * (a1 * t_pos**p1 - _SPM_C * t_pos**p2)
    return result


def gamma_hrf(
    t: ArrayLike,
    shape: float = 6.0,
    rate: float = 1.0,
) -> NDArray[np.float64]:
    """Gamma density hemodynamic response function.
    
    Args:
        t: Time points in seconds
        shape: Shape parameter of gamma distribution
        rate: Rate parameter of gamma distribution
        
    Returns:
        HRF values at time points t
    """
    t = np.asarray(t, dtype=np.float64)
    
    # Set negative times to zero
    result = np.zeros_like(t)
    mask = t >= 0
    
    if np.any(mask):
        # Note: scipy uses scale = 1/rate parameterization
        result[mask] = stats.gamma.pdf(t[mask], a=shape, scale=1.0/rate)
    
    return result


def gaussian_hrf(
    t: ArrayLike,
    mean: float = 6.0,
    sd: float = 2.0,
) -> NDArray[np.float64]:
    """Gaussian hemodynamic response function.
    
    Args:
        t: Time points in seconds
        mean: Mean of the Gaussian
        sd: Standard deviation of the Gaussian
        
    Returns:
        HRF values at time points t
    """
    t = np.asarray(t, dtype=np.float64)
    return stats.norm.pdf(t, loc=mean, scale=sd)


def bspline_hrf(
    t: ArrayLike,
    n_basis: int = 5,
    degree: int = 3,
    span: float = 24.0,
) -> NDArray[np.float64]:
    """B-spline basis set for HRF.

    Uses quantile-based interior knot placement matching R's splines::bs,
    where knots are placed at quantile positions of a uniform grid over
    the span.

    Args:
        t: Time points in seconds
        n_basis: Number of basis functions
        degree: Degree of B-splines (3 for cubic)
        span: Temporal span of the basis set

    Returns:
        Matrix of B-spline basis functions evaluated at t
    """
    t = np.asarray(t, dtype=np.float64)

    # Mirror R hrf_bspline_generator(): evaluate only inside [0, span],
    # keep zeros outside, and place interior knots from quantiles of t_valid.
    valid = (t >= 0) & (t <= span)
    if not np.any(valid):
        return np.zeros((len(t), int(n_basis)), dtype=np.float64)

    t_valid = t[valid]
    ord_ = degree + 1
    n_interior = max(0, int(n_basis) - ord_ + 1)

    if n_interior > 0:
        if degree == 1:
            # Keep tent basis stable as a partition of unity by using fixed,
            # evenly spaced interior knots over the full support.
            interior_knots = np.linspace(0.0, float(span), n_interior + 2)[1:-1]
        else:
            # Match R hrf_bspline_generator() default knot placement:
            # quantiles of the in-support evaluation grid.
            probs = np.arange(1, n_interior + 1, dtype=np.float64) / (n_interior + 1)
            interior_knots = np.quantile(t_valid, probs)
    else:
        interior_knots = np.array([], dtype=np.float64)

    knots = np.concatenate([
        np.repeat(0.0, ord_),
        interior_knots.astype(np.float64),
        np.repeat(float(span), ord_),
    ])

    # Build the full basis then drop the first column (intercept=FALSE),
    # matching splines::bs(..., intercept = FALSE).
    n_basis_full = len(knots) - degree - 1
    basis_full = np.zeros((len(t_valid), n_basis_full), dtype=np.float64)

    for i in range(n_basis_full):
        c = np.zeros(n_basis_full, dtype=np.float64)
        c[i] = 1.0
        bspl = interpolate.BSpline(knots, c, degree, extrapolate=False)
        basis_full[:, i] = np.nan_to_num(bspl(t_valid), nan=0.0)

    if n_basis_full > int(n_basis):
        basis_valid = basis_full[:, 1:(int(n_basis) + 1)]
    else:
        basis_valid = basis_full

    out = np.zeros((len(t), int(n_basis)), dtype=np.float64)
    out[valid, :basis_valid.shape[1]] = basis_valid
    return out


def daguerre_basis(
    t: ArrayLike,
    n_basis: int = 3,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Daguerre spherical basis functions using Laguerre polynomial recurrence.

    These are orthogonal polynomials on [0, Inf) with weight w(x) = x^2 * exp(-x).
    They naturally decay to zero, making them suitable for HRF modeling.

    Args:
        t: Time points in seconds
        n_basis: Number of basis functions to generate
        scale: Scale parameter for the time axis

    Returns:
        Matrix of basis functions, shape (len(t), n_basis), each column normalized
        to unit peak.
    """
    t = np.asarray(t, dtype=np.float64)
    x = t / scale

    basis = np.zeros((len(x), n_basis))

    # First basis function (n=0)
    basis[:, 0] = np.exp(-x / 2)

    if n_basis > 1:
        # Second basis function (n=1)
        basis[:, 1] = (1 - x) * np.exp(-x / 2)

    if n_basis > 2:
        # Higher order basis functions using three-term recurrence relation
        for n in range(2, n_basis):
            k = n
            basis[:, n] = ((2 * k - 1 - x) * basis[:, n - 1] - (k - 1) * basis[:, n - 2]) / k

    # Normalize each basis function to unit peak
    for i in range(n_basis):
        max_abs_val = np.max(np.abs(basis[:, i]))
        if max_abs_val > 1e-10:
            basis[:, i] = basis[:, i] / max_abs_val

    return basis


def fourier_hrf(
    t: ArrayLike,
    n_basis: int = 5,
    span: float = 24.0,
) -> NDArray[np.float64]:
    """Fourier basis set for HRF.

    Creates a basis set with alternating sine and cosine functions.
    Odd-indexed basis functions use sine, even-indexed use cosine,
    matching the R implementation.

    Args:
        t: Time points in seconds
        n_basis: Number of basis functions
        span: Temporal span of the basis set (default: 24)

    Returns:
        Matrix of Fourier basis functions evaluated at t, shape
        ``(len(t), n_basis)``.  Values outside [0, span] are zeroed out.
    """
    t = np.asarray(t, dtype=np.float64)
    in_support = (t >= 0) & (t <= span)

    # Frequencies: ceiling(1:n_basis / 2) -> [1, 1, 2, 2, 3, 3, ...]
    freqs = np.ceil(np.arange(1, n_basis + 1) / 2).astype(int)

    basis = np.zeros((len(t), n_basis))
    for k in range(n_basis):
        n = freqs[k]
        if (k + 1) % 2 == 1:  # odd k (1-based) -> sin
            basis[:, k] = np.sin(2 * np.pi * n * t / span)
        else:  # even k (1-based) -> cos
            basis[:, k] = np.cos(2 * np.pi * n * t / span)

    # Zero out values outside support
    basis[~in_support, :] = 0

    return basis


def fir_basis(
    t: ArrayLike,
    n_basis: int = 10,
    span: float = 20.0,
) -> NDArray[np.float64]:
    """Finite Impulse Response (FIR) basis set.

    Creates a basis of boxcar functions.

    Args:
        t: Time points in seconds
        n_basis: Number of basis functions
        span: Temporal span of the basis set

    Returns:
        Matrix of FIR basis functions evaluated at t
    """
    t = np.asarray(t, dtype=np.float64)

    # Create bin edges
    bin_width = span / n_basis
    bins = np.arange(n_basis + 1) * bin_width

    # Create basis functions
    basis = np.zeros((len(t), n_basis))

    for i in range(n_basis):
        # Boxcar function for this bin
        basis[:, i] = ((t >= bins[i]) & (t < bins[i + 1])).astype(float)

    # Handle edge case where t == span
    if n_basis > 0:
        basis[t == span, -1] = 1.0

    return basis


def mexhat_hrf(
    t: ArrayLike,
    sigma: float = 1.0,
    center: float = 6.0,
) -> NDArray[np.float64]:
    """Mexican hat (Ricker) wavelet HRF.
    
    Args:
        t: Time points in seconds
        sigma: Width parameter
        center: Center of the wavelet
        
    Returns:
        HRF values at time points t
    """
    t = np.asarray(t, dtype=np.float64)
    
    # Shift and scale time
    z = (t - center) / sigma
    
    # Mexican hat formula
    normalization = 2.0 / (np.sqrt(3 * sigma) * np.pi**0.25)
    return normalization * (1.0 - z**2) * np.exp(-z**2 / 2.0)


def sine_hrf(
    t: ArrayLike,
    frequency: float = 0.5,
    phase: float = 0.0,
) -> NDArray[np.float64]:
    """Sine wave HRF.
    
    Args:
        t: Time points in seconds
        frequency: Frequency in Hz
        phase: Phase shift in radians
        
    Returns:
        HRF values at time points t
    """
    t = np.asarray(t, dtype=np.float64)
    return np.sin(2 * np.pi * frequency * t + phase)


def half_cosine_hrf(
    t: ArrayLike,
    width: float = 10.0,
    center: float = 5.0,
) -> NDArray[np.float64]:
    """Half-cosine HRF.
    
    Args:
        t: Time points in seconds
        width: Width of the half-cosine
        center: Center of the function
        
    Returns:
        HRF values at time points t
    """
    t = np.asarray(t, dtype=np.float64)
    
    # Create half-cosine centered at 'center'
    result = np.zeros_like(t)
    
    # Define active region
    t_start = center - width / 2
    t_end = center + width / 2
    mask = (t >= t_start) & (t <= t_end)
    
    if np.any(mask):
        # Map to [0, pi] range
        scaled_t = (t[mask] - t_start) * np.pi / width
        result[mask] = 0.5 * (1 + np.cos(scaled_t))
    
    return result


def inv_logit_hrf(
    t: ArrayLike,
    center: float = 6.0,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Inverse logit (sigmoid) HRF.
    
    Args:
        t: Time points in seconds
        center: Center of the sigmoid
        scale: Scale parameter (smaller = steeper)
        
    Returns:
        HRF values at time points t
    """
    t = np.asarray(t, dtype=np.float64)
    return special.expit((t - center) / scale)


def lwu_hrf(
    t: ArrayLike,
    n_basis: int = 3,
    center: float = 6.0,
    width: float = 3.0,
) -> NDArray[np.float64]:
    """Lindquist-Wager-Ungerleider (LWU) basis functions.
    
    These are based on Taylor series expansion around the canonical HRF.
    
    Args:
        t: Time points in seconds
        n_basis: Number of basis functions (1-3)
        center: Center time for expansion
        width: Width parameter for derivatives
        
    Returns:
        Matrix of LWU basis functions evaluated at t
    """
    t = np.asarray(t, dtype=np.float64)
    
    if n_basis < 1 or n_basis > 3:
        raise ValueError("n_basis must be between 1 and 3")
    
    # Start with canonical HRF
    canonical = spm_canonical(t)
    
    if n_basis == 1:
        return canonical.reshape(-1, 1)
    
    # Create basis matrix
    basis = np.zeros((len(t), n_basis))
    basis[:, 0] = canonical
    
    # Add derivative terms
    if n_basis >= 2:
        # First derivative approximation
        dt = 0.01
        derivative1 = (spm_canonical(t + dt) - spm_canonical(t - dt)) / (2 * dt)
        basis[:, 1] = derivative1 * width
    
    if n_basis >= 3:
        # Second derivative approximation
        derivative2 = (spm_canonical(t + dt) - 2 * canonical + spm_canonical(t - dt)) / (dt**2)
        basis[:, 2] = derivative2 * (width**2) / 2
    
    return basis


def hrf_time(
    t: ArrayLike,
    max_time: float = 22.0,
) -> NDArray[np.float64]:
    """Linear time HRF.

    Simple linear function of time when t is between 0 and max_time.

    Args:
        t: Time points in seconds
        max_time: Maximum time point in the domain

    Returns:
        HRF values at time points
    """
    t = np.asarray(t, dtype=np.float64)
    return np.where((t > 0) & (t < max_time), t, 0.0)


def hrf_mexhat(
    t: ArrayLike,
    mean: float = 6.0,
    sd: float = 2.0
) -> NDArray[np.float64]:
    """Mexican hat wavelet HRF.
    
    Computes the Mexican hat (Ricker) wavelet at given time points.
    
    Args:
        t: Time points in seconds
        mean: Center of the wavelet
        sd: Width parameter
        
    Returns:
        HRF values at time points
    """
    t = np.asarray(t, dtype=np.float64)
    t0 = t - mean
    a = (1 - (t0 / sd)**2) * np.exp(-t0**2 / (2 * sd**2))
    scale = np.sqrt(2 / (3 * sd * np.pi**(1/4)))
    return scale * a


def hrf_inv_logit(
    t: ArrayLike,
    mu1: float = 6.0,
    s1: float = 1.0,
    mu2: float = 16.0,
    s2: float = 1.0,
    lag: float = 0.0
) -> NDArray[np.float64]:
    """Inverse logit (sigmoid difference) HRF.
    
    HRF using the difference of two inverse logit functions.
    
    Args:
        t: Time points in seconds
        mu1: Time-to-peak for rising phase
        s1: Width of first logistic function
        mu2: Time-to-peak for falling phase
        s2: Width of second logistic function
        lag: Time delay
        
    Returns:
        HRF values at time points
    """
    t = np.asarray(t, dtype=np.float64)
    inv_logit1 = 1 / (1 + np.exp(-(t - lag - mu1) / s1))
    inv_logit2 = 1 / (1 + np.exp(-(t - lag - mu2) / s2))
    return inv_logit1 - inv_logit2


def hrf_half_cosine(
    t: ArrayLike,
    h1: float = 1.0,
    h2: float = 5.0,
    h3: float = 7.0,
    h4: float = 7.0,
    f1: float = 0.0,
    f2: float = 0.0
) -> NDArray[np.float64]:
    """Half-cosine basis HRF.

    Creates an HRF using four half-cosine segments with smooth transitions
    between levels: 0 -> f1 -> 1 (peak) -> f2 -> 0.

    Reference: Woolrich, Behrens & Smith (2004), NeuroImage.

    Args:
        t: Time points in seconds
        h1: Duration of transition from 0 to f1 (initial dip)
        h2: Duration of transition from f1 to 1 (rise to peak)
        h3: Duration of transition from 1 to f2 (undershoot)
        h4: Duration of transition from f2 to 0 (recovery)
        f1: Initial dip level (default 0), typically in [-0.2, 0]
        f2: Undershoot level (default 0), typically in [-0.3, 0]

    Returns:
        HRF values at time points
    """
    t = np.asarray(t, dtype=np.float64)
    result = np.zeros_like(t)

    # Transition function matching R: smoothly transitions from a to b
    def trans(tt, a, b, t0, w):
        return a + 0.5 * (b - a) * (1 - np.cos(np.pi * (tt - t0) / w))

    t1 = h1
    t2 = h1 + h2
    t3 = h1 + h2 + h3
    t4 = t3 + h4

    # segment 1: 0 -> f1
    idx = (t >= 0) & (t <= t1)
    result[idx] = trans(t[idx], 0, f1, 0, h1)

    # segment 2: f1 -> 1
    idx = (t > t1) & (t <= t2)
    result[idx] = trans(t[idx], f1, 1, t1, h2)

    # segment 3: 1 -> f2 (undershoot)
    idx = (t > t2) & (t <= t3)
    result[idx] = trans(t[idx], 1, f2, t2, h3)

    # segment 4: f2 -> 0 (recovery)
    idx = (t > t3) & (t <= t4)
    result[idx] = trans(t[idx], f2, 0, t3, h4)

    # before/after window: 0
    result[(t < 0) | (t > t4)] = 0

    return result


def hrf_sine(
    t: ArrayLike,
    span: float = 24.0,
    n_basis: int = 5,
) -> NDArray[np.float64]:
    """Sine basis HRF.

    Creates a set of sine basis functions over a temporal window.

    Args:
        t: Time points in seconds
        span: Temporal window span
        n_basis: Number of basis functions

    Returns:
        Matrix of sine basis functions ``(len(t), n_basis)``
    """
    t = np.asarray(t, dtype=np.float64)
    basis = np.zeros((len(t), n_basis))

    for n in range(1, n_basis + 1):
        basis[:, n-1] = np.sin(2 * np.pi * n * t / span)

    # Zero out values outside [0, span] support
    out_of_support = (t < 0) | (t > span)
    basis[out_of_support, :] = 0

    return basis


def hrf_lwu(
    t: ArrayLike,
    tau: float = 6.0,
    sigma: float = 2.5,
    rho: float = 0.35,
    normalize: Literal["none", "height", "area"] = "none",
) -> NDArray[np.float64]:
    """Lag-Width-Undershoot (LWU) HRF.
    
    Computes the LWU hemodynamic response function using two Gaussian
    components to model the main response and an optional undershoot.
    
    Formula:
    h(t; τ, σ, ρ) = exp(-(t-τ)²/(2σ²)) - ρ*exp(-(t-τ-2σ)²/(2(1.6σ)²))
    
    Args:
        t: Time points in seconds
        tau: Lag of main Gaussian (time-to-peak)
        sigma: Width of main Gaussian (must be > 0.05)
        rho: Amplitude of undershoot (0 to 1.5)
        normalize: Normalization type ('none' or 'height')
        
    Returns:
        HRF values at time points
    """
    t = np.asarray(t, dtype=np.float64)
    
    # Validate parameters
    if sigma <= 0.05:
        raise ValueError("sigma must be > 0.05")
    if not 0 <= rho <= 1.5:
        raise ValueError("rho must be between 0 and 1.5")
    if normalize not in ["none", "height", "area"]:
        raise ValueError("normalize must be 'none', 'height', or 'area'")
    
    # Main positive Gaussian component
    term1 = np.exp(-((t - tau)**2) / (2 * sigma**2))
    
    # Undershoot Gaussian component
    term2 = rho * np.exp(-((t - (tau + 2 * sigma))**2) / (2 * (1.6 * sigma)**2))
    
    response = term1 - term2
    
    if normalize == "height":
        max_abs_val = np.max(np.abs(response))
        if max_abs_val > 1e-10:
            response = response / max_abs_val
    elif normalize == "area":
        warnings.warn(
            "normalize='area' is not implemented; returning unnormalised LWU HRF",
            UserWarning,
            stacklevel=2,
        )
    
    return response


def hrf_basis_lwu(
    theta0: ArrayLike,
    t: ArrayLike,
    normalize_primary: str = "none"
) -> NDArray[np.float64]:
    """LWU HRF basis for Taylor expansion.
    
    Constructs the basis set for the LWU HRF model, consisting of the
    LWU HRF evaluated at expansion point theta0 and its partial derivatives.
    
    Args:
        theta0: Expansion point [tau0, sigma0, rho0]
        t: Time points in seconds
        normalize_primary: Normalization for primary HRF ('none' or 'height')
        
    Returns:
        Matrix of basis functions (len(t) x 4)
    """
    t = np.asarray(t, dtype=np.float64)
    theta0 = np.asarray(theta0, dtype=np.float64)
    
    if len(theta0) != 3:
        raise ValueError("theta0 must have length 3 [tau, sigma, rho]")
    
    tau0, sigma0, rho0 = theta0
    
    # Primary HRF at expansion point
    h0 = hrf_lwu(t, tau0, sigma0, rho0, normalize=normalize_primary)
    
    # Compute partial derivatives numerically
    eps = 1e-5
    
    # Partial derivative w.r.t. tau
    h_tau_plus = hrf_lwu(t, tau0 + eps, sigma0, rho0)
    h_tau_minus = hrf_lwu(t, tau0 - eps, sigma0, rho0)
    dh_dtau = (h_tau_plus - h_tau_minus) / (2 * eps)
    
    # Partial derivative w.r.t. sigma
    h_sigma_plus = hrf_lwu(t, tau0, sigma0 + eps, rho0)
    h_sigma_minus = hrf_lwu(t, tau0, sigma0 - eps, rho0)
    dh_dsigma = (h_sigma_plus - h_sigma_minus) / (2 * eps)
    
    # Partial derivative w.r.t. rho
    h_rho_plus = hrf_lwu(t, tau0, sigma0, rho0 + eps)
    h_rho_minus = hrf_lwu(t, tau0, sigma0, rho0 - eps)
    dh_drho = (h_rho_plus - h_rho_minus) / (2 * eps)
    
    # Stack as columns
    basis = np.column_stack([h0, dh_dtau, dh_dsigma, dh_drho])

    return basis


def boxcar_hrf(
    t: ArrayLike,
    width: float = 1.0,
    amplitude: float = 1.0,
    normalize: bool = False,
) -> NDArray[np.float64]:
    """Boxcar (step function) HRF.

    Returns a constant value for ``0 <= t < width``, zero elsewhere.
    When ``normalize=True`` the amplitude is set to ``1/width`` so that
    the integral equals 1 and the GLM coefficient estimates the mean
    signal in the window.

    Args:
        t: Time points in seconds.
        width: Duration of the boxcar in seconds (must be > 0).
        amplitude: Height of the boxcar.
        normalize: If True, set ``amplitude = 1/width``.

    Returns:
        HRF values at time points *t*.
    """
    if width <= 0:
        raise ValueError("`width` must be positive.")

    if normalize:
        amplitude = 1.0 / width

    t = np.asarray(t, dtype=np.float64)
    return np.where((t >= 0) & (t < width), amplitude, 0.0)

def weighted_hrf(
    t: ArrayLike,
    weights: Sequence[float],
    width: Optional[float] = None,
    times: Optional[Sequence[float]] = None,
    method: Literal["constant", "linear"] = "constant",
    normalize: bool = False,
) -> NDArray[np.float64]:
    """Weighted HRF with user-specified weights.

    Creates a flexible HRF that maps weights to time points with no
    built-in hemodynamic delay.  Two specification modes are supported:

    * ``width + weights``: weights are evenly spaced from 0 to *width*.
    * ``times + weights``: explicit time points for each weight.

    Args:
        t: Time points in seconds at which to evaluate.
        weights: Numeric sequence with >= 2 elements.
        width: Total window duration (used when *times* is ``None``).
        times: Explicit time points for each weight.  Must be strictly
            increasing and have the same length as *weights*.  When
            provided, *width* is ignored.
        method: ``"constant"`` (step / piece-wise constant) or
            ``"linear"`` (linear interpolation).
        normalize: If True, scale weights so they sum (constant) or
            integrate (linear) to 1.

    Returns:
        HRF values at time points *t*.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) < 2:
        raise ValueError("`weights` must have at least 2 elements.")

    if method not in ("constant", "linear"):
        raise ValueError("`method` must be 'constant' or 'linear'.")

    # Resolve time points
    if times is not None:
        times = np.asarray(times, dtype=np.float64)
        if len(times) < 2:
            raise ValueError("`times` must have at least 2 elements.")
        if len(times) != len(weights):
            raise ValueError("`times` and `weights` must have the same length.")
        if not np.all(np.diff(times) > 0):
            raise ValueError("`times` must be strictly increasing.")
        if times[0] < 0:
            raise ValueError("`times` must start at 0 or later.")
    elif width is not None:
        if width <= 0:
            raise ValueError("`width` must be positive.")
        times = np.linspace(0, width, len(weights))
    else:
        raise ValueError("Either `width` or `times` must be provided.")

    # Normalize
    if normalize:
        if method == "constant":
            weight_sum = np.sum(weights[:-1])
            if abs(weight_sum) > 1e-10:
                weights = weights / weight_sum
        else:
            intervals = np.diff(times)
            avg_w = (weights[:-1] + weights[1:]) / 2.0
            integral = np.sum(intervals * avg_w)
            if abs(integral) > 1e-10:
                weights = weights / integral

    # Build interpolation function and evaluate
    t = np.asarray(t, dtype=np.float64)
    kind = "zero" if method == "constant" else "linear"
    interp_func = interpolate.interp1d(
        times, weights, kind=kind, bounds_error=False, fill_value=0.0
    )
    result = interp_func(t)
    # Ensure zero outside the domain
    result = np.where((t < times[0]) | (t > times[-1]), 0.0, result)
    return result
