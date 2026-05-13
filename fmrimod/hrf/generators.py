"""HRF generator functions."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import HRF, FunctionHRF, as_hrf, bind_basis
from .decorators import block_hrf, lag_hrf
from .functions import (
    boxcar_hrf,
    weighted_hrf,
)


def gen_hrf(
    hrf: Union[HRF, Callable, str],
    lag: float = 0.0,
    width: float = 0.0,
    precision: float = 0.1,
    half_life: float = float('inf'),
    summate: bool = True,
    normalize: bool = False,
    name: Optional[str] = None,
    span: Optional[float] = None,
    **kwargs: Any,
) -> HRF:
    """Generate an HRF with optional lag and block width.

    This is a general-purpose HRF generator that can create HRFs from
    functions, existing HRF objects, or registry names, and apply
    lag and/or block decorators.

    Args:
        hrf: HRF object, function, or registry name
        lag: Temporal lag in seconds
        width: Block width in seconds (0 for no blocking)
        precision: Time step for block convolution (default 0.1)
        half_life: Half-life for exponential decay (default Inf for no decay)
        summate: If True, sum (integrate) block convolution; if False, average
        normalize: If True, normalize to unit peak
        name: Optional name to assign to the HRF
        span: Optional span to assign to the HRF
        **kwargs: Additional parameters passed to HRF construction

    Returns:
        Generated HRF with decorators applied
    """
    # Handle string input (registry lookup)
    if isinstance(hrf, str):
        from .registry import get_hrf
        base_hrf = get_hrf(hrf, **kwargs)
    # Handle callable input
    elif callable(hrf) and not isinstance(hrf, HRF):
        # Check if it's a generator function that needs parameters
        if kwargs:
            # Create a partial function with the kwargs
            import functools
            partial_func = functools.partial(hrf, **kwargs)
            
            # Determine nbasis by sampling
            test_result = partial_func(np.array([1.0]))
            if isinstance(test_result, np.ndarray) and test_result.ndim == 2:
                nbasis = test_result.shape[1]
            else:
                nbasis = 1
            
            base_hrf = as_hrf(partial_func, nbasis=nbasis)
        else:
            # No parameters, determine nbasis by sampling
            test_result = hrf(np.array([1.0]))
            if isinstance(test_result, np.ndarray) and test_result.ndim == 2:
                nbasis = test_result.shape[1]
            else:
                nbasis = 1
            base_hrf = as_hrf(hrf, nbasis=nbasis)
    else:
        # Already an HRF
        base_hrf = hrf
    
    # Apply block if specified
    if width > 0:
        base_hrf = block_hrf(base_hrf, width, precision=precision, half_life=half_life, summate=summate)

    # Apply lag if specified
    if lag != 0:
        base_hrf = lag_hrf(base_hrf, lag)
    
    # Apply normalization if requested
    if normalize:
        from .normalization import normalize as _normalize

        base_hrf = _normalize(base_hrf, "unit_peak_per_basis")

    # Apply name and/or span if specified
    if name is not None:
        base_hrf = FunctionHRF(
            func=base_hrf,
            name=name,
            nbasis=base_hrf.nbasis,
            span=base_hrf.span if span is None else span,
            params=base_hrf.params,
        )
    elif span is not None:
        base_hrf = FunctionHRF(
            func=base_hrf,
            name=base_hrf.name,
            nbasis=base_hrf.nbasis,
            span=span,
            params=base_hrf.params,
        )

    return base_hrf


def gen_hrf_set(*hrfs: Union[HRF, Callable], name: Optional[str] = None) -> HRF:
    """Combine multiple HRFs into a single multi-basis HRF.
    
    Args:
        *hrfs: HRF objects or functions to combine
        name: Optional name for the combined HRF
        
    Returns:
        Multi-basis HRF combining all inputs
    """
    # Convert all to HRF objects
    hrf_objects = []
    for hrf in hrfs:
        if isinstance(hrf, HRF):
            hrf_objects.append(hrf)
        else:
            hrf_objects.append(as_hrf(hrf))
    
    # Combine using bind_basis
    combined = bind_basis(*hrf_objects)
    
    # Set custom name if provided
    if name is not None:
        combined = FunctionHRF(
            func=combined,
            name=name,
            nbasis=combined.nbasis,
            span=combined.span,
            params=combined.params,
        )
    
    return combined


def hrf_set(*hrfs: Union[HRF, Callable], name: str = "hrf_set") -> HRF:
    """R-compatible alias for :func:`gen_hrf_set`.

    The original ``fmrihrf`` public API stabilized ``hrf_set()`` as the
    preferred name and deprecated ``gen_hrf_set()``. In Python both names point
    to the same implementation.
    """
    return gen_hrf_set(*hrfs, name=name)


def gamma_generator(
    shape: float = 6.0,
    rate: float = 1.0,
    name: Optional[str] = None,
) -> HRF:
    """Generate Gamma HRF with custom parameters.

    Returns a typed :class:`GammaHRF` instance.
    """
    from .library import GammaHRF

    return GammaHRF(
        name=name or f"gamma_s{shape}_r{rate}",
        nbasis=1,
        span=24.0,
        shape=shape,
        rate=rate,
    )


def bspline_generator(
    n_basis: int = 5,
    degree: int = 3,
    span: float = 24.0,
    name: Optional[str] = None,
) -> HRF:
    """Generate B-spline basis HRF.

    Returns a typed :class:`BSplineHRF` instance.
    """
    from .library import BSplineHRF

    return BSplineHRF(
        name=name or f"bspline_N{n_basis}_d{degree}",
        nbasis=n_basis,
        span=span,
        degree=degree,
    )


def fir_generator(
    n_basis: int = 10,
    span: float = 24.0,
    name: Optional[str] = None,
) -> HRF:
    """Generate FIR (Finite Impulse Response) basis HRF.

    Returns a typed :class:`FIRHRF` instance.
    """
    from .library import FIRHRF

    return FIRHRF(
        name=name or f"fir_N{n_basis}",
        nbasis=n_basis,
        span=span,
    )


def fourier_generator(
    n_basis: int = 5,
    span: float = 24.0,
    name: Optional[str] = None,
) -> HRF:
    """Generate Fourier basis HRF.

    Returns a typed :class:`FourierHRF` instance.
    """
    from .library import FourierHRF

    return FourierHRF(
        name=name or f"fourier_N{n_basis}",
        nbasis=n_basis,
        span=span,
    )


def daguerre_generator(
    n_basis: int = 3,
    span: float = 24.0,
    scale: float = 4.0,
    name: Optional[str] = None,
) -> HRF:
    """Generate Daguerre basis HRF using Laguerre polynomial recurrence.

    Creates orthogonal basis functions on [0, Inf) that naturally decay to zero,
    suitable for HRF modeling.

    Args:
        n_basis: Number of basis functions (default: 3)
        span: Temporal span of the basis set (default: 24)
        scale: Scale parameter for the time axis (default: 4)
        name: Optional custom name

    Returns:
        Typed :class:`DaguerreHRF` instance.
    """
    from .library import DaguerreHRF

    return DaguerreHRF(
        name=name or "daguerre",
        nbasis=n_basis,
        span=span,
        scale=scale,
    )


def make_hrf(
    hrf_spec: Union[str, Dict[str, Any], HRF],
    lag: float = 0.0,
    normalize: bool = False,
) -> HRF:
    """Create an HRF from a typed specification.

    Accepted forms:

    - A plain canonical name or alias (``"spmg1"``, ``"bspline"``,
      ``"spm_canonical"``). The string is routed through
      :func:`fmrimod.hrf.aliases._normalize_hrf_name`.
    - A dict with a ``"type"`` key plus per-kind parameters
      (``{"type": "gamma", "shape": 6, "rate": 1}``).
    - An :class:`HRF` instance (returned as-is, with ``lag`` /
      ``normalize`` applied).

    The legacy string-DSL form (``"bspline(N=7, degree=3)"``) was
    retired by bead ``bd-01KRGCZD8QC6W4XE73JW0DJK5A``: callers that
    want parameterised construction should call the typed generator
    directly (``bspline_generator(n_basis=7, degree=3)``) or
    ``get_hrf("bspline", n_basis=7, degree=3)``.

    Args:
        hrf_spec: Plain string name, dict spec, or :class:`HRF` instance.
        lag: Optional temporal lag.
        normalize: If True, normalize to unit peak per basis.

    Returns:
        Generated HRF.

    Raises:
        ValueError: If the string contains ``(`` (retired DSL form) or
            the canonical name is unknown.
    """
    if isinstance(hrf_spec, str):
        if "(" in hrf_spec:
            raise ValueError(
                "string-DSL HRF specs (e.g. \"bspline(N=7)\") are retired; "
                "use bspline_generator(n_basis=7) or "
                "get_hrf('bspline', n_basis=7) instead. See "
                "bd-01KRGCZD8QC6W4XE73JW0DJK5A."
            )
        return gen_hrf(hrf_spec, lag=lag, normalize=normalize)

    if isinstance(hrf_spec, dict):
        func_name = hrf_spec.get("type", "spmg1")
        params = {k: v for k, v in hrf_spec.items() if k != "type"}
        return gen_hrf(func_name, lag=lag, normalize=normalize, **params)

    # HRF instance or callable.
    return gen_hrf(hrf_spec, lag=lag, normalize=normalize)


def tent_generator(
    nbasis: int = 5,
    span: float = 24.0,
    name: Optional[str] = None,
) -> HRF:
    """Generate tent (degree-1 B-spline) basis HRF.

    A tent basis *is* a B-spline with ``degree=1`` and piecewise-linear
    hat functions, so it shares the BSpline typed kind. The ``name``
    keeps the ``tent`` prefix for backwards compatibility with callers
    that inspect ``hrf.name``.
    """
    from .library import BSplineHRF

    return BSplineHRF(
        name=name or f"tent_N{nbasis}",
        nbasis=nbasis,
        span=span,
        degree=1,
    )


def hrf_bspline_generator(
    nbasis: int = 5,
    span: float = 24.0,
    degree: int = 3,
    name: Optional[str] = None,
) -> HRF:
    """R-compatible B-spline HRF basis generator."""
    return bspline_generator(n_basis=nbasis, degree=degree, span=span, name=name)


def hrf_tent_generator(
    nbasis: int = 5,
    span: float = 24.0,
    name: Optional[str] = None,
) -> HRF:
    """R-compatible tent HRF basis generator."""
    return tent_generator(nbasis=nbasis, span=span, name=name)


def hrf_fourier_generator(
    nbasis: int = 5,
    span: float = 24.0,
    name: Optional[str] = None,
) -> HRF:
    """R-compatible Fourier HRF basis generator."""
    return fourier_generator(n_basis=nbasis, span=span, name=name)


def hrf_daguerre_generator(
    nbasis: int = 3,
    scale: float = 4.0,
    span: float = 24.0,
    name: Optional[str] = None,
) -> HRF:
    """R-compatible Daguerre HRF basis generator."""
    return daguerre_generator(n_basis=nbasis, span=span, scale=scale, name=name)


def hrf_fir_generator(
    nbasis: int = 12,
    span: float = 24.0,
    name: Optional[str] = None,
) -> HRF:
    """R-compatible FIR HRF basis generator."""
    return fir_generator(n_basis=nbasis, span=span, name=name)


def boxcar_generator(
    width: float = 1.0,
    amplitude: float = 1.0,
    normalize: bool = False,
    name: Optional[str] = None,
) -> HRF:
    """Generate a boxcar HRF with custom parameters.

    Args:
        width: Duration of the boxcar in seconds.
        amplitude: Height of the boxcar.
        normalize: If True, set amplitude = 1/width.
        name: Optional custom name.

    Returns:
        Boxcar HRF.
    """
    if normalize:
        raise ValueError(
            "boxcar_generator(normalize=True) is retired; "
            "set amplitude=1/width or use normalize(boxcar_generator(...), mode)."
        )
    eff_amp = amplitude

    def boxcar_func(t: ArrayLike) -> NDArray[np.float64]:
        return boxcar_hrf(t, width=width, amplitude=eff_amp)

    return FunctionHRF(
        func=boxcar_func,
        name=name or f"boxcar[{width:.2g}]",
        nbasis=1,
        span=width,
        params={"width": width, "amplitude": eff_amp, "normalize": normalize},
    )


def weighted_generator(
    weights: Sequence[float],
    width: Optional[float] = None,
    times: Optional[Sequence[float]] = None,
    method: str = "constant",
    normalize: bool = False,
    name: Optional[str] = None,
) -> HRF:
    """Generate a weighted HRF with custom parameters.

    Args:
        weights: Numeric sequence of weights (>= 2 elements).
        width: Total window duration (ignored when *times* is given).
        times: Explicit time points for each weight.
        method: 'constant' or 'linear'.
        normalize: Scale weights to sum/integrate to 1.
        name: Optional custom name.

    Returns:
        Weighted HRF.
    """
    if normalize:
        raise ValueError(
            "weighted_generator(normalize=True) is retired; "
            "scale weights explicitly or use normalize(weighted_generator(...), mode)."
        )
    _weights = np.asarray(weights, dtype=np.float64)

    if times is not None:
        _times = np.asarray(times, dtype=np.float64)
        _span = float(_times[-1])
    elif width is not None:
        _times = np.linspace(0, width, len(_weights))
        _span = float(width)
    else:
        raise ValueError("Either `width` or `times` must be provided.")

    def weighted_func(t: ArrayLike) -> NDArray[np.float64]:
        return weighted_hrf(
            t, weights=_weights, times=_times,
            method=method, normalize=normalize,
        )

    if name is None:
        name = f"weighted[w={_span:.2g}, {len(_weights)} wts]"

    return FunctionHRF(
        func=weighted_func,
        name=name,
        nbasis=1,
        span=_span,
        params={
            "times": _times.tolist(),
            "weights": _weights.tolist(),
            "method": method,
            "normalize": normalize,
        },
    )
