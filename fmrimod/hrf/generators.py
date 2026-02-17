"""HRF generator functions."""

from __future__ import annotations

import ast
from typing import Union, Optional, Callable, Any, Dict, Sequence
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import HRF, FunctionHRF, as_hrf, bind_basis
from .decorators import lag_hrf, block_hrf
from .functions import gamma_hrf, bspline_hrf, fir_basis, fourier_hrf, boxcar_hrf, weighted_hrf, daguerre_basis


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
        from .decorators import normalize_hrf
        base_hrf = normalize_hrf(base_hrf)

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


def gamma_generator(
    shape: float = 6.0,
    rate: float = 1.0,
    name: Optional[str] = None,
) -> HRF:
    """Generate Gamma HRF with custom parameters.
    
    Args:
        shape: Shape parameter of the gamma distribution
        rate: Rate parameter of the gamma distribution
        name: Optional custom name
        
    Returns:
        Gamma HRF
    """
    def gamma_func(t: ArrayLike) -> NDArray[np.float64]:
        return gamma_hrf(t, shape=shape, rate=rate)
    
    return FunctionHRF(
        func=gamma_func,
        name=name or f"gamma_s{shape}_r{rate}",
        nbasis=1,
        span=24.0,
        params={"shape": shape, "rate": rate},
    )


def bspline_generator(
    n_basis: int = 5,
    degree: int = 3,
    span: float = 24.0,
    name: Optional[str] = None,
) -> HRF:
    """Generate B-spline basis HRF.

    Args:
        n_basis: Number of basis functions
        degree: Degree of B-splines (3 for cubic)
        span: Temporal span of the basis set
        name: Optional custom name

    Returns:
        B-spline basis HRF
    """
    def bspline_func(t: ArrayLike) -> NDArray[np.float64]:
        return bspline_hrf(t, n_basis=n_basis, degree=degree, span=span)

    return FunctionHRF(
        func=bspline_func,
        name=name or f"bspline_N{n_basis}_d{degree}",
        nbasis=n_basis,
        span=span,
        params={"n_basis": n_basis, "degree": degree},
    )


def fir_generator(
    n_basis: int = 10,
    span: float = 24.0,
    name: Optional[str] = None,
) -> HRF:
    """Generate FIR (Finite Impulse Response) basis HRF.

    Args:
        n_basis: Number of basis functions
        span: Temporal span of the basis set
        name: Optional custom name

    Returns:
        FIR basis HRF
    """
    def fir_func(t: ArrayLike) -> NDArray[np.float64]:
        return fir_basis(t, n_basis=n_basis, span=span)

    return FunctionHRF(
        func=fir_func,
        name=name or f"fir_N{n_basis}",
        nbasis=n_basis,
        span=span,
        params={"n_basis": n_basis},
    )


def fourier_generator(
    n_basis: int = 5,
    span: float = 24.0,
    name: Optional[str] = None,
) -> HRF:
    """Generate Fourier basis HRF.

    Args:
        n_basis: Number of basis functions
        span: Temporal span of the basis set
        name: Optional custom name

    Returns:
        Fourier basis HRF
    """
    def fourier_func(t: ArrayLike) -> NDArray[np.float64]:
        return fourier_hrf(t, n_basis=n_basis, span=span)

    return FunctionHRF(
        func=fourier_func,
        name=name or f"fourier_N{n_basis}",
        nbasis=n_basis,
        span=span,
        params={"n_basis": n_basis},
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
        Daguerre basis HRF
    """
    def daguerre_func(t: ArrayLike) -> NDArray[np.float64]:
        return daguerre_basis(t, n_basis=n_basis, scale=scale)

    return FunctionHRF(
        func=daguerre_func,
        name=name or "daguerre",
        nbasis=n_basis,
        span=span,
        params={"n_basis": n_basis, "scale": scale},
    )


def make_hrf(
    hrf_spec: Union[str, Dict[str, Any], HRF],
    lag: float = 0.0,
    normalize: bool = False,
) -> HRF:
    """Create an HRF from a specification string or dictionary.
    
    This function provides a flexible way to create HRFs from specifications.
    
    Args:
        hrf_spec: Either a string like "bspline(N=7)" or a dict with 'type' and parameters
        lag: Optional temporal lag
        normalize: If True, normalize to unit peak
        
    Returns:
        Generated HRF
        
    Examples:
        make_hrf("spmg1")
        make_hrf("bspline(N=7, degree=3)")
        make_hrf({"type": "gamma", "shape": 6, "rate": 1})
    """
    if isinstance(hrf_spec, str):
        # Parse string specification
        if '(' in hrf_spec:
            # Extract function name and parameters
            func_name = hrf_spec[:hrf_spec.index('(')]
            params_str = hrf_spec[hrf_spec.index('(')+1:-1]
            
            # Parse parameters
            params = {}
            if params_str:
                for param in params_str.split(','):
                    key, value = param.strip().split('=')
                    # Try to convert to appropriate type
                    try:
                        params[key.strip()] = ast.literal_eval(value.strip())
                    except (ValueError, SyntaxError):
                        params[key.strip()] = value.strip()
            
            # If parameters are provided, prefer generators over pre-defined HRFs
            # Try adding "hrf_" prefix to find generators
            if params and not func_name.startswith("hrf_"):
                try:
                    return gen_hrf(f"hrf_{func_name}", lag=lag, normalize=normalize, **params)
                except ValueError:
                    # Fall back to original name
                    pass
        else:
            func_name = hrf_spec
            params = {}
    elif isinstance(hrf_spec, dict):
        # Dictionary specification
        func_name = hrf_spec.get('type', 'spmg1')
        params = {k: v for k, v in hrf_spec.items() if k != 'type'}
        
        # If parameters are provided, prefer generators over pre-defined HRFs
        if params and not func_name.startswith("hrf_"):
            try:
                return gen_hrf(f"hrf_{func_name}", lag=lag, normalize=normalize, **params)
            except ValueError:
                # Fall back to original name
                pass
    elif isinstance(hrf_spec, HRF):
        return gen_hrf(hrf_spec, lag=lag, normalize=normalize)
    else:
        # Fallback for callable or object-like specs
        return gen_hrf(hrf_spec, lag=lag, normalize=normalize)
    
    # Create HRF using gen_hrf
    return gen_hrf(func_name, lag=lag, normalize=normalize, **params)


def tent_generator(
    nbasis: int = 5,
    span: float = 24.0,
    name: Optional[str] = None,
) -> HRF:
    """Generate tent (degree-1 B-spline) basis HRF.

    A tent basis is a B-spline basis with ``degree=1``, producing
    piecewise-linear hat functions.

    Args:
        nbasis: Number of basis functions.
        span: Temporal span of the basis set in seconds.
        name: Optional custom name.

    Returns:
        Tent basis HRF.
    """
    def tent_func(t: ArrayLike) -> NDArray[np.float64]:
        return bspline_hrf(t, n_basis=nbasis, degree=1, span=span)

    return FunctionHRF(
        func=tent_func,
        name=name or f"tent_N{nbasis}",
        nbasis=nbasis,
        span=span,
        params={"n_basis": nbasis, "degree": 1, "span": span},
    )


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
    eff_amp = 1.0 / width if normalize else amplitude

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
