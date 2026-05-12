"""HRF decorator functions for modifying HRF behavior."""

from __future__ import annotations

from typing import Optional, Union, Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import HRF, FunctionHRF, as_hrf


def _block_offsets_weights(width: float, precision: float):
    """Compute trapezoidal quadrature offsets and weights for block evaluation.

    Matches the R helper ``.block_offsets_weights(width, precision)``.

    Args:
        width: Block duration in seconds.
        precision: Sampling step size in seconds.

    Returns:
        Tuple of (offsets, weights) as 1-D numpy arrays.
    """
    offsets = np.arange(0, width, precision)
    # Ensure the endpoint is included
    if len(offsets) == 0 or offsets[-1] < width:
        offsets = np.append(offsets, width)

    if len(offsets) == 1:
        return offsets, np.array([1.0])

    deltas = np.diff(offsets)
    weights = np.zeros(len(offsets))
    weights[0] = deltas[0] / 2.0
    weights[-1] = deltas[-1] / 2.0
    if len(offsets) > 2:
        weights[1:-1] = (deltas[:-1] + deltas[1:]) / 2.0

    return offsets, weights


def lag_hrf(hrf: HRF, lag: float) -> HRF:
    """Apply temporal lag to an HRF.
    
    This decorator shifts the HRF in time by the specified lag amount.
    
    Args:
        hrf: The HRF to lag
        lag: Time lag in seconds (must be finite)
        
    Returns:
        New HRF with temporal lag applied
        
    Raises:
        ValueError: If lag is not finite
    """
    if not np.isfinite(lag):
        raise ValueError("lag must be finite")
    
    # Create lagged function
    def lagged_func(t: ArrayLike) -> NDArray[np.float64]:
        """Evaluate lagged HRF."""
        t = np.asarray(t)
        return hrf(t - lag)
    
    # Create new HRF with updated attributes
    new_name = f"{hrf.name}_lag({lag})"
    new_span = hrf.span + max(0, lag)
    new_params = hrf.params.copy()
    new_params['_lag'] = lag
    
    return FunctionHRF(
        func=lagged_func,
        name=new_name,
        nbasis=hrf.nbasis,
        span=new_span,
        params=new_params,
    )


def block_hrf(
    hrf: HRF,
    width: float,
    precision: float = 0.1,
    half_life: float = float('inf'),
    summate: bool = True,
    normalize: bool = False,
) -> HRF:
    """Create a blocked/sustained version of an HRF.
    
    This decorator convolves the HRF with a boxcar function to model
    sustained stimulation.
    
    Args:
        hrf: The HRF to block
        width: Duration of the block in seconds (must be finite)
        precision: Temporal precision for convolution (must be finite and positive)
        half_life: Half-life for exponential decay (default: no decay)
        summate: If True, responses accumulate (peak grows with duration);
            if False, responses are averaged (same shape, peak stays constant)
        normalize: If True, normalize result to unit peak
        
    Returns:
        New HRF representing blocked response
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not np.isfinite(width):
        raise ValueError("width must be finite")
    if not np.isfinite(precision) or precision <= 0:
        raise ValueError("precision must be finite and positive")
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    
    # For very small widths, just return the original HRF
    if width <= precision:
        return hrf

    orig_nbasis = hrf.nbasis

    # Create blocked function using trapezoidal quadrature
    def blocked_func(t: ArrayLike) -> NDArray[np.float64]:
        """Evaluate blocked HRF via trapezoidal quadrature."""
        t = np.asarray(t)

        quad_offsets, quad_weights = _block_offsets_weights(width, precision)

        # Evaluate HRF at each offset, applying per-offset decay
        hmat_list = []
        for offset in quad_offsets:
            decay_factor = (
                1.0 if not np.isfinite(half_life)
                else np.exp(-np.log(2) * offset / half_life)
            )
            vals = hrf(t - offset) * decay_factor
            hmat_list.append(vals)

        if orig_nbasis == 1:
            # Stack columns: each entry is shape (len(t),)
            hmat = np.column_stack(hmat_list)  # (len(t), n_offsets)
            result = hmat @ quad_weights
            if not summate:
                # Same convolution shape but amplitude doesn't grow with duration
                weight_sum = np.sum(quad_weights)
                if weight_sum > 0:
                    result = result / weight_sum
        else:
            # Multi-basis: each entry is shape (len(t), nbasis)
            weighted = [vals * wt for vals, wt in zip(hmat_list, quad_weights)]
            result = weighted[0].copy()
            for w in weighted[1:]:
                result += w
            if not summate:
                weight_sum = np.sum(quad_weights)
                if weight_sum > 0:
                    result = result / weight_sum

        # Normalize if requested
        if normalize:
            if result.ndim == 2:
                for i in range(result.shape[1]):
                    max_val = np.max(np.abs(result[:, i]))
                    if max_val > 0:
                        result[:, i] /= max_val
            else:
                max_val = np.max(np.abs(result))
                if max_val > 0:
                    result /= max_val

        return result
    
    # Create new HRF with updated attributes
    new_name = f"{hrf.name}_block(w={width})"
    new_span = hrf.span + width
    new_params = hrf.params.copy()
    new_params.update({
        '_width': width,
        '_precision': precision,
        '_half_life': half_life,
        '_summate': summate,
        '_normalize': normalize,
    })
    
    return FunctionHRF(
        func=blocked_func,
        name=new_name,
        nbasis=hrf.nbasis,
        span=new_span,
        params=new_params,
    )


def normalize_hrf(hrf: HRF) -> HRF:
    """Normalize an HRF to unit peak amplitude.

    This decorator scales the HRF so that its maximum absolute value is 1.

    Args:
        hrf: The HRF to normalize

    Returns:
        New HRF with unit peak amplitude
    """
    # Sample HRF to find peak (use high resolution matching R reference)
    import math
    ref_n = max(1001, min(20001, math.ceil(hrf.span / 0.01) + 1))
    t_sample = np.linspace(0, hrf.span, ref_n)
    hrf_sample = hrf(t_sample)
    
    # Find peak values
    if hrf.nbasis == 1:
        peak = np.max(np.abs(hrf_sample))
        if peak == 0:
            peak = 1.0  # Avoid division by zero
    else:
        # For multi-basis, normalize each separately
        peak = np.zeros(hrf.nbasis)
        for i in range(hrf.nbasis):
            peak[i] = np.max(np.abs(hrf_sample[:, i]))
            if peak[i] == 0:
                peak[i] = 1.0
    
    # Create normalized function
    def normalized_func(t: ArrayLike) -> NDArray[np.float64]:
        """Evaluate normalized HRF."""
        result = hrf(t)
        
        if hrf.nbasis == 1:
            return result / peak
        else:
            # Ensure result is 2D for multi-basis
            if result.ndim == 1:
                result = result.reshape(1, -1)
            
            # Normalize each basis
            normalized = result / peak[np.newaxis, :]
            
            # Handle single time point output shape
            t_array = np.asarray(t)
            if t_array.ndim == 0 or (t_array.ndim == 1 and len(t_array) == 1):
                return normalized
            else:
                return normalized
    
    # Create new HRF with updated attributes
    new_name = f"{hrf.name}_norm"
    new_params = hrf.params.copy()
    new_params['_normalized'] = True
    
    return FunctionHRF(
        func=normalized_func,
        name=new_name,
        nbasis=hrf.nbasis,
        span=hrf.span,
        params=new_params,
    )


def gen_hrf_lagged(
    hrf: Union[HRF, Callable],
    lags: ArrayLike,
    name: Optional[str] = None,
) -> HRF:
    """Generate a set of lagged HRFs.
    
    Creates a multi-basis HRF with each basis function being the original
    HRF shifted by different lag amounts.
    
    Args:
        hrf: Base HRF or function to lag
        lags: Array of lag values in seconds
        name: Optional name for the combined HRF
        
    Returns:
        Multi-basis HRF with lagged versions
    """
    # Ensure HRF object
    if not isinstance(hrf, HRF):
        hrf = as_hrf(hrf)
    
    lags = np.atleast_1d(np.asarray(lags, dtype=np.float64))
    
    # Create lagged HRFs
    lagged_hrfs = [lag_hrf(hrf, lag) for lag in lags]
    
    # Combine them
    from .core import bind_basis
    combined = bind_basis(*lagged_hrfs)
    
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


def hrf_lagged(
    hrf: Union[HRF, Callable],
    lag: ArrayLike = 2.0,
    normalize: bool = False,
    name: Optional[str] = None,
) -> HRF:
    """R-compatible alias for creating lagged HRF bases."""
    lagged = gen_hrf_lagged(hrf, lag, name=name)
    if normalize:
        lagged = normalize_hrf(lagged)
    return lagged


def gen_hrf_blocked(
    hrf: Union[HRF, Callable],
    widths: ArrayLike,
    precision: float = 0.1,
    half_life: float = float('inf'),
    summate: bool = True,
    normalize: bool = False,
    name: Optional[str] = None,
) -> HRF:
    """Generate a set of blocked HRFs with different widths.
    
    Creates a multi-basis HRF with each basis function being the original
    HRF convolved with boxcars of different widths.
    
    Args:
        hrf: Base HRF or function to block
        widths: Array of block widths in seconds
        precision: Temporal precision for convolution
        half_life: Half-life for exponential decay
        summate: If True, responses accumulate; if False, averaged
        normalize: If True, normalize each basis
        name: Optional name for the combined HRF
        
    Returns:
        Multi-basis HRF with blocked versions
    """
    # Ensure HRF object
    if not isinstance(hrf, HRF):
        hrf = as_hrf(hrf)
    
    widths = np.atleast_1d(np.asarray(widths, dtype=np.float64))
    
    # Create blocked HRFs
    blocked_hrfs = []
    for width in widths:
        if width == 0 and normalize:
            # For width=0, block_hrf returns original, so normalize manually
            blocked_hrfs.append(normalize_hrf(hrf))
        else:
            blocked_hrfs.append(
                block_hrf(hrf, width, precision, half_life, summate, normalize)
            )
    
    # Combine them
    from .core import bind_basis
    combined = bind_basis(*blocked_hrfs)
    
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


def hrf_blocked(
    hrf: Optional[Union[HRF, Callable]] = None,
    width: ArrayLike = 5.0,
    precision: float = 0.1,
    half_life: float = float('inf'),
    summate: bool = True,
    normalize: bool = False,
    name: Optional[str] = None,
) -> HRF:
    """R-compatible alias for creating blocked HRF bases."""
    if hrf is None:
        from .library import GAUSSIAN_HRF
        hrf = GAUSSIAN_HRF
    return gen_hrf_blocked(
        hrf,
        width,
        precision=precision,
        half_life=half_life,
        summate=summate,
        normalize=normalize,
        name=name,
    )
