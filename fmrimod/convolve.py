"""Direct convolution of event objects with hemodynamic response functions.

This module provides the :func:`convolve` generic function which accepts
any event type (``EventFactor``, ``EventVariable``, ``EventMatrix``,
``EventBasis``, list, or raw ``numpy.ndarray``) and returns convolved
time-series columns. When fmrimod is installed, convolution is performed
via ``fmrimod.regressor().evaluate()`` for high-fidelity results;
otherwise a fallback impulse-train convolution is used.
"""

from functools import singledispatch
from typing import List, Optional

import numpy as np

from ._warnings import call_safely, suppress_fmrimod_warnings
from .events import EventBasis, EventFactor, EventMatrix, EventVariable
from .hrf.registry import get_hrf
from .types import Array, EventProtocol

try:
    with suppress_fmrimod_warnings():
        from . import regressor as fmrimod_regressor
    HAS_PYFMRIHRF = True
except ImportError:
    HAS_PYFMRIHRF = False


def _peak_normalize(result: Array) -> Array:
    """Peak-normalize each column so max(abs(col)) == 1.

    Parameters
    ----------
    result : array
        Array to normalize (1D or 2D)

    Returns
    -------
    array
        Peak-normalized array
    """
    if result.ndim > 1:
        for i in range(result.shape[1]):
            mx = np.max(np.abs(result[:, i]))
            if mx > 0:
                result[:, i] = result[:, i] / mx
    else:
        mx = np.max(np.abs(result))
        if mx > 0:
            result = result / mx
    return result


def _prepare_sampling_grid(grid: Array) -> Array:
    """Return a validated 1D sampling grid used for convolution."""
    grid = np.asarray(grid, dtype=float).reshape(-1)
    if grid.size == 0:
        raise ValueError("sampling_frame must contain at least one time point")
    if not np.all(np.isfinite(grid)):
        raise ValueError("sampling_frame must contain only finite values")
    return grid


def _validate_sampling_rate(sampling_rate: float) -> float:
    """Validate sampling rate is finite and strictly positive."""
    try:
        sampling_rate = float(sampling_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("sampling_rate must be a finite positive number") from exc
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be a finite positive number")
    return sampling_rate


def _validate_total_duration(total_duration: float) -> float:
    """Validate total duration is finite and strictly positive."""
    try:
        total_duration = float(total_duration)
    except (TypeError, ValueError) as exc:
        raise ValueError("total_duration must be a finite positive number") from exc
    if not np.isfinite(total_duration) or total_duration <= 0:
        raise ValueError("total_duration must be a finite positive number")
    return total_duration


def _get_fallback_timing(
    grid: Array,
    sampling_rate: float,
    used_sampling_frame: bool,
) -> tuple[float, float, float]:
    """Derive fallback timing parameters from the sampling grid."""
    if used_sampling_frame:
        if grid.size == 1:
            if not np.isfinite(sampling_rate) or sampling_rate <= 0:
                raise ValueError("sampling_rate must be a finite positive number")
            dt = 1.0 / float(sampling_rate)
        else:
            diffs = np.diff(grid)
            if not np.all(np.isfinite(diffs)) or np.any(diffs <= 0):
                raise ValueError("sampling_frame must be strictly increasing")
            dt = float(np.median(diffs))
            if not np.allclose(diffs, dt, rtol=1e-6, atol=1e-8):
                raise ValueError(
                    "sampling_frame must be uniformly spaced when using array/callable HRF"
                )
        origin = float(grid[0])
        total_duration = float(grid[-1] - origin + dt)
        return 1.0 / dt, total_duration, origin

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be a finite positive number")
    return float(sampling_rate), float(grid[-1] + 1 / sampling_rate), 0.0


@singledispatch
def convolve(x, hrf=None, sampling_rate: float = 1.0,
             sampling_frame: Optional[Array] = None, **kwargs: object) -> Array:
    """Convolve event(s) with hemodynamic response function.

    This is a generic function that convolves various event types with an HRF.

    Parameters
    ----------
    x : object
        Event object or array to convolve
    hrf : HRF, array, or None
        Hemodynamic response function. If None, uses SPM canonical HRF
    sampling_rate : float
        Sampling rate in Hz (default 1.0). Ignored if sampling_frame is provided.
    sampling_frame : array-like, optional
        Explicit time points at which to sample the convolved signal.
        If provided, takes precedence over sampling_rate.
    normalize : bool, optional
        If True, peak-normalize each regressor column after convolution
        so that max(abs(col)) == 1. Default False.
    summate : bool, optional
        Whether overlapping HRF responses are summed (default True).
        If False, max is used instead. Passed to fmrimod.regressor().
    **kwargs
        Additional arguments passed to convolution

    Returns
    -------
    Array
        Convolved signal

    Examples
    --------
    >>> # Convolve event with default HRF
    >>> event = EventVariable(onsets=[1, 5, 10], durations=[2, 2, 2],
    ...                      values=[1, 2, 3], name="stimulus")
    >>> convolved = convolve(event, sampling_rate=0.5)  # TR=2s

    >>> # Use custom HRF
    >>> from fmrimod import HRF
    >>> custom_hrf = HRF(a1=6, a2=12)
    >>> convolved = convolve(event, hrf=custom_hrf)

    >>> # Use explicit sampling frame
    >>> sampling_times = np.arange(0, 100, 2.0)  # TR=2s
    >>> convolved = convolve(event, hrf=custom_hrf, sampling_frame=sampling_times)
    """
    raise NotImplementedError(f"convolve not implemented for {type(x)}")


def _get_hrf_array(hrf, sampling_rate: float, duration: float = 32.0) -> Array:
    """Get HRF as array for convolution (fallback for non-fmrimod path).

    Note: This is maintained for backward compatibility when fmrimod is not available.
    When fmrimod is available, _convolve_with_regressor should be used instead.
    """
    if hrf is None:
        # Use default SPM canonical
        hrf_obj = get_hrf("spm")
    elif isinstance(hrf, np.ndarray):
        return hrf
    elif callable(hrf):
        # Plain callable function - evaluate it
        t = np.arange(0, duration, 1/sampling_rate)
        return hrf(t)
    else:
        hrf_obj = hrf

    # Sample HRF object
    t = np.arange(0, duration, 1/sampling_rate)
    return hrf_obj.evaluate(t)


def _get_hrf_object(hrf):
    """Get HRF object suitable for fmrimod.regressor().

    Returns a fmrimod HRF object that can be used with fmrimod.regressor(),
    or None if hrf is an array (which signals to use fallback path).
    """
    with suppress_fmrimod_warnings():
        from .hrf import library as _hrf_library
        from .hrf import registry as _hrf_registry

    if hrf is None:
        return _hrf_library.SPM_CANONICAL
    elif isinstance(hrf, str):
        hrf_lower = hrf.lower()
        if hrf_lower in ('spm', 'spm_canonical', 'canonical', 'spmg1', 'simple'):
            return _hrf_library.SPM_CANONICAL
        elif hrf_lower == 'spmg2':
            return _hrf_library.SPM_WITH_DERIVATIVE
        elif hrf_lower == 'spmg3':
            return _hrf_library.SPM_WITH_DISPERSION
        else:
            try:
                return call_safely(_hrf_registry.get_hrf, hrf_lower)
            except (KeyError, ValueError):
                return _hrf_library.SPM_CANONICAL
    elif isinstance(hrf, np.ndarray):
        # Array HRF - signal to use fallback path
        return None
    elif callable(hrf) and not hasattr(hrf, 'nbasis'):
        # Plain callable function - signal to use fallback path
        return None
    else:
        # Try using directly - check if it's fmrimod compatible
        try:
            fmrimod_regressor(onsets=np.array([0.0]), hrf=hrf)
            return hrf
        except Exception:
            return _hrf_library.SPM_CANONICAL


def _convolve_with_regressor(
    onsets: Array,
    amplitudes: Array,
    durations: Array,
    hrf,
    sampling_grid: Array,
    precision: float = 0.1,
    summate: bool = True
) -> Array:
    """Convolve events with HRF using fmrimod.regressor().

    Parameters
    ----------
    onsets : array
        Event onset times in seconds
    amplitudes : array
        Event amplitudes/values
    durations : array
        Event durations in seconds
    hrf : HRF object or string
        Hemodynamic response function
    sampling_grid : array
        Time points at which to evaluate the convolved signal
    precision : float
        Temporal precision for convolution (default 0.1s)
    summate : bool
        Whether overlapping HRF responses are summed (default True).
        If False, max is used instead.

    Returns
    -------
    array
        Convolved signal at sampling_grid time points
    """
    if not HAS_PYFMRIHRF:
        raise RuntimeError(
            "fmrimod is required for convolution. "
            "Install with: pip install fmrimod"
        )

    # Get HRF object
    hrf_obj = _get_hrf_object(hrf)

    # Create regressor using fmrimod
    reg = fmrimod_regressor(
        onsets=onsets,
        hrf=hrf_obj,
        duration=durations,
        amplitude=amplitudes,
        summate=summate
    )

    # Evaluate at sampling grid
    result = reg.evaluate(sampling_grid, precision=precision)

    return result


def _convolve_impulses(times: Array, values: Array, durations: Array,
                      hrf_array: Array, sampling_rate: float,
                      total_duration: float) -> Array:
    """Convolve impulses with HRF (fallback implementation).

    Note: This is maintained for backward compatibility when fmrimod is not available.
    """
    # Create impulse train
    n_samples = int(total_duration * sampling_rate)
    signal = np.zeros(n_samples)

    for t, v, d in zip(times, values, durations):
        # Convert to samples
        start_idx = int(t * sampling_rate)
        duration_samples = max(1, int(d * sampling_rate))

        # Add boxcar
        if start_idx < n_samples:
            end_idx = min(start_idx + duration_samples, n_samples)
            signal[start_idx:end_idx] += v

    # Convolve with HRF
    convolved = np.convolve(signal, hrf_array, mode='same')

    return convolved


def _convolve_impulses_on_grid(
    times: Array,
    values: Array,
    durations: Array,
    hrf_array: Array,
    sampling_rate: float,
    sampling_grid: Array,
) -> Array:
    """Fallback convolution sampled at explicit grid points.

    Uses the legacy discrete impulse convolution on an internal uniform grid,
    then linearly interpolates back to the requested ``sampling_grid``.
    This preserves output length and supports non-zero / non-unit grids.
    """
    grid = _prepare_sampling_grid(sampling_grid)
    onsets = np.asarray(times, dtype=float).reshape(-1)
    durs = np.asarray(durations, dtype=float).reshape(-1)
    amps = np.asarray(values, dtype=float).reshape(-1)

    grid_min = float(np.min(grid))
    effective_sampling_rate = _effective_sampling_rate_from_grid(grid, sampling_rate)

    if onsets.size > 0:
        # Anchor quantization to the explicit grid origin (or earlier onsets),
        # not absolute time zero, to avoid phase shifts for non-zero starts.
        origin = min(grid_min, float(np.min(onsets)))
        max_event_end = float(np.max(onsets + durs))
    else:
        origin = grid_min
        max_event_end = float(np.max(grid))

    shifted_grid = grid - origin
    shifted_onsets = onsets - origin
    max_time = max(float(np.max(shifted_grid)), max_event_end - origin)
    total_duration = max_time + 1.0 / effective_sampling_rate

    dense = _convolve_impulses(
        shifted_onsets,
        amps,
        durs,
        hrf_array,
        effective_sampling_rate,
        total_duration,
    )
    dense_grid = np.arange(dense.shape[0], dtype=float) / effective_sampling_rate
    return np.interp(shifted_grid, dense_grid, dense, left=0.0, right=0.0)


def _validate_single_point_sampling_rate(sampling_grid: Array, sampling_rate: float) -> None:
    """Validate sampling rate when output grid has one sample.

    Callable HRFs derive sample spacing from ``sampling_rate`` even with
    single-point grids, so invalid values must fail with a clear ValueError.
    """
    if sampling_grid.size == 1 and (
        not np.isfinite(sampling_rate) or sampling_rate <= 0
    ):
        raise ValueError("sampling_rate must be a finite positive number")


def _effective_sampling_rate_from_grid(sampling_grid: Array, sampling_rate: float) -> float:
    """Resolve effective sampling rate from explicit sampling grid when regular."""
    grid = _prepare_sampling_grid(sampling_grid)
    effective_sampling_rate = float(sampling_rate)
    if grid.size > 1:
        diffs = np.diff(grid)
        if not np.all(np.isfinite(diffs)) or np.any(diffs <= 0):
            raise ValueError("sampling_frame must be strictly increasing")
        dt = float(np.median(diffs))
        if not np.allclose(diffs, dt, rtol=1e-6, atol=1e-8):
            raise ValueError(
                "sampling_frame must be uniformly spaced when using array/callable HRF"
            )
        effective_sampling_rate = 1.0 / dt
    if not np.isfinite(effective_sampling_rate) or effective_sampling_rate <= 0:
        raise ValueError("sampling_rate must be a finite positive number")
    return effective_sampling_rate


@convolve.register(EventFactor)
def _convolve_event_factor(event: EventFactor, hrf=None,
                          sampling_rate: float = 1.0,
                          sampling_frame: Optional[Array] = None,
                          total_duration: Optional[float] = None,
                          precision: float = 0.1,
                          normalize: bool = False,
                          summate: bool = True,
                          **kwargs: object) -> Array:
    """Convolve a categorical EventFactor with an HRF.

    Each factor level produces one column in the output. Events matching
    that level are convolved with unit amplitude.

    Parameters
    ----------
    event : EventFactor
        Categorical event to convolve.
    hrf : HRF, array, str, or None
        HRF specification. ``None`` uses the SPM canonical HRF.
    sampling_rate : float
        Sampling rate in Hz (ignored when ``sampling_frame`` is given).
    sampling_frame : array-like, optional
        Explicit sampling time points.
    total_duration : float, optional
        Total signal duration in seconds (auto-detected if omitted).
    precision : float
        Temporal precision for fmrimod convolution (default 0.1 s).
    normalize : bool
        Peak-normalize each column after convolution.
    summate : bool
        Sum overlapping HRF responses (True) or take max (False).

    Returns
    -------
    Array
        Convolved array of shape ``(n_timepoints, n_levels)``.
    """
    # Determine sampling grid
    if sampling_frame is not None:
        grid = _prepare_sampling_grid(sampling_frame)
    else:
        sampling_rate = _validate_sampling_rate(sampling_rate)
        # Determine total duration
        if total_duration is None:
            total_duration = np.max(event.onsets + event.durations) + 32.0
        total_duration = _validate_total_duration(total_duration)
        grid = _prepare_sampling_grid(np.arange(0, total_duration, 1/sampling_rate))

    # Check if we should use fmrimod or fallback
    use_fmrimod = HAS_PYFMRIHRF
    if use_fmrimod and (isinstance(hrf, np.ndarray) or (callable(hrf) and not hasattr(hrf, 'nbasis'))):
        # Array or plain callable HRF - use fallback path
        use_fmrimod = False

    if use_fmrimod:
        # Use fmrimod.regressor() for each level
        n_levels = len(event.levels)
        convolved = np.zeros((len(grid), n_levels))

        # Convolve each level separately
        for i, level in enumerate(event.levels):
            # Get events for this level
            mask = event.values == level
            if np.any(mask):
                level_values = np.ones(np.sum(mask))
                result = _convolve_with_regressor(
                    event.onsets[mask],
                    level_values,
                    event.durations[mask],
                    hrf,
                    grid,
                    precision=precision,
                    summate=summate
                )
                # Handle single-basis case
                if result.ndim == 1:
                    convolved[:, i] = result
                else:
                    convolved[:, i] = result[:, 0]
    else:
        # Fallback to manual convolution
        _validate_single_point_sampling_rate(grid, sampling_rate)
        fallback_rate = _effective_sampling_rate_from_grid(grid, sampling_rate)
        hrf_array = _get_hrf_array(hrf, fallback_rate)
        n_levels = len(event.levels)
        n_samples = len(grid)
        convolved = np.zeros((n_samples, n_levels))

        for i, level in enumerate(event.levels):
            mask = event.values == level
            if np.any(mask):
                level_values = np.ones(np.sum(mask))
                convolved[:, i] = _convolve_impulses_on_grid(
                    event.onsets[mask],
                    level_values,
                    event.durations[mask],
                    hrf_array,
                    fallback_rate,
                    grid,
                )

    if normalize:
        convolved = _peak_normalize(convolved)

    return convolved


@convolve.register(EventVariable)
def _convolve_event_variable(event: EventVariable, hrf=None,
                            sampling_rate: float = 1.0,
                            sampling_frame: Optional[Array] = None,
                            total_duration: Optional[float] = None,
                            precision: float = 0.1,
                            normalize: bool = False,
                            summate: bool = True,
                            **kwargs: object) -> Array:
    """Convolve a continuous EventVariable with an HRF.

    The event's (possibly centered/scaled) ``values`` are used as
    amplitudes during convolution, producing a single column.

    Parameters
    ----------
    event : EventVariable
        Continuous event to convolve.
    hrf : HRF, array, str, or None
        HRF specification.
    sampling_rate : float
        Sampling rate in Hz.
    sampling_frame : array-like, optional
        Explicit sampling time points.
    total_duration : float, optional
        Total signal duration in seconds.
    precision : float
        Temporal precision for fmrimod convolution.
    normalize : bool
        Peak-normalize after convolution.
    summate : bool
        Sum overlapping responses (True) or take max (False).

    Returns
    -------
    Array
        Convolved column of shape ``(n_timepoints, 1)``.
    """
    # Determine sampling grid
    if sampling_frame is not None:
        grid = _prepare_sampling_grid(sampling_frame)
    else:
        sampling_rate = _validate_sampling_rate(sampling_rate)
        # Determine total duration
        if total_duration is None:
            total_duration = np.max(event.onsets + event.durations) + 32.0
        total_duration = _validate_total_duration(total_duration)
        grid = _prepare_sampling_grid(np.arange(0, total_duration, 1/sampling_rate))

    # Check if we should use fmrimod or fallback
    use_fmrimod = HAS_PYFMRIHRF
    if use_fmrimod and (isinstance(hrf, np.ndarray) or (callable(hrf) and not hasattr(hrf, 'nbasis'))):
        # Array or plain callable HRF - use fallback path
        use_fmrimod = False

    if use_fmrimod:
        # Use fmrimod.regressor()
        result = _convolve_with_regressor(
            event.onsets,
            event.values,  # Use actual values
            event.durations,
            hrf,
            grid,
            precision=precision,
            summate=summate
        )
        # Ensure column vector
        if result.ndim == 1:
            result = result.reshape(-1, 1)
    else:
        # Fallback to manual convolution
        _validate_single_point_sampling_rate(grid, sampling_rate)
        fallback_rate = _effective_sampling_rate_from_grid(grid, sampling_rate)
        hrf_array = _get_hrf_array(hrf, fallback_rate)
        result = _convolve_impulses_on_grid(
            event.onsets,
            event.values,
            event.durations,
            hrf_array,
            fallback_rate,
            grid,
        )
        result = result.reshape(-1, 1)

    if normalize:
        result = _peak_normalize(result)

    return result


@convolve.register(EventMatrix)
def _convolve_event_matrix(event: EventMatrix, hrf=None,
                          sampling_rate: float = 1.0,
                          sampling_frame: Optional[Array] = None,
                          total_duration: Optional[float] = None,
                          precision: float = 0.1,
                          normalize: bool = False,
                          summate: bool = True,
                          **kwargs: object) -> Array:
    """Convolve a multi-column EventMatrix with an HRF.

    Each column of the matrix is convolved independently, preserving
    the number of columns.

    Parameters
    ----------
    event : EventMatrix
        Multi-column event to convolve.
    hrf : HRF, array, str, or None
        HRF specification.
    sampling_rate : float
        Sampling rate in Hz.
    sampling_frame : array-like, optional
        Explicit sampling time points.
    total_duration : float, optional
        Total signal duration in seconds.
    precision : float
        Temporal precision for fmrimod convolution.
    normalize : bool
        Peak-normalize each column after convolution.
    summate : bool
        Sum overlapping responses (True) or take max (False).

    Returns
    -------
    Array
        Convolved array of shape ``(n_timepoints, n_columns)``.
    """
    # Determine sampling grid
    if sampling_frame is not None:
        grid = _prepare_sampling_grid(sampling_frame)
    else:
        sampling_rate = _validate_sampling_rate(sampling_rate)
        # Determine total duration
        if total_duration is None:
            total_duration = np.max(event.onsets + event.durations) + 32.0
        total_duration = _validate_total_duration(total_duration)
        grid = _prepare_sampling_grid(np.arange(0, total_duration, 1/sampling_rate))

    n_cols = event.n_columns
    convolved = np.zeros((len(grid), n_cols))

    # Check if we should use fmrimod or fallback
    use_fmrimod = HAS_PYFMRIHRF
    if use_fmrimod and (isinstance(hrf, np.ndarray) or (callable(hrf) and not hasattr(hrf, 'nbasis'))):
        # Array or plain callable HRF - use fallback path
        use_fmrimod = False

    if use_fmrimod:
        # Use fmrimod.regressor() for each column
        for i in range(n_cols):
            result = _convolve_with_regressor(
                event.onsets,
                event.values[:, i],
                event.durations,
                hrf,
                grid,
                precision=precision,
                summate=summate
            )
            # Handle single-basis case
            if result.ndim == 1:
                convolved[:, i] = result
            else:
                convolved[:, i] = result[:, 0]
    else:
        # Fallback to manual convolution
        _validate_single_point_sampling_rate(grid, sampling_rate)
        fallback_rate = _effective_sampling_rate_from_grid(grid, sampling_rate)
        hrf_array = _get_hrf_array(hrf, fallback_rate)

        for i in range(n_cols):
            convolved[:, i] = _convolve_impulses_on_grid(
                event.onsets,
                event.values[:, i],
                event.durations,
                hrf_array,
                fallback_rate,
                grid,
            )

    if normalize:
        convolved = _peak_normalize(convolved)

    return convolved


@convolve.register(EventBasis)
def _convolve_event_basis(event: EventBasis, hrf=None,
                         sampling_rate: float = 1.0,
                         sampling_frame: Optional[Array] = None,
                         total_duration: Optional[float] = None,
                         precision: float = 0.1,
                         normalize: bool = False,
                         summate: bool = True,
                         **kwargs: object) -> Array:
    """Convolve EventBasis with HRF.

    For EventBasis, the HRF parameter is typically None since the basis
    functions themselves represent the HRF. This function evaluates the
    basis at the specified sampling times.
    """
    # Determine sampling grid
    if sampling_frame is not None:
        grid = _prepare_sampling_grid(sampling_frame)
    else:
        sampling_rate = _validate_sampling_rate(sampling_rate)
        # Determine total duration
        if total_duration is None:
            total_duration = np.max(event.onsets + event.durations) + 32.0
        total_duration = _validate_total_duration(total_duration)
        grid = _prepare_sampling_grid(np.arange(0, total_duration, 1/sampling_rate))

    if HAS_PYFMRIHRF:
        # EventBasis represents multiple basis functions
        # Each event should be convolved with each basis function
        n_events = len(event.onsets)
        n_basis = event.n_basis

        # For EventBasis, we treat each basis function as an HRF
        # The event.values matrix contains the coefficients for each basis
        convolved = np.zeros((len(grid), n_events * n_basis))

        for i in range(n_events):
            for j in range(n_basis):
                # Get the basis function (HRF) for this column
                basis_hrf = event.basis.functions[j] if hasattr(event.basis, 'functions') else hrf

                # Convolve single event with this basis function
                result = _convolve_with_regressor(
                    event.onsets[i:i+1],
                    event.expanded_values[i:i+1, j:j+1].flatten(),
                    event.durations[i:i+1],
                    basis_hrf,
                    grid,
                    precision=precision,
                    summate=summate
                )
                # Handle single-basis case
                if result.ndim == 1:
                    convolved[:, i * n_basis + j] = result
                else:
                    convolved[:, i * n_basis + j] = result[:, 0]

        if normalize:
            convolved = _peak_normalize(convolved)

        return convolved
    else:
        # Fallback implementation
        # For EventBasis without fmrimod, we need sampling_frame
        if sampling_frame is None:
            raise NotImplementedError(
                "EventBasis convolution without fmrimod requires sampling_frame. "
                "Either provide sampling_frame or install fmrimod."
            )

        # Simplified fallback: treat as EventMatrix
        _validate_single_point_sampling_rate(grid, sampling_rate)
        fallback_rate = _effective_sampling_rate_from_grid(grid, sampling_rate)
        hrf_array = _get_hrf_array(hrf, fallback_rate)
        n_basis = event.n_basis
        convolved = np.zeros((len(grid), n_basis))

        for i in range(n_basis):
            convolved[:, i] = _convolve_impulses_on_grid(
                event.onsets,
                event.expanded_values[:, i],
                event.durations,
                hrf_array,
                fallback_rate,
                grid,
            )

        if normalize:
            convolved = _peak_normalize(convolved)

        return convolved


@convolve.register(list)
def _convolve_list(events: List[EventProtocol], hrf=None,
                   sampling_rate: float = 1.0,
                   sampling_frame: Optional[Array] = None,
                   total_duration: Optional[float] = None,
                   normalize: bool = False,
                   summate: bool = True,
                   **kwargs: object) -> List[Array]:
    """Convolve a list of event objects with an HRF.

    Each element is convolved independently via :func:`convolve`.

    Parameters
    ----------
    events : list of event objects
        Events to convolve.
    hrf : HRF, array, str, or None
        HRF specification applied to every event.
    sampling_rate : float
        Sampling rate in Hz.
    sampling_frame : array-like, optional
        Explicit sampling time points.
    total_duration : float, optional
        Total signal duration in seconds.
    normalize : bool
        Peak-normalize each result.
    summate : bool
        Sum overlapping responses.

    Returns
    -------
    list of Array
        One convolved array per input event.
    """
    return [convolve(event, hrf, sampling_rate, sampling_frame, total_duration,
                     normalize=normalize, summate=summate, **kwargs)
            for event in events]


# Register for numpy arrays (assume impulse times and values)
@convolve.register(np.ndarray)
def _convolve_array(arr: np.ndarray, hrf=None,
                   sampling_rate: float = 1.0,
                   sampling_frame: Optional[Array] = None,
                   total_duration: Optional[float] = None,
                   precision: float = 0.1,
                   normalize: bool = False,
                   summate: bool = True,
                   **kwargs: object) -> Array:
    """Convolve array with HRF.

    Assumes array has shape (n_events, 3) with columns:
    [onset, duration, value]
    """
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            "Array must have shape (n_events, 3) with columns "
            "[onset, duration, value]"
        )
    if arr.shape[0] == 0:
        raise ValueError("Array must contain at least one event row")

    arr = np.asarray(arr, dtype=float)
    onsets = arr[:, 0]
    durations = arr[:, 1]
    values = arr[:, 2]

    if not np.all(np.isfinite(onsets)):
        raise ValueError("Array event onsets must be finite")
    if not np.all(np.isfinite(durations)):
        raise ValueError("Array event durations must be finite")
    if not np.all(np.isfinite(values)):
        raise ValueError("Array event values must be finite")
    if np.any(onsets < 0):
        raise ValueError("Array event onsets must be non-negative")
    if np.any(durations < 0):
        raise ValueError("Array event durations must be non-negative")

    # Determine sampling grid
    if sampling_frame is not None:
        grid = _prepare_sampling_grid(sampling_frame)
    else:
        sampling_rate = _validate_sampling_rate(sampling_rate)
        # Determine total duration
        if total_duration is None:
            total_duration = np.max(arr[:, 0] + arr[:, 1]) + 32.0
        total_duration = _validate_total_duration(total_duration)
        grid = _prepare_sampling_grid(np.arange(0, total_duration, 1/sampling_rate))

    # Check if we should use fmrimod or fallback
    use_fmrimod = HAS_PYFMRIHRF
    if use_fmrimod and (isinstance(hrf, np.ndarray) or (callable(hrf) and not hasattr(hrf, 'nbasis'))):
        # Array or plain callable HRF - use fallback path
        use_fmrimod = False

    if use_fmrimod:
        # Use fmrimod.regressor()
        result = _convolve_with_regressor(
            arr[:, 0],  # onsets
            arr[:, 2],  # values
            arr[:, 1],  # durations
            hrf,
            grid,
            precision=precision,
            summate=summate
        )
    else:
        # Fallback to manual convolution
        _validate_single_point_sampling_rate(grid, sampling_rate)
        fallback_rate = _effective_sampling_rate_from_grid(grid, sampling_rate)
        hrf_array = _get_hrf_array(hrf, fallback_rate)

        result = _convolve_impulses_on_grid(
            arr[:, 0],  # onsets
            arr[:, 2],  # values
            arr[:, 1],  # durations
            hrf_array,
            fallback_rate,
            grid,
        )

    if normalize:
        result = _peak_normalize(result)

    return result
