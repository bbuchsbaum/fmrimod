"""Convolution methods for HRF and neural input."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import scipy.signal
from numpy.typing import NDArray

from ..hrf import HRF
from ..utils.cache import cached_hrf_eval

# There is one convolution engine. ``"conv"`` is it.
#
# The module used to advertise three: ``"conv"`` and ``"direct"`` dispatched to
# the identical ``scipy.signal.convolve`` call, and ``"fft"`` reimplemented that
# convolution by hand over ``scipy.fft`` with power-of-two zero-padding. All
# three agreed to ~1e-15, so the public API offered three names for one
# computation.
#
# Benchmarked on this side rather than inheriting the R result (fmrimod#17):
# across 200-2000 volumes, 20-1000 events, 1-5 basis functions and precision
# 0.33 down to 0.01, the hand-rolled FFT was slower than ``conv`` in six of
# seven configurations (1.1x-2.2x) and faster in one. It has nothing to offer,
# because ``scipy.signal.convolve`` defaults to ``method="auto"`` and already
# calls ``scipy.signal.choose_conv_method`` to switch to FFT on exactly the
# large inputs where FFT wins — verified: it picks ``fft`` at 45k x 481 and
# 225k x 2401, ``direct`` below that.
#
# ``"fft"`` and ``"direct"`` remain accepted as deprecated aliases so existing
# call sites keep working; they warn and route to ``"conv"``.
ConvolutionMethod = Literal["conv", "fft", "direct"]
CANONICAL_CONVOLUTION_METHOD: ConvolutionMethod = "conv"
DEPRECATED_CONVOLUTION_METHODS: tuple[ConvolutionMethod, ...] = ("fft", "direct")
CONVOLUTION_METHODS: tuple[ConvolutionMethod, ...] = ("conv", "fft", "direct")


def validate_convolution_method(method: str) -> ConvolutionMethod:
    """Return *method* as a closed convolution method or raise clearly.

    ``"fft"`` and ``"direct"`` are deprecated aliases for ``"conv"``: they warn
    and are folded to it, so the returned value is always ``"conv"``.
    """
    if method not in CONVOLUTION_METHODS:
        raise ValueError(
            f"method must be one of 'conv', 'fft', 'direct'; got {method!r}."
        )
    if method in DEPRECATED_CONVOLUTION_METHODS:
        warnings.warn(
            f"convolution method {method!r} is deprecated and now evaluates via "
            f"'conv', the only engine. 'direct' was always the same "
            f"scipy.signal.convolve call as 'conv'; the hand-rolled 'fft' was "
            f"redundant with scipy's own direct/FFT selection and slower in "
            f"almost every measured configuration. Pass method='conv' (the "
            f"default) instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return CANONICAL_CONVOLUTION_METHOD


def convolve_hrf(
    grid: NDArray[np.float64],
    onsets: NDArray[np.float64],
    durations: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    hrf: HRF,
    span: float,
    precision: float = 0.33,
    method: ConvolutionMethod = "conv",
    summate: bool = True,
) -> NDArray[np.float64]:
    """Convolve neural input with HRF to generate predicted response.

    This function creates a neural input time series from events and
    convolves it with the HRF to produce the predicted fMRI response.

    Args:
        grid: Time points at which to evaluate (seconds)
        onsets: Event onset times
        durations: Event durations
        amplitudes: Event amplitudes
        hrf: Hemodynamic response function
        span: HRF temporal span
        precision: Temporal precision for convolution
        method: Convolution engine. ``'conv'`` is the only one; ``'fft'`` and
            ``'direct'`` are deprecated aliases that warn and route to it.
        summate: If True the block response accumulates over its duration; if
            False it is divided by the duration, giving the duration-averaged
            (unit-mass) response.

    Returns:
        Convolved response evaluated at grid points
        Shape is (n_grid, n_basis) where n_basis is hrf.nbasis
    """
    # Determine fine grid bounds (match R prep_reg_inputs / C++ wrappers)
    grid_min: float = float(np.min(grid))
    grid_max: float = float(np.max(grid))
    max_onset_plus_dur = np.max(onsets + durations) if len(onsets) > 0 else grid_max
    fine_start = grid_min - span
    fine_end = max(grid_max, max_onset_plus_dur) + span

    # Create fine time grid using fixed step size from fine_start
    n_fine = int(np.floor((fine_end - fine_start) / precision)) + 1
    fine_time = fine_start + np.arange(n_fine, dtype=np.float64) * precision

    # Generate neural input on fine grid
    neural = _build_neural_input(
        fine_time, onsets, durations, amplitudes, precision, summate=summate
    )

    # Evaluate HRF on fine grid (from 0 to span) with caching
    hrf_values = cached_hrf_eval(hrf, span, precision)  # Shape: (n_hrf_points, n_basis)

    # Perform convolution for each basis function
    n_basis = hrf_values.shape[1]
    result = np.zeros((len(grid), n_basis))

    # Folds the deprecated 'fft'/'direct' aliases to the single engine.
    method = validate_convolution_method(method)

    for b in range(n_basis):
        convolved = _convolve(neural, hrf_values[:, b])

        # Trim convolution result to match fine grid length
        convolved = convolved[:n_fine]

        # Interpolate to output grid
        result[:, b] = np.interp(grid, fine_time, convolved)

    # Return as 1D if single basis
    if n_basis == 1:
        result = result.ravel()

    return result


def _add_hat_step(
    diff: NDArray[np.float64],
    pos: float,
    coef: float,
    dt: float,
    n_bins: int,
) -> None:
    """Accumulate ``coef * A_j(pos)`` into a difference array.

    ``A_j`` is the running integral of the linear hat (tent) basis function
    centred on bin ``j``. A block over ``[s, e]`` projected onto the hat basis
    has bin weights ``A_j(e) - A_j(s)``, which is trapezoid quadrature with
    both endpoints placed at their exact sub-bin positions. Both properties
    matter: bin-aligned trapezoid weights quantise the block edges, and exact
    edges with plain rectangle weights drop the quadrature to first order.
    """
    if pos <= 0.0:
        return  # nothing accumulated yet
    max_pos = float(n_bins - 1)
    if pos > max_pos:
        pos = max_pos

    a = int(np.floor(pos))
    f = pos - float(a)
    v = coef * dt

    # Full weight on every bin strictly left of a.
    if a > 0:
        diff[0] += v
        diff[a] -= v

    # Partial weight on the two bins bracketing pos.
    wa = v * (0.5 + f - 0.5 * f * f)
    wa1 = v * (0.5 * f * f)
    diff[a] += wa
    diff[a + 1] -= wa
    diff[a + 1] += wa1
    diff[a + 2] -= wa1


def _build_neural_input(
    time: NDArray[np.float64],
    onsets: NDArray[np.float64],
    durations: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    dt: float,
    summate: bool = True,
) -> NDArray[np.float64]:
    """Build fine-grid neural input in O(E + N) with a difference array.

    Point events (``duration <= 0``) contribute a unit-mass impulse, so that
    convolution with the sampled HRF reproduces ``amplitude * h(t - onset)``
    independently of ``dt``.

    Block events (``duration > 0``) contribute the hat-basis projection of the
    boxcar, so that convolution approximates
    ``amplitude * integral_0^duration h(t - onset - u) du`` and converges as
    ``dt -> 0``. Previously the boxcar had unit height and no ``dt`` factor at
    all, so a block response was a bare sample count that grew as ``1/dt``:
    the amplitude of every epoch regressor was a function of ``precision``.

    Neither onsets nor block edges are snapped to the grid; both are placed at
    their exact sub-bin positions.

    With ``summate=False`` each block is divided by its total mass
    (``duration``), giving the duration-averaged response rather than the
    accumulated one.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive in _build_neural_input")

    n_bins = len(time)
    t0 = float(time[0])
    max_pos = float(n_bins - 1)

    # Difference array with two guard slots (see _add_hat_step).
    diff = np.zeros(n_bins + 2, dtype=np.float64)

    for onset, duration, amplitude in zip(onsets, durations, amplitudes):
        d = float(duration)
        pos_s = (float(onset) - t0) / dt
        if pos_s > max_pos:
            continue  # starts past the window

        if not (d > 0.0):
            # Impulse of unit mass, split linearly across the two bracketing
            # bins. Snapping it to the nearest bin instead cost up to dt/2 of
            # onset error, which at the default precision of 0.33 s was ~12%
            # of peak for an onset that fell between bins.
            if pos_s < 0.0:
                pos_s = 0.0
            a = int(np.floor(pos_s))
            f = pos_s - float(a)
            w0 = float(amplitude) * (1.0 - f)
            w1 = float(amplitude) * f
            diff[a] += w0
            diff[a + 1] -= w0
            diff[a + 1] += w1
            diff[a + 2] -= w1
            continue

        pos_e = (float(onset) + d - t0) / dt
        if pos_e <= 0.0:
            continue  # ends before the window

        # Block weights are the difference of the two running hat integrals,
        # so the block carries mass amplitude * duration for any sub-bin
        # alignment.
        c = float(amplitude) * (1.0 if summate else 1.0 / d)
        _add_hat_step(diff, pos_e, c, dt, n_bins)
        _add_hat_step(diff, pos_s, -c, dt, n_bins)

    # Cumulative sum to get actual values (drop the guard slots).
    neural = np.cumsum(diff[:n_bins])

    return np.asarray(neural, dtype=np.float64)


def _convolve(
    signal: NDArray[np.float64], kernel: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The convolution engine.

    ``scipy.signal.convolve`` defaults to ``method="auto"``, which consults
    ``choose_conv_method`` and switches to FFT on the input sizes where FFT
    actually wins. That subsumes the hand-rolled FFT path this module used to
    carry as a separate public method (see the note on ``ConvolutionMethod``).
    """
    return np.asarray(
        scipy.signal.convolve(signal, kernel, mode="full"), dtype=np.float64
    )


def convolve_hrf_per_event(
    grid: NDArray[np.float64],
    onsets: NDArray[np.float64],
    durations: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    hrfs: list[HRF],
    span: float,
    precision: float = 0.33,
    summate: bool = True,
) -> NDArray[np.float64]:
    """Per-event convolution for trial-varying (list) HRFs.

    Each event is convolved with its own HRF and the contributions are
    summed (or averaged, when *summate* is False).

    Args:
        grid: Output time points (seconds).
        onsets: Per-event onset times.
        durations: Per-event durations.
        amplitudes: Per-event amplitudes.
        hrfs: List of HRF objects, one per event.
        span: Maximum HRF span.
        precision: Fine-grid resolution.
        summate: Sum overlapping responses if True.

    Returns:
        Result array of shape ``(len(grid),)`` or ``(len(grid), nbasis)``.
    """
    n_basis = hrfs[0].nbasis if len(hrfs) > 0 else 1
    result = np.zeros((len(grid), n_basis))

    for onset, dur, amp, hrf in zip(onsets, durations, amplitudes, hrfs):
        # Determine which grid points this event can influence. A blocked
        # event responds out to span + duration, not span; masking at the bare
        # span truncated the tail of every blocked event, leaving an error
        # floor that did not shrink as precision decreased.
        rel_times = grid - onset
        event_span = span + max(float(dur), 0.0)
        mask = (rel_times >= 0) & (rel_times <= event_span)
        if not np.any(mask):
            continue

        # Evaluate the HRF at relative times for the affected grid points
        t_eval = rel_times[mask]
        hrf_vals = hrf.evaluate(
            t_eval, duration=dur, precision=precision, summate=summate
        )

        # Scale by amplitude
        if hrf_vals.ndim == 1:
            hrf_vals = hrf_vals[:, np.newaxis]
        hrf_vals = hrf_vals * amp

        # Event contributions are always additive across events in R's eval_loop.
        # `summate` controls within-event block behavior via HRF.evaluate().
        result[mask, :] += hrf_vals

    # Flatten if single basis
    if n_basis == 1:
        result = result.ravel()

    return result
