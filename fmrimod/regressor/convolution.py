"""Convolution methods for HRF and neural input."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import scipy.fft
import scipy.signal
from numpy.typing import NDArray

from ..hrf import HRF
from ..utils.cache import cached_hrf_eval

# Closed set of convolution backends. ``"conv"`` and ``"direct"`` both route
# through scipy.signal.convolve; ``"fft"`` uses scipy.fft.
ConvolutionMethod = Literal["conv", "fft", "direct"]
CONVOLUTION_METHODS: tuple[ConvolutionMethod, ...] = ("conv", "fft", "direct")


def validate_convolution_method(method: str) -> ConvolutionMethod:
    """Return *method* as a closed convolution method or raise clearly."""
    if method not in CONVOLUTION_METHODS:
        raise ValueError(
            f"method must be one of 'conv', 'fft', 'direct'; got {method!r}."
        )
    return cast(ConvolutionMethod, method)


def convolve_hrf(
    grid: NDArray[np.float64],
    onsets: NDArray[np.float64],
    durations: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    hrf: HRF,
    span: float,
    precision: float = 0.33,
    method: ConvolutionMethod = "conv",
    summate: bool = True
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
        method: Convolution method ('conv', 'fft', 'direct')
        summate: Whether overlapping responses sum
        
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
    neural = _build_neural_input(fine_time, onsets, durations, amplitudes, precision)
    
    # Evaluate HRF on fine grid (from 0 to span) with caching
    hrf_values = cached_hrf_eval(hrf, span, precision)  # Shape: (n_hrf_points, n_basis)
    
    # Perform convolution for each basis function
    n_basis = hrf_values.shape[1]
    result = np.zeros((len(grid), n_basis))
    
    method = validate_convolution_method(method)

    for b in range(n_basis):
        # Convolve neural input with this basis function
        if method == "fft":
            convolved = _convolve_fft(neural, hrf_values[:, b])
        else:  # "conv" and "direct" both use scipy.signal.convolve
            convolved = _convolve_direct(neural, hrf_values[:, b])
        
        # Trim convolution result to match fine grid length
        convolved = convolved[:n_fine]
        
        # Interpolate to output grid
        result[:, b] = np.interp(grid, fine_time, convolved)
    
    # Return as 1D if single basis
    if n_basis == 1:
        result = result.ravel()
    
    return result


def _build_neural_input(
    time: NDArray[np.float64],
    onsets: NDArray[np.float64],
    durations: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    dt: float
) -> NDArray[np.float64]:
    """Build neural input using efficient difference array method.
    
    This is an O(E + N) algorithm where E is number of events
    and N is number of time points.
    """
    n_bins = len(time)
    t0 = float(time[0])

    # Difference array with guard slot (matches C++ buildImpulseTrain)
    diff = np.zeros(n_bins + 1, dtype=np.float64)

    for onset, duration, amplitude in zip(onsets, durations, amplitudes):
        if onset <= t0:
            start_idx = 0
        else:
            start_idx = int(np.floor((onset - t0) / dt))

        # C++ clamp then skip if onset starts outside range
        if start_idx >= n_bins:
            start_idx = n_bins

        end_idx = int(np.floor((onset + duration - t0) / dt))
        if end_idx >= n_bins:
            end_idx = n_bins - 1

        if start_idx > end_idx and start_idx < n_bins:
            continue
        if start_idx == n_bins:
            continue

        diff[start_idx] += amplitude
        diff[end_idx + 1] -= amplitude
    
    # Cumulative sum to get actual values (exclude guard slot)
    neural = np.cumsum(diff[:n_bins])
    
    return np.asarray(neural, dtype=np.float64)


def _convolve_direct(
    signal: NDArray[np.float64],
    kernel: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Direct convolution using scipy.signal.convolve."""
    return np.asarray(scipy.signal.convolve(signal, kernel, mode="full"), dtype=np.float64)


def _convolve_fft(
    signal: NDArray[np.float64],
    kernel: NDArray[np.float64]
) -> NDArray[np.float64]:
    """FFT-based convolution for efficiency with long signals."""
    # Determine optimal FFT size (next power of 2)
    n_fft = _next_power_of_2(len(signal) + len(kernel) - 1)
    
    # Compute FFTs
    signal_fft = scipy.fft.fft(signal, n=n_fft)
    kernel_fft = scipy.fft.fft(kernel, n=n_fft)
    
    # Multiply in frequency domain and inverse transform
    result = scipy.fft.ifft(signal_fft * kernel_fft)
    
    # Return real part (imaginary part should be negligible)
    return np.asarray(np.real(result), dtype=np.float64)


def _next_power_of_2(n: int) -> int:
    """Find the next power of 2 greater than or equal to n."""
    if n <= 0:
        return 1

    # Bit manipulation trick
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    if n.bit_length() > 32:
        n |= n >> 32
    return n + 1


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
        # Determine which grid points this event can influence
        rel_times = grid - onset
        mask = (rel_times >= 0) & (rel_times <= span)
        if not np.any(mask):
            continue

        # Evaluate the HRF at relative times for the affected grid points
        t_eval = rel_times[mask]
        hrf_vals = hrf.evaluate(t_eval, duration=dur, precision=precision, summate=summate)

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
