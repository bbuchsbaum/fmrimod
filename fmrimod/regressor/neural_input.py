"""Neural input generation for fMRI regressors."""

from __future__ import annotations

from typing import Tuple, Optional, TYPE_CHECKING, Any
import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .core import Regressor
else:
    Regressor = Any


def neural_input_core(
    onsets: NDArray[np.float64],
    durations: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    start: float,
    end: float,
    resolution: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate neural input time series from events.
    
    This function creates a boxcar time series representing neural activity,
    where each event contributes its amplitude during its duration.
    
    Args:
        onsets: Event onset times
        durations: Event durations
        amplitudes: Event amplitudes
        start: Start time
        end: End time
        resolution: Time resolution
        
    Returns:
        Tuple of (time_points, neural_input_values)
    """
    # Create time grid
    n_points = int((end - start) / resolution) + 1
    time = np.linspace(start, end, n_points)
    
    # Initialize neural input
    neural = np.zeros(n_points)
    
    # For each event, add amplitude during its duration
    for onset, duration, amplitude in zip(onsets, durations, amplitudes):
        if duration > 0:
            # Find indices within event duration
            event_mask = (time >= onset) & (time < onset + duration)
            neural[event_mask] += amplitude
        else:
            # Impulse event - find closest time point
            idx = np.argmin(np.abs(time - onset))
            if 0 <= idx < n_points:
                neural[idx] += amplitude
    
    return time, neural


def neural_input_fast(
    onsets: NDArray[np.float64],
    durations: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    start: float,
    end: float,
    resolution: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fast neural input generation using difference array method.
    
    This is an optimized O(E + N) algorithm where E is number of events
    and N is number of time points.
    
    Args:
        onsets: Event onset times
        durations: Event durations
        amplitudes: Event amplitudes
        start: Start time
        end: End time
        resolution: Time resolution
        
    Returns:
        Tuple of (time_points, neural_input_values)
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    
    # Number of time bins
    n_bins = int((end - start) / resolution) + 1
    
    # Initialize difference array with extra guard slot
    diff = np.zeros(n_bins + 1)
    
    # Process each event
    for onset, duration, amplitude in zip(onsets, durations, amplitudes):
        # Calculate start bin index
        if onset <= start:
            start_idx = 0
        else:
            start_idx = int((onset - start) / resolution)
        
        # Skip if event starts after our time range
        if start_idx >= n_bins:
            continue
        
        # Calculate end bin index
        end_idx = int((onset + duration - start) / resolution)
        if end_idx >= n_bins:
            end_idx = n_bins - 1
        
        # Skip if invalid range
        if start_idx > end_idx:
            continue
        
        # Add amplitude at start, subtract at end+1
        diff[start_idx] += amplitude
        diff[end_idx + 1] -= amplitude
    
    # Cumulative sum to get actual values (exclude guard slot)
    neural = np.cumsum(diff[:-1])
    
    # Create time array
    time = np.arange(n_bins) * resolution + start
    
    return time, neural


def neural_input(
    reg: "Regressor",
    start: float = 0.0,
    end: Optional[float] = None,
    resolution: float = 0.33
) -> dict:
    """Generate neural input time series from a Regressor.
    
    Args:
        reg: Regressor object
        start: Start time in seconds
        end: End time in seconds (if None, auto-determined)
        resolution: Time resolution in seconds
        
    Returns:
        Dictionary with 'time' and 'neural_input' arrays
        
    Examples:
        >>> reg = regressor(onsets=[10, 30, 50], duration=2)
        >>> input = neural_input(reg, start=0, end=60)
        >>> plt.plot(input['time'], input['neural_input'])
    """
    # Use the regressor's neural_input method
    time, neural = reg.neural_input(start, end, resolution)
    
    return {
        'time': time,
        'neural_input': neural
    }
