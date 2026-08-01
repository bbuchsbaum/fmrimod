"""Neural input generation for fMRI regressors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .core import Regressor
else:
    Regressor = Any


@dataclass(frozen=True)
class NeuralInput:
    """Time-locked neural input series produced by :func:`neural_input`.

    Attributes
    ----------
    time : NDArray[np.float64]
        Time grid in seconds.
    values : NDArray[np.float64]
        Neural input values, same length as ``time``.
    """

    time: NDArray[np.float64]
    values: NDArray[np.float64]


def neural_input_core(
    onsets: NDArray[np.float64],
    durations: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    start: float,
    end: float,
    resolution: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate neural input time series from events.

    This function creates a boxcar time series representing neural activity,
    where each event contributes its amplitude during its duration.

    This is the *stimulus function* itself -- what the experiment delivered,
    for inspection and plotting. It is deliberately a unit-height boxcar and
    is **not** the quadrature-weighted fine-grid input the convolution path
    builds in :func:`fmrimod.regressor.convolution._build_neural_input`.
    Those two are different objects: the boxcar has units of amplitude, the
    convolution input carries the ``dt`` factors that turn the subsequent
    convolution into an integral. The R reference draws the same distinction
    -- ``neural_input_rcpp`` stayed a plain boxcar through the fmrihrf#45
    quadrature fix, which only touched ``buildImpulseTrain``. Do not "fix"
    this one to match; they are not supposed to match.

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


# ``neural_input_fast()`` used to live here: a second difference-array builder
# for the same stimulus function. It had no callers, was not exported, and did
# not agree with ``neural_input_core()`` -- it binned event edges by integer
# index where the core builder masks on time, so for onsets/durations that fell
# between bins the two returned different boxcars (summed amplitude 23 vs 20 on
# a two-event example). A faster-looking duplicate that silently computes
# something else is a trap, so it is gone. Removed under fmrimod#17.


def neural_input(
    reg: "Regressor",
    start: float = 0.0,
    end: Optional[float] = None,
    resolution: float = 0.33,
) -> NeuralInput:
    """Generate neural input time series from a Regressor.

    Args:
        reg: Regressor object
        start: Start time in seconds
        end: End time in seconds (if None, auto-determined)
        resolution: Time resolution in seconds

    Returns:
        :class:`NeuralInput` with ``time`` and ``values`` arrays.

    Examples:
        >>> reg = regressor(onsets=[10, 30, 50], duration=2)
        >>> ni = neural_input(reg, start=0, end=60)
        >>> plt.plot(ni.time, ni.values)
    """
    time, values = reg.neural_input(start, end, resolution)
    return NeuralInput(time=time, values=values)
