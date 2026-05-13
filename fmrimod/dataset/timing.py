"""Timing helpers for canonical dataset contracts."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..sampling import SamplingFrame


def sample_indices(frame: SamplingFrame, *, one_based: bool = True) -> NDArray[np.intp]:
    """Return integer sample indices for a sampling frame.

    ``SamplingFrame.samples`` is reserved for acquisition times in seconds.
    This helper provides the explicit integer-index surface needed for
    fmridataset compatibility.
    """
    start = 1 if one_based else 0
    stop = int(frame.n_scans) + start
    return np.arange(start, stop, dtype=np.intp)
