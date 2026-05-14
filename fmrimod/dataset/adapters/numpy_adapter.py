"""Numpy adapter: wraps raw ndarrays into DatasetProtocol."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ...sampling import SamplingFrame


class NumpyAdapter:
    r"""Adapt raw numpy arrays to the :class:`DatasetProtocol`.

    This is the simplest adapter — it takes a list of 2-D arrays
    (one per run) plus a :class:`SamplingFrame` and optional mask.

    Parameters
    ----------
    data : list of NDArray or single NDArray
        Per-run data matrices, each of shape ``(n_timepoints, n_voxels)``.
        If a single 2-D array is provided it is treated as one run.
    sampling_frame : SamplingFrame
        Temporal sampling specification.
    mask : NDArray[np.bool\_], optional
        3-D boolean mask.  If *None*, a flat mask of all ``True`` is used
        with shape ``(n_voxels, 1, 1)``.
    """

    def __init__(
        self,
        data: Union[NDArray[np.float64], List[NDArray[np.float64]]],
        sampling_frame: SamplingFrame,
        mask: Optional[NDArray[np.bool_]] = None,
    ):
        # Normalise to list of 2-D arrays
        if isinstance(data, np.ndarray) and data.ndim == 2:
            data = [data]
        self._data: List[NDArray[np.float64]] = [
            np.asarray(d, dtype=np.float64) for d in data
        ]

        # Validate dimensions
        for i, d in enumerate(self._data):
            if d.ndim != 2:
                raise ValueError(
                    f"Run {i}: expected 2-D array (time x voxels), got shape {d.shape}"
                )
        # All runs must have the same number of voxels
        nvox = self._data[0].shape[1]
        for i, d in enumerate(self._data):
            if d.shape[1] != nvox:
                raise ValueError(
                    f"Run {i} has {d.shape[1]} voxels, expected {nvox}"
                )

        self._sampling_frame = sampling_frame

        # Validate run counts match
        if len(self._data) != len(sampling_frame.blocklens):
            raise ValueError(
                f"Number of data arrays ({len(self._data)}) does not match "
                f"number of blocks in sampling_frame ({len(sampling_frame.blocklens)})"
            )
        # Validate timepoints per run
        for i, (d, bl) in enumerate(zip(self._data, sampling_frame.blocklens)):
            if d.shape[0] != bl:
                raise ValueError(
                    f"Run {i}: data has {d.shape[0]} timepoints but "
                    f"sampling_frame expects {bl}"
                )

        if mask is not None:
            self._mask = np.asarray(mask, dtype=bool)
        else:
            # Default flat mask
            self._mask = np.ones((nvox, 1, 1), dtype=bool)

    # -- DatasetProtocol implementation --

    def get_data(self, run: int) -> NDArray[np.float64]:
        """Return data matrix for the given run."""
        if run < 0 or run >= len(self._data):
            raise IndexError(f"Run {run} out of range [0, {len(self._data)})")
        return self._data[run]

    def get_mask(self) -> NDArray[np.bool_]:
        """Return the spatial mask."""
        return self._mask

    def get_sampling_frame(self) -> SamplingFrame:
        """Return the sampling frame."""
        return self._sampling_frame

    @property
    def n_runs(self) -> int:
        return len(self._data)

    @property
    def n_timepoints(self) -> List[int]:
        return [d.shape[0] for d in self._data]

    @property
    def n_voxels(self) -> int:
        return self._data[0].shape[1]
