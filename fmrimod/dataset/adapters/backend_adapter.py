"""Adapter from canonical storage backends to the current dataset protocol."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fmrimod.dataset.backend_protocol import StorageBackend
from fmrimod.sampling import SamplingFrame


class BackendAdapter:
    """Adapt a matrix-oriented storage backend to run-wise dataset access."""

    def __init__(self, backend: StorageBackend, sampling_frame: SamplingFrame) -> None:
        self.backend = backend
        self._sampling_frame = sampling_frame
        if backend.get_dims().time != sampling_frame.n_scans:
            raise ValueError(
                "backend time dimension must match sampling frame scan count"
            )

    def get_data(self, run: int) -> NDArray[np.float64]:
        """Return run data as ``time x voxels``."""
        if run < 0 or run >= self.n_runs:
            raise IndexError(f"Run {run} out of range [0, {self.n_runs})")
        start = int(np.sum(self.run_lengths[:run]))
        end = start + int(self.run_lengths[run])
        data = self.backend.get_data(rows=np.arange(start, end, dtype=np.intp))
        return np.asarray(data, dtype=np.float64)

    def get_mask(self) -> NDArray[np.bool_]:
        """Return the backend mask."""
        return self.backend.get_mask()

    def get_sampling_frame(self) -> SamplingFrame:
        """Return the sampling frame."""
        return self._sampling_frame

    @property
    def n_runs(self) -> int:
        return int(self._sampling_frame.n_blocks)

    @property
    def run_lengths(self) -> list[int]:
        return [int(v) for v in self._sampling_frame.blocklens]

    @property
    def n_timepoints(self) -> list[int]:
        return self.run_lengths

    @property
    def n_voxels(self) -> int:
        return int(self.backend.get_mask().sum())
