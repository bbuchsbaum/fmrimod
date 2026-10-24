"""Run-wise facade over a canonical storage backend."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fmrimod.dataset.backend_protocol import StorageBackend
from fmrimod.sampling import SamplingFrame


class BackendAdapter:
    """Pair a storage backend with the sampling frame that splits it into runs.

    :class:`~fmrimod.dataset.FmriDataset` reads ``backend`` directly; the
    run-wise methods here exist for the :class:`DatasetProtocol` callers.
    """

    def __init__(self, backend: StorageBackend, sampling_frame: SamplingFrame) -> None:
        from fmrimod.dataset.fmri_dataset import check_run_lengths

        check_run_lengths(backend, sampling_frame)
        self.backend = backend
        self._sampling_frame = sampling_frame
        self._starts = np.concatenate([[0], np.cumsum(sampling_frame.blocklens)])

    def get_data(self, run: int) -> NDArray[np.float64]:
        """Return run data as ``time x voxels``."""
        if run < 0 or run >= self.n_runs:
            raise IndexError(f"Run {run} out of range [0, {self.n_runs})")
        rows = np.arange(self._starts[run], self._starts[run + 1], dtype=np.intp)
        return np.asarray(self.backend.get_data(rows=rows), dtype=np.float64)

    def get_mask(self) -> NDArray[np.bool_]:
        """Return the mask: in the grid's shape for images, flat for matrices."""
        return self.backend.space.mask_array()

    def get_affine(self) -> NDArray[np.float64] | None:
        """Return the backend's affine, or ``None`` when unknown."""
        return self.backend.space.affine

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
        return int(self.backend.space.n_voxels)
