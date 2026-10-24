"""Storage-backend view of a run-wise :class:`DatasetProtocol` object.

User code may implement the run-wise protocol (``get_data(run)``,
``get_mask()``, ``get_sampling_frame()``, ...) directly. This wrapper is how
:class:`~fmrimod.dataset.FmriDataset` reads such an object through the one
storage contract; fmrimod's own sources are backends already.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from fmrimod.dataset.backend_protocol import (
    BackendDims,
    StorageBackend,
    _normalize_indices,
)
from fmrimod.dataset.space import VolumeSpace


class RunwiseSourceBackend(StorageBackend):
    """Adapt a run-wise source to ``get_data(rows, cols)`` reads."""

    def __init__(self, source: Any) -> None:
        self._source = source
        self._lengths = tuple(int(v) for v in source.n_timepoints)
        self._starts = np.concatenate([[0], np.cumsum(self._lengths)]).astype(np.intp)
        mask = np.asarray(source.get_mask(), dtype=bool)
        affine = None
        get_affine = getattr(source, "get_affine", None)
        if callable(get_affine):
            raw = get_affine()
            if raw is not None and np.shape(raw) == (4, 4):
                affine = np.asarray(raw, dtype=np.float64)
        if mask.ndim == 3:
            shape = cast("tuple[int, int, int]", tuple(int(v) for v in mask.shape))
        else:
            shape = (int(mask.size), 1, 1)
        self._space = VolumeSpace(
            shape, mask.reshape(-1), affine=affine,
            mask_shape=mask.shape if mask.ndim > 1 else None,
        )

    @property
    def source(self) -> Any:
        """The wrapped run-wise object."""
        return self._source

    def open(self) -> None:
        """No-op: run-wise sources manage their own resources."""

    def close(self) -> None:
        """No-op."""

    def get_dims(self) -> BackendDims:
        return BackendDims(spatial=self._space.shape, time=int(self._starts[-1]))

    def get_mask(self) -> NDArray[np.bool_]:
        return self._space.mask.copy()

    @property
    def space(self) -> VolumeSpace:
        return self._space

    @property
    def source_run_lengths(self) -> tuple[int, ...]:
        return self._lengths

    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        col_idx = _normalize_indices(cols, self._space.n_voxels, "cols")
        row_idx = _normalize_indices(rows, int(self._starts[-1]), "rows")

        def run(r: int) -> NDArray[np.float64]:
            data = np.asarray(self._source.get_data(r), dtype=np.float64)
            return data if col_idx is None else data[:, col_idx]

        if row_idx is None:
            return np.vstack([run(r) for r in range(len(self._lengths))])
        n_cols = self._space.n_voxels if col_idx is None else col_idx.size
        out = np.empty((row_idx.size, n_cols))
        run_of = np.searchsorted(self._starts, row_idx, side="right") - 1
        for r in np.unique(run_of):
            where = np.flatnonzero(run_of == r)
            out[where] = run(int(r))[row_idx[where] - self._starts[r]]
        return out

    def get_metadata(self) -> dict[str, object]:
        return {"format": "runwise", "source": type(self._source).__name__}


__all__ = ["RunwiseSourceBackend"]
