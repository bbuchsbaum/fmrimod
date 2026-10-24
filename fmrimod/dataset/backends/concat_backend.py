"""Row-wise concatenation of storage backends (runs, scans or files)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from fmrimod.dataset.backend_protocol import (
    BackendDims,
    StorageBackend,
    _normalize_indices,
)
from fmrimod.dataset.errors import ConfigError
from fmrimod.dataset.space import VolumeSpace


class ConcatBackend(StorageBackend):
    """Stack backends that share their columns along the time axis.

    Every part must have the same grid, affine and mask as part 0 (spatial
    identity, never shape alone); the error names the offending part. Reads
    are routed to the parts that hold the requested rows, in request order.
    """

    def __init__(self, parts: Sequence[StorageBackend], *, part_label: str = "Run") -> None:
        if not parts:
            raise ConfigError("ConcatBackend requires at least one backend")
        self._parts = list(parts)
        ref = self._parts[0].space
        for i, part in enumerate(self._parts[1:], start=1):
            ref.require_same_columns(part.space, f"{part_label} {i}")
        self._space = ref
        self._lengths = tuple(int(p.get_dims().time) for p in self._parts)
        self._starts = np.concatenate([[0], np.cumsum(self._lengths)]).astype(np.intp)

    @property
    def parts(self) -> tuple[StorageBackend, ...]:
        return tuple(self._parts)

    def open(self) -> None:
        for part in self._parts:
            part.open()

    def close(self) -> None:
        for part in self._parts:
            part.close()

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
        n_cols = self._space.n_voxels
        col_idx = _normalize_indices(cols, n_cols, "cols")
        row_idx = _normalize_indices(rows, int(self._starts[-1]), "rows")
        if row_idx is None:
            return np.vstack([p.get_data(cols=col_idx) for p in self._parts])
        out = np.empty((row_idx.size, n_cols if col_idx is None else col_idx.size))
        part_of = np.searchsorted(self._starts, row_idx, side="right") - 1
        for k in np.unique(part_of):
            where = np.flatnonzero(part_of == k)
            local = row_idx[where] - self._starts[k]
            out[where] = self._parts[int(k)].get_data(rows=local, cols=col_idx)
        return out

    def get_metadata(self) -> dict[str, object]:
        return {"format": "concat", "parts": [p.get_metadata() for p in self._parts]}


__all__ = ["ConcatBackend"]
