"""In-memory matrix storage backend."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fmrimod.dataset.backend_protocol import BackendDims, StorageBackend
from fmrimod.dataset.errors import ConfigError


class MatrixBackend(StorageBackend):
    """Backend that wraps a 2-D NumPy matrix."""

    def __init__(
        self,
        data_matrix: NDArray[np.floating[Any]],
        mask: NDArray[np.bool_] | None = None,
        spatial_dims: tuple[int, int, int] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        data = np.asarray(data_matrix, dtype=np.float64)
        if data.ndim != 2:
            raise ConfigError(
                "data_matrix must be a 2-D array",
                parameter="data_matrix",
                value=data.ndim,
            )

        self._data = data
        n_voxels = int(data.shape[1])
        self._mask = (
            np.ones(n_voxels, dtype=np.bool_)
            if mask is None
            else np.asarray(mask, dtype=np.bool_)
        )
        if self._mask.shape != (n_voxels,):
            raise ConfigError(
                f"mask length ({self._mask.size}) must equal "
                f"number of columns ({n_voxels})",
                parameter="mask",
            )

        self._spatial_dims = spatial_dims or (n_voxels, 1, 1)
        if len(self._spatial_dims) != 3:
            raise ConfigError(
                "spatial_dims must have exactly 3 elements",
                parameter="spatial_dims",
                value=self._spatial_dims,
            )
        if int(np.prod(self._spatial_dims)) != n_voxels:
            raise ConfigError(
                f"prod(spatial_dims) ({int(np.prod(self._spatial_dims))}) "
                f"must equal number of voxels ({n_voxels})",
                parameter="spatial_dims",
            )

        self._metadata = dict(metadata or {})

    def open(self) -> None:
        """No-op for in-memory data."""

    def close(self) -> None:
        """No-op for in-memory data."""

    def get_dims(self) -> BackendDims:
        """Return matrix dimensions."""
        return BackendDims(spatial=self._spatial_dims, time=int(self._data.shape[0]))

    def get_mask(self) -> NDArray[np.bool_]:
        """Return a copy of the flat mask."""
        return self._mask.copy()

    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Return masked matrix data with optional row/column selection."""
        data = self._data[:, self._mask]
        if rows is not None:
            data = data[np.asarray(rows, dtype=np.intp), :]
        if cols is not None:
            data = data[:, np.asarray(cols, dtype=np.intp)]
        return data

    def get_metadata(self) -> dict[str, object]:
        """Return matrix metadata."""
        return {"format": "matrix", **self._metadata}
