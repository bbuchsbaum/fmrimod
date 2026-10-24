"""In-memory matrix storage backend."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fmrimod.dataset.backend_protocol import (
    BackendDims,
    StorageBackend,
    _normalize_indices,
)
from fmrimod.dataset.errors import ConfigError
from fmrimod.dataset.space import VolumeSpace


class MatrixBackend(StorageBackend):
    """Backend that wraps a 2-D NumPy matrix.

    ``data_matrix`` holds one column per grid cell and ``mask`` selects the
    cells that become data columns. :meth:`from_masked` takes a matrix that
    already holds only the in-mask columns. ``spatial_dims`` marks the data
    as a volume (the mask is then reported in that shape); ``affine`` gives
    the grid world coordinates when known.
    """

    def __init__(
        self,
        data_matrix: NDArray[np.float64],
        mask: NDArray[np.bool_] | None = None,
        spatial_dims: tuple[int, int, int] | None = None,
        metadata: dict[str, object] | None = None,
        *,
        affine: NDArray[np.float64] | None = None,
    ) -> None:
        data = np.asarray(data_matrix, dtype=np.float64)
        if data.ndim != 2:
            raise ConfigError(
                "data_matrix must be a 2-D array",
                parameter="data_matrix",
                value=data.ndim,
            )
        n_cells = int(data.shape[1])
        mask_vec = (
            np.ones(n_cells, dtype=np.bool_)
            if mask is None
            else np.asarray(mask, dtype=np.bool_).reshape(-1)
        )
        if mask_vec.shape != (n_cells,):
            raise ConfigError(
                f"mask length ({mask_vec.size}) must equal "
                f"number of columns ({n_cells})",
                parameter="mask",
            )
        masked = data if mask_vec.all() else data[:, mask_vec]
        self._init(masked, mask_vec, spatial_dims, metadata, affine)

    @classmethod
    def from_masked(
        cls,
        masked_data: NDArray[np.float64],
        mask: NDArray[np.bool_],
        spatial_dims: tuple[int, int, int] | None = None,
        metadata: dict[str, object] | None = None,
        *,
        affine: NDArray[np.float64] | None = None,
    ) -> MatrixBackend:
        """Wrap ``time x in-mask voxels`` data whose grid is described by ``mask``."""
        data = np.asarray(masked_data, dtype=np.float64)
        if data.ndim != 2:
            raise ConfigError("masked_data must be a 2-D array", parameter="masked_data")
        mask_arr = np.asarray(mask, dtype=np.bool_)
        # An in-memory mask keeps the shape it was given in (3-D grid, or
        # e.g. a 2-D slab); a flat mask stays flat.
        mask_shape = mask_arr.shape if mask_arr.ndim > 1 else None
        if spatial_dims is None and mask_arr.ndim > 1:
            padded = tuple(int(v) for v in mask_arr.shape) + (1, 1)
            spatial_dims = padded[:3] if mask_arr.ndim <= 3 else None  # type: ignore[assignment]
        mask_vec = mask_arr.reshape(-1)
        if int(mask_vec.sum()) != int(data.shape[1]):
            raise ConfigError(
                f"mask selects {int(mask_vec.sum())} voxels but the data has "
                f"{data.shape[1]} columns",
                parameter="mask",
            )
        out = cls.__new__(cls)
        out._init(data, mask_vec, spatial_dims, metadata, affine, mask_shape=mask_shape)
        return out

    def _init(
        self,
        masked: NDArray[np.float64],
        mask_vec: NDArray[np.bool_],
        spatial_dims: tuple[int, int, int] | None,
        metadata: dict[str, object] | None,
        affine: NDArray[np.float64] | None,
        *,
        mask_shape: tuple[int, ...] | None = None,
    ) -> None:
        n_cells = int(mask_vec.size)
        if mask_shape is None and spatial_dims is not None:
            mask_shape = tuple(spatial_dims)
        dims = tuple(spatial_dims) if spatial_dims is not None else (n_cells, 1, 1)
        if len(dims) != 3:
            raise ConfigError(
                "spatial_dims must have exactly 3 elements",
                parameter="spatial_dims",
                value=dims,
            )
        if int(np.prod(dims)) != n_cells:
            raise ConfigError(
                f"prod(spatial_dims) ({int(np.prod(dims))}) "
                f"must equal number of voxels ({n_cells})",
                parameter="spatial_dims",
            )
        # Reads hand out views of one cached matrix, so they are read-only;
        # the caller's own array is not frozen.
        view = masked.view()
        view.setflags(write=False)
        self._data = view
        self._mask = mask_vec
        self._spatial_dims = dims
        self._space = VolumeSpace(
            dims, mask_vec, affine=affine, mask_shape=mask_shape  # type: ignore[arg-type]
        )
        self._metadata = dict(metadata or {})

    def open(self) -> None:
        """No-op for in-memory data."""

    def close(self) -> None:
        """No-op for in-memory data."""

    def get_dims(self) -> BackendDims:
        """Return matrix dimensions."""
        return BackendDims(spatial=self._spatial_dims, time=int(self._data.shape[0]))  # type: ignore[arg-type]

    def get_mask(self) -> NDArray[np.bool_]:
        """Return a copy of the flat mask."""
        return self._mask.copy()

    @property
    def space(self) -> VolumeSpace:
        return self._space

    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        """Return masked matrix data with optional row/column selection.

        Every read is a fresh, writable array; the cached matrix is not shared.
        """
        data = self._data
        row_idx = _normalize_indices(rows, data.shape[0], "rows")
        col_idx = _normalize_indices(cols, data.shape[1], "cols")
        if row_idx is None and col_idx is None:
            return data.copy()
        if row_idx is not None:
            data = data[row_idx, :]
        if col_idx is not None:
            data = data[:, col_idx]
        return data

    def get_metadata(self) -> dict[str, object]:
        """Return matrix metadata."""
        return {"format": "matrix", **self._metadata}
