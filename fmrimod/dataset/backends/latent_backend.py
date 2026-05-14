"""Latent-space storage backend."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backend_protocol import BackendDims, StorageBackend
from ..errors import BackendIOError, ConfigError


def _validate_indices(
    indices: NDArray[np.intp] | Sequence[int] | int | None,
    upper: int,
    name: str,
) -> NDArray[np.intp] | None:
    if indices is None:
        return None
    arr = np.atleast_1d(np.asarray(indices))
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} indices must be integers")
    result = arr.astype(np.intp, copy=False)
    if np.any(result < 0) or np.any(result >= upper):
        raise ValueError(f"{name} indices must be within [0, {upper - 1}]")
    return result


class LatentBackend(StorageBackend):
    """Backend for HDF5 latent decompositions.

    Each source file must contain ``basis`` and one file must contain
    ``loadings``. Voxel-space data are reconstructed as
    ``basis @ loadings.T + offset``.
    """

    def __init__(
        self,
        source: str | Path | Sequence[str | Path],
        *,
        preload: bool = False,
    ) -> None:
        if isinstance(source, (str, Path)):
            sources = [Path(source)]
        else:
            sources = [Path(item) for item in source]
        if not sources:
            raise ConfigError(
                "source must contain at least one path",
                parameter="source",
            )

        self._source = sources
        self._preload = bool(preload)
        self._basis_parts: list[NDArray[np.float64]] = []
        self._loadings: NDArray[np.float64] | None = None
        self._offset: NDArray[np.float64] | None = None
        self._data: NDArray[np.float64] | None = None
        self._dims: BackendDims | None = None
        self._run_lengths: list[int] = []
        self._is_open = False

    def open(self) -> None:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ConfigError(
                "h5py is required for LatentBackend. Install with: pip install h5py"
            ) from exc

        self._basis_parts = []
        self._loadings = None
        self._offset = None
        self._data = None
        self._run_lengths = []
        total_time = 0

        for src in self._source:
            if not src.exists():
                raise BackendIOError(
                    f"Source file not found: {src}",
                    file=str(src),
                    operation="open",
                )
            with h5py.File(str(src), "r") as handle:
                if "basis" not in handle:
                    raise BackendIOError(
                        f"'basis' dataset not found in {src}",
                        file=str(src),
                        operation="open",
                    )
                basis = np.asarray(handle["basis"], dtype=np.float64)
                if basis.ndim != 2:
                    raise BackendIOError(
                        "'basis' must be a 2-D dataset",
                        file=str(src),
                        operation="open",
                    )
                self._basis_parts.append(basis)
                self._run_lengths.append(int(basis.shape[0]))
                total_time += int(basis.shape[0])

                if self._loadings is None and "loadings" in handle:
                    self._loadings = np.asarray(handle["loadings"], dtype=np.float64)
                if self._offset is None and "offset" in handle:
                    self._offset = np.asarray(handle["offset"], dtype=np.float64)

        if self._loadings is None:
            raise BackendIOError(
                "No 'loadings' dataset found in any source file",
                operation="open",
            )
        if self._loadings.ndim != 2:
            raise BackendIOError("'loadings' must be a 2-D dataset", operation="open")

        n_components = int(self._loadings.shape[1])
        for basis in self._basis_parts:
            if basis.shape[1] != n_components:
                raise BackendIOError(
                    "basis columns must match loadings components",
                    operation="open",
                )

        n_voxels = int(self._loadings.shape[0])
        if self._offset is not None and self._offset.shape != (n_voxels,):
            raise BackendIOError(
                "offset length must match loadings rows",
                operation="open",
            )

        self._dims = BackendDims(spatial=(n_voxels, 1, 1), time=total_time)
        self._is_open = True
        if self._preload:
            self._data = self.reconstruct_voxels()

    def close(self) -> None:
        self._data = None
        self._is_open = False

    def _require_open(self, operation: str) -> None:
        if not self._is_open:
            raise BackendIOError("Backend not opened", operation=operation)

    @property
    def run_lengths(self) -> list[int]:
        self._require_open("run_lengths")
        return list(self._run_lengths)

    def get_dims(self) -> BackendDims:
        self._require_open("get_dims")
        if self._dims is None:
            raise BackendIOError("Backend not opened", operation="get_dims")
        return self._dims

    def get_mask(self) -> NDArray[np.bool_]:
        self._require_open("get_mask")
        return np.ones(self.get_dims().spatial[0], dtype=np.bool_)

    def _basis(self) -> NDArray[np.float64]:
        return np.concatenate(self._basis_parts, axis=0)

    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        """Return latent scores in ``timepoints x components`` orientation."""
        self._require_open("get_data")
        basis = self._basis()
        row_idx = _validate_indices(rows, basis.shape[0], "rows")
        col_idx = _validate_indices(cols, basis.shape[1], "cols")
        data = basis
        if row_idx is not None:
            data = data[row_idx, :]
        if col_idx is not None:
            data = data[:, col_idx]
        return np.asarray(data, dtype=np.float64)

    def reconstruct_voxels(
        self,
        rows: NDArray[np.intp] | None = None,
        voxels: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        """Return reconstructed voxel data in ``timepoints x voxels`` orientation."""
        self._require_open("reconstruct_voxels")
        dims = self.get_dims()
        row_idx = _validate_indices(rows, dims.time, "rows")
        voxel_idx = _validate_indices(voxels, dims.spatial[0], "voxels")

        if self._data is not None:
            data = self._data
            if row_idx is not None:
                data = data[row_idx, :]
            if voxel_idx is not None:
                data = data[:, voxel_idx]
            return np.asarray(data, dtype=np.float64)

        basis = self._basis() if row_idx is None else self._basis()[row_idx, :]
        assert self._loadings is not None
        loadings = self._loadings if voxel_idx is None else self._loadings[voxel_idx, :]
        data = basis @ loadings.T
        if self._offset is not None:
            offset = self._offset if voxel_idx is None else self._offset[voxel_idx]
            data = data + offset[np.newaxis, :]
        return np.asarray(data, dtype=np.float64)

    def get_loadings(
        self,
        components: NDArray[np.intp] | Sequence[int] | int | None = None,
    ) -> NDArray[np.float64]:
        """Return spatial loadings in ``voxels x components`` orientation."""
        self._require_open("get_loadings")
        assert self._loadings is not None
        comp_idx = _validate_indices(components, self._loadings.shape[1], "components")
        if comp_idx is None:
            return self._loadings.copy()
        return self._loadings[:, comp_idx]

    def get_metadata(self) -> dict[str, Any]:
        self._require_open("get_metadata")
        dims = self.get_dims()
        assert self._loadings is not None
        loadings_norm = np.sqrt(np.sum(self._loadings**2, axis=0))
        basis = self._basis()
        basis_variance = (
            np.var(basis, axis=0, ddof=1)
            if basis.shape[0] > 1
            else np.zeros(
                basis.shape[1],
                dtype=np.float64,
            )
        )
        return {
            "format": "latent_h5",
            "storage_format": "latent",
            "n_components": int(self._loadings.shape[1]),
            "n_voxels": int(dims.spatial[0]),
            "n_runs": len(self._run_lengths),
            "has_offset": self._offset is not None,
            "basis_variance": basis_variance,
            "loadings_norm": loadings_norm,
            "loadings_sparsity": 0.0,
        }


class InMemoryLatentBackend(StorageBackend):
    """In-memory latent backend used by the array constructor."""

    def __init__(
        self,
        scores: NDArray[np.float64],
        *,
        loadings: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
        run_lengths: Sequence[int] | None = None,
    ) -> None:
        self._scores = np.asarray(scores, dtype=np.float64)
        if self._scores.ndim != 2:
            raise ConfigError("scores must be a 2-D matrix", parameter="scores")
        self._loadings = (
            None if loadings is None else np.asarray(loadings, dtype=np.float64)
        )
        if self._loadings is not None:
            if self._loadings.ndim != 2:
                raise ConfigError("loadings must be a 2-D matrix", parameter="loadings")
            if self._loadings.shape[1] != self._scores.shape[1]:
                raise ConfigError(
                    "loadings columns must match score columns",
                    parameter="loadings",
                )
        self._offset = None if offset is None else np.asarray(offset, dtype=np.float64)
        n_voxels = (
            self._scores.shape[1]
            if self._loadings is None
            else self._loadings.shape[0]
        )
        if self._offset is not None and self._offset.shape != (n_voxels,):
            raise ConfigError(
                "offset length must match voxel count",
                parameter="offset",
            )
        self._run_lengths = (
            [int(self._scores.shape[0])]
            if run_lengths is None
            else [int(v) for v in run_lengths]
        )
        if sum(self._run_lengths) != self._scores.shape[0]:
            raise ConfigError(
                "run_lengths must sum to score rows",
                parameter="run_lengths",
            )

    def open(self) -> None:
        """No-op for in-memory latent data."""

    def close(self) -> None:
        """No-op for in-memory latent data."""

    @property
    def run_lengths(self) -> list[int]:
        return list(self._run_lengths)

    def get_dims(self) -> BackendDims:
        n_voxels = (
            self._scores.shape[1]
            if self._loadings is None
            else self._loadings.shape[0]
        )
        return BackendDims(
            spatial=(int(n_voxels), 1, 1),
            time=int(self._scores.shape[0]),
        )

    def get_mask(self) -> NDArray[np.bool_]:
        return np.ones(self.get_dims().spatial[0], dtype=np.bool_)

    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        row_idx = _validate_indices(rows, self._scores.shape[0], "rows")
        col_idx = _validate_indices(cols, self._scores.shape[1], "cols")
        data = self._scores
        if row_idx is not None:
            data = data[row_idx, :]
        if col_idx is not None:
            data = data[:, col_idx]
        return np.asarray(data, dtype=np.float64)

    def reconstruct_voxels(
        self,
        rows: NDArray[np.intp] | None = None,
        voxels: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        dims = self.get_dims()
        row_idx = _validate_indices(rows, dims.time, "rows")
        voxel_idx = _validate_indices(voxels, dims.spatial[0], "voxels")
        scores = self._scores if row_idx is None else self._scores[row_idx, :]
        if self._loadings is None:
            data = scores
            if voxel_idx is not None:
                data = data[:, voxel_idx]
            return np.asarray(data, dtype=np.float64)
        loadings = self._loadings if voxel_idx is None else self._loadings[voxel_idx, :]
        data = scores @ loadings.T
        if self._offset is not None:
            offset = self._offset if voxel_idx is None else self._offset[voxel_idx]
            data = data + offset[np.newaxis, :]
        return np.asarray(data, dtype=np.float64)

    def get_loadings(
        self,
        components: NDArray[np.intp] | Sequence[int] | int | None = None,
    ) -> NDArray[np.float64]:
        if self._loadings is None:
            return np.eye(self._scores.shape[1], dtype=np.float64)
        comp_idx = _validate_indices(components, self._loadings.shape[1], "components")
        if comp_idx is None:
            return self._loadings.copy()
        return self._loadings[:, comp_idx]

    def get_metadata(self) -> dict[str, Any]:
        loadings = self.get_loadings()
        basis_variance = (
            np.var(self._scores, axis=0, ddof=1)
            if self._scores.shape[0] > 1
            else np.zeros(self._scores.shape[1], dtype=np.float64)
        )
        return {
            "format": "latent_memory",
            "storage_format": "latent",
            "n_components": int(self._scores.shape[1]),
            "n_voxels": int(self.get_dims().spatial[0]),
            "n_runs": len(self._run_lengths),
            "has_offset": self._offset is not None,
            "basis_variance": basis_variance,
            "loadings_norm": np.sqrt(np.sum(loadings**2, axis=0)),
            "loadings_sparsity": 0.0,
        }


__all__ = ["InMemoryLatentBackend", "LatentBackend"]
