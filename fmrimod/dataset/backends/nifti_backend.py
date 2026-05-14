"""NIfTI storage backend using nibabel."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fmrimod.dataset.backend_protocol import BackendDims, StorageBackend
from fmrimod.dataset.errors import BackendIOError, ConfigError


def _as_paths(source: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(source, (str, Path)):
        paths = [Path(source)]
    else:
        paths = [Path(item) for item in source]
    if not paths:
        raise ConfigError("source must contain at least one path", parameter="source")
    return paths


def _normalize_indices(
    indices: NDArray[np.intp] | Sequence[int] | int | None,
    upper: int,
    name: str,
) -> NDArray[np.intp]:
    if indices is None:
        return np.arange(upper, dtype=np.intp)

    arr = np.atleast_1d(np.asarray(indices))
    if arr.dtype == np.bool_:
        if arr.size != upper:
            raise ValueError(
                f"{name} boolean mask length ({arr.size}) must equal {upper}"
            )
        arr = np.nonzero(arr)[0]
    elif not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} indices must be integers")

    out = arr.astype(np.intp, copy=False)
    if np.any(out < 0) or np.any(out >= upper):
        raise ValueError(f"{name} indices must be within [0, {upper - 1}]")
    return out


class NiftiBackend(StorageBackend):
    """Backend for NIfTI images in ``timepoints x masked voxels`` orientation."""

    def __init__(
        self,
        source: str | Path | Sequence[str | Path],
        mask_source: str | Path,
        *,
        preload: bool = False,
    ) -> None:
        self._source = _as_paths(source)
        self._mask_source = Path(mask_source)
        self._preload = bool(preload)
        self._data: NDArray[np.float64] | None = None
        self._mask_vec: NDArray[np.bool_] | None = None
        self._dims: BackendDims | None = None
        self._metadata: dict[str, object] = {}
        self._file_time_dims: list[int] = []
        self._is_open = False

    def open(self) -> None:  # noqa: A003
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ConfigError(
                "nibabel is required for NiftiBackend. "
                "Install with: pip install fmrimod[nibabel]",
                parameter="nibabel",
            ) from exc

        if not self._mask_source.exists():
            raise BackendIOError(
                f"Mask file not found: {self._mask_source}",
                file=str(self._mask_source),
                operation="open",
            )

        mask_img = nib.load(str(self._mask_source))
        mask_data = np.asarray(mask_img.dataobj)
        if mask_data.ndim < 3:
            raise BackendIOError(
                "mask image must be at least 3-D",
                file=str(self._mask_source),
                operation="open",
            )
        spatial_shape = tuple(int(v) for v in mask_data.shape[:3])
        mask_volume = mask_data[..., 0] if mask_data.ndim > 3 else mask_data
        self._mask_vec = mask_volume.reshape(-1) > 0

        total_time = 0
        self._file_time_dims = []
        for src in self._source:
            if not src.exists():
                raise BackendIOError(
                    f"Source file not found: {src}",
                    file=str(src),
                    operation="open",
                )
            img = nib.load(str(src))
            shape = tuple(int(v) for v in img.shape)
            if len(shape) < 3 or shape[:3] != spatial_shape:
                raise BackendIOError(
                    f"Source spatial shape {shape[:3]} does not match mask "
                    f"shape {spatial_shape}",
                    file=str(src),
                    operation="open",
                )
            n_time = shape[3] if len(shape) > 3 else 1
            self._file_time_dims.append(int(n_time))
            total_time += int(n_time)

        self._dims = BackendDims(spatial=spatial_shape, time=total_time)
        self._metadata = {
            "format": "nifti",
            "affine": np.asarray(mask_img.affine),
            "voxel_dims": np.asarray(mask_img.header.get_zooms()[:3]),
            "source": [str(path) for path in self._source],
            "mask_source": str(self._mask_source),
        }
        self._data = self._read_data_subset() if self._preload else None
        self._is_open = True

    def close(self) -> None:
        self._data = None
        self._is_open = False

    def get_dims(self) -> BackendDims:
        if self._dims is None:
            raise BackendIOError("Backend not opened", operation="get_dims")
        return self._dims

    def get_mask(self) -> NDArray[np.bool_]:
        if self._mask_vec is None:
            raise BackendIOError("Backend not opened", operation="get_mask")
        return self._mask_vec.copy()

    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        if self._data is None:
            return self._read_data_subset(rows=rows, cols=cols)

        data = self._data
        if rows is not None:
            row_idx = _normalize_indices(rows, self._dims.time, "rows")
            data = data[row_idx, :]
        if cols is not None:
            col_idx = _normalize_indices(cols, int(self._mask_vec.sum()), "cols")
            data = data[:, col_idx]
        return np.asarray(data, dtype=np.float64)

    def get_metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    def _read_data_subset(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ConfigError(
                "nibabel is required for NiftiBackend. "
                "Install with: pip install fmrimod[nibabel]",
                parameter="nibabel",
            ) from exc

        if self._mask_vec is None or self._dims is None:
            raise BackendIOError("Backend not opened", operation="read")

        row_idx = _normalize_indices(rows, self._dims.time, "rows")
        col_idx = _normalize_indices(cols, int(self._mask_vec.sum()), "cols")
        voxel_idx = np.flatnonzero(self._mask_vec)[col_idx]
        result = np.empty((len(row_idx), len(voxel_idx)), dtype=np.float64)

        cursor = 0
        for src, n_time in zip(self._source, self._file_time_dims):
            positions = np.where((row_idx >= cursor) & (row_idx < cursor + n_time))[0]
            local_rows = row_idx[positions] - cursor
            cursor += n_time
            if local_rows.size == 0:
                continue

            img = nib.load(str(src))
            source_rows: list[NDArray[np.float64]] = []
            for row in local_rows:
                if len(img.shape) > 3:
                    volume = np.asarray(img.dataobj[..., int(row)], dtype=np.float64)
                else:
                    volume = np.asarray(img.dataobj, dtype=np.float64)
                source_rows.append(volume.reshape(-1)[voxel_idx])
            result[positions, :] = np.vstack(source_rows)

        return result


def nifti_backend(
    source: str | Path | Sequence[str | Path],
    mask_source: str | Path,
    *,
    preload: bool = False,
) -> NiftiBackend:
    """Construct a NIfTI storage backend."""
    return NiftiBackend(source=source, mask_source=mask_source, preload=preload)


__all__ = ["NiftiBackend", "nifti_backend"]
