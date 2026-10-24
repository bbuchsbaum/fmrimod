"""NIfTI storage backend using nibabel."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from fmrimod.dataset.backend_protocol import (
    BackendDims,
    StorageBackend,
    _normalize_indices,
)
from fmrimod.dataset.errors import BackendIOError, ConfigError
from fmrimod.dataset.space import VolumeSpace


def _require_nibabel() -> None:
    if importlib.util.find_spec("nibabel") is None:
        raise ConfigError(
            "nibabel is required for NiftiBackend. "
            "Install with: pip install fmrimod[nibabel]",
            parameter="nibabel",
        )


def _is_image(obj: object) -> bool:
    return hasattr(obj, "dataobj") and hasattr(obj, "affine")


def _as_sources(source: object) -> list[Any]:
    """Paths stay paths; in-memory nibabel images are kept as they are."""
    items = [source] if isinstance(source, (str, Path)) or _is_image(source) else list(
        cast(Sequence[Any], source)
    )
    if not items:
        raise ConfigError("source must contain at least one path", parameter="source")
    return [item if _is_image(item) else Path(item) for item in items]


def _source_label(src: object) -> str:
    return str(src) if isinstance(src, Path) else f"<{type(src).__name__}>"


class NiftiBackend(StorageBackend):
    """Backend for NIfTI images in ``timepoints x masked voxels`` orientation.

    ``source`` is one or more paths or in-memory nibabel images, read lazily
    volume by volume. ``mask_source`` is a path, a nibabel image, or a 3-D
    array (which takes the first source's affine); image masks keep their
    voxels with values above zero. Every source must share the mask's grid
    and affine.
    """

    def __init__(
        self,
        source: str | Path | Sequence[str | Path] | Any,
        mask_source: str | Path | Any,
        *,
        preload: bool = False,
    ) -> None:
        self._source = _as_sources(source)
        self._mask_source: Any = (
            mask_source
            if _is_image(mask_source) or isinstance(mask_source, np.ndarray)
            else Path(mask_source)
        )
        self._preload = bool(preload)
        self._data: NDArray[np.float64] | None = None
        self._mask_vec: NDArray[np.bool_] | None = None
        self._dims: BackendDims | None = None
        self._metadata: dict[str, object] = {}
        self._file_time_dims: list[int] = []
        self._space: VolumeSpace | None = None
        self._is_open = False

    def open(self) -> None:  # noqa: A003
        _require_nibabel()

        mask_label = _source_label(self._mask_source)
        if isinstance(self._mask_source, np.ndarray):
            mask_data = np.asarray(self._mask_source, dtype=bool)
            mask_affine = np.asarray(self._image(self._source[0]).affine, dtype=np.float64)
            mask_label = "<ndarray mask>"
        else:
            if isinstance(self._mask_source, Path) and not self._mask_source.exists():
                raise BackendIOError(
                    f"Mask file not found: {self._mask_source}",
                    file=mask_label,
                    operation="open",
                )
            mask_img = self._image(self._mask_source)
            mask_data = np.asarray(mask_img.dataobj)
            mask_affine = np.asarray(mask_img.affine, dtype=np.float64)
        if mask_data.ndim < 3:
            raise BackendIOError(
                "mask image must be at least 3-D",
                file=mask_label,
                operation="open",
            )
        spatial_shape = tuple(int(v) for v in mask_data.shape[:3])
        mask_volume = mask_data[..., 0] if mask_data.ndim > 3 else mask_data
        self._mask_vec = mask_volume.reshape(-1) > 0

        total_time = 0
        self._file_time_dims = []
        for src in self._source:
            if isinstance(src, Path) and not src.exists():
                raise BackendIOError(
                    f"Source file not found: {src}",
                    file=str(src),
                    operation="open",
                )
            img = self._image(src)
            shape = tuple(int(v) for v in img.shape)
            if len(shape) < 3 or shape[:3] != spatial_shape:
                raise BackendIOError(
                    f"Source spatial shape {shape[:3]} does not match mask "
                    f"shape {spatial_shape}",
                    file=_source_label(src),
                    operation="open",
                )
            # Spatial identity is the space, not the shape (fmridataset 7cb121e).
            if not np.allclose(np.asarray(img.affine), mask_affine, atol=1e-6):
                raise BackendIOError(
                    f"Source is not in the mask's spatial space: affine "
                    f"{np.asarray(img.affine).tolist()} != {mask_affine.tolist()}",
                    file=_source_label(src),
                    operation="open",
                )
            n_time = shape[3] if len(shape) > 3 else 1
            self._file_time_dims.append(int(n_time))
            total_time += int(n_time)

        self._dims = BackendDims(spatial=cast("tuple[int, int, int]", spatial_shape), time=total_time)
        self._space = VolumeSpace(
            cast("tuple[int, int, int]", spatial_shape),
            self._mask_vec,
            affine=mask_affine,
            mask_shape=spatial_shape,
        )
        self._metadata = {
            "format": "nifti",
            "affine": mask_affine.copy(),
            "voxel_dims": np.sqrt((mask_affine[:3, :3] ** 2).sum(axis=0)),
            "source": [_source_label(src) for src in self._source],
            "mask_source": mask_label,
        }
        self._data = self._read_data_subset() if self._preload else None
        self._is_open = True

    def close(self) -> None:
        self._data = None
        self._is_open = False

    @staticmethod
    def _image(src: Any) -> Any:
        if _is_image(src):
            return src
        import nibabel as nib

        return cast(Any, nib).load(str(src))

    def get_dims(self) -> BackendDims:
        if self._dims is None:
            raise BackendIOError("Backend not opened", operation="get_dims")
        return self._dims

    def get_mask(self) -> NDArray[np.bool_]:
        if self._mask_vec is None:
            raise BackendIOError("Backend not opened", operation="get_mask")
        return self._mask_vec.copy()

    @property
    def space(self) -> VolumeSpace:
        if self._space is None:
            raise BackendIOError("Backend not opened", operation="space")
        return self._space

    @property
    def source_run_lengths(self) -> tuple[int, ...]:
        if self._dims is None:
            raise BackendIOError("Backend not opened", operation="source_run_lengths")
        return tuple(self._file_time_dims)

    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        if self._data is None:
            return self._read_data_subset(rows=rows, cols=cols)

        data = self._data
        assert self._dims is not None and self._mask_vec is not None
        row_idx = _normalize_indices(rows, self._dims.time, "rows")
        col_idx = _normalize_indices(cols, int(self._mask_vec.sum()), "cols")
        if row_idx is None and col_idx is None:
            # Never hand out the preload cache: callers may write into reads.
            return data.copy()
        if row_idx is not None:
            data = data[row_idx, :]
        if col_idx is not None:
            data = data[:, col_idx]
        return np.asarray(data, dtype=np.float64)

    def get_metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    def _read_data_subset(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        _require_nibabel()

        if self._mask_vec is None or self._dims is None:
            raise BackendIOError("Backend not opened", operation="read")

        n_cols = int(self._mask_vec.sum())
        row_idx = _normalize_indices(rows, self._dims.time, "rows")
        if row_idx is None:
            row_idx = np.arange(self._dims.time, dtype=np.intp)
        col_idx = _normalize_indices(cols, n_cols, "cols")
        if col_idx is None:
            col_idx = np.arange(n_cols, dtype=np.intp)
        voxel_idx = np.flatnonzero(self._mask_vec)[col_idx]
        result = np.empty((len(row_idx), len(voxel_idx)), dtype=np.float64)

        cursor = 0
        for src, n_time in zip(self._source, self._file_time_dims):
            positions = np.where((row_idx >= cursor) & (row_idx < cursor + n_time))[0]
            local_rows = row_idx[positions] - cursor
            cursor += n_time
            if local_rows.size == 0:
                continue

            img = self._image(src)
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
