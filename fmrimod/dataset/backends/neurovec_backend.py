"""Storage backend over one :class:`neuroim.NeuroVec` (one run)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from fmrimod.dataset.backend_protocol import (
    BackendDims,
    StorageBackend,
    _normalize_indices,
)
from fmrimod.dataset.errors import ConfigError
from fmrimod.dataset.space import VolumeSpace


def _require_neuroim(what: str) -> Any:
    try:
        import neuroim
    except ImportError as exc:  # pragma: no cover - import-time guard
        raise ConfigError(
            f"neuroim is required for {what}. Install with: pip install neuroim",
            parameter="neuroim",
        ) from exc
    return neuroim


def is_neurovec(obj: object) -> bool:
    """True iff ``obj`` is a ``neuroim.NeuroVec`` instance."""
    try:
        import neuroim
    except ImportError:
        return False
    return isinstance(obj, neuroim.NeuroVec)


def is_neurovol(obj: object) -> bool:
    """True iff ``obj`` is a ``neuroim.NeuroVol`` (or subclass) instance."""
    try:
        import neuroim
    except ImportError:
        return False
    return isinstance(obj, neuroim.NeuroVol)


def _array4d(vec: object) -> NDArray[Any]:
    if hasattr(vec, "data"):
        return np.asarray(cast(Any, vec).data)
    if hasattr(vec, "__array__"):
        return np.asarray(vec)
    return np.asarray(cast(Any, vec)[:])


def resolve_volume_mask(mask: object, vec: object) -> NDArray[np.bool_]:
    """3-D boolean mask for ``vec`` from a NeuroVol, ndarray or path.

    ``None`` keeps the voxels that are non-zero in the first volume. A
    NeuroVol mask must be in ``vec``'s spatial space (shape and affine); a
    plain array carries no geometry and is checked by shape only.
    """
    space = cast(Any, vec).space
    shape = tuple(int(d) for d in space.dim[:3])
    if isinstance(mask, (str, Path)):
        mask = _require_neuroim("reading a mask path").read_vol(str(mask))
    if mask is None:
        arr = np.asarray(_array4d(vec)[..., 0] != 0, dtype=bool)
    elif is_neurovol(mask):
        try:
            space.compatible_with(cast(Any, mask).space)
        except ValueError as exc:
            raise ValueError(f"Mask is not in the run's spatial space: {exc}") from exc
        arr = np.asarray(cast(Any, mask).data, dtype=bool)
    elif isinstance(mask, np.ndarray) or hasattr(mask, "data"):
        raw = mask if isinstance(mask, np.ndarray) else cast(Any, mask).data
        arr = np.asarray(raw, dtype=bool)
        if arr.ndim == 1 and arr.size == int(np.prod(shape)):
            arr = arr.reshape(shape)
    else:
        raise TypeError(f"Unsupported mask type: {type(mask).__name__}")
    if arr.shape != shape:
        raise ValueError(f"Mask shape {arr.shape} != spatial shape {shape}")
    return arr


class NeuroVecBackend(StorageBackend):
    """``time x in-mask voxels`` view of one 4-D :class:`neuroim.NeuroVec`.

    Reads index the voxels and volumes they need; the full matrix is never
    materialised unless asked for. Use a
    :class:`~fmrimod.dataset.backends.ConcatBackend` for several runs.
    """

    def __init__(self, vec: object, mask: object = None) -> None:
        _require_neuroim("NeuroVecBackend")
        if not is_neurovec(vec):
            raise TypeError(f"NeuroVecBackend expects a NeuroVec; got {type(vec).__name__}")
        space = cast(Any, vec).space
        if space.ndim != 4:
            raise ValueError(f"NeuroVec must wrap a 4-D NeuroSpace; got {space.ndim}-D")
        self._vec = vec
        mask_arr = resolve_volume_mask(mask, vec)
        self._space = VolumeSpace.from_neurospace(space, mask_arr)
        self._voxel_ids = self._space.voxel_ids
        self._n_time = int(space.dim[3])

    @property
    def vec(self) -> object:
        """The wrapped NeuroVec."""
        return self._vec

    def open(self) -> None:
        """No-op: the NeuroVec is already in memory or memory-mapped."""

    def close(self) -> None:
        """No-op."""

    def get_dims(self) -> BackendDims:
        return BackendDims(spatial=self._space.shape, time=self._n_time)

    def get_mask(self) -> NDArray[np.bool_]:
        return self._space.mask.copy()

    @property
    def space(self) -> VolumeSpace:
        return self._space

    @property
    def source_run_lengths(self) -> tuple[int, ...]:
        return (self._n_time,)

    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        row_idx = _normalize_indices(rows, self._n_time, "rows")
        col_idx = _normalize_indices(cols, self._voxel_ids.size, "cols")
        voxels = self._voxel_ids if col_idx is None else self._voxel_ids[col_idx]
        flat = _array4d(self._vec).reshape(-1, self._n_time)
        block = flat[voxels] if row_idx is None else flat[voxels][:, row_idx]
        return np.asarray(block.T, dtype=np.float64)

    def get_metadata(self) -> dict[str, object]:
        return {"format": "neurovec", "affine": self._space.affine}


__all__ = ["NeuroVecBackend", "is_neurovec", "is_neurovol", "resolve_volume_mask"]
