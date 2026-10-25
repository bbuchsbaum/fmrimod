"""Storage backend protocol for canonical fMRI datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .errors import ConfigError
from .space import VolumeSpace


@dataclass(frozen=True)
class BackendDims:
    """Dimensions reported by a storage backend."""

    spatial: tuple[int, int, int]
    time: int

    def __post_init__(self) -> None:
        if len(self.spatial) != 3:
            raise ConfigError(
                "spatial must have exactly 3 elements",
                parameter="spatial",
                value=self.spatial,
            )
        if any(s < 1 for s in self.spatial):
            raise ConfigError(
                "all spatial dimensions must be >= 1",
                parameter="spatial",
                value=self.spatial,
            )
        if self.time < 1:
            raise ConfigError(
                "time dimension must be >= 1",
                parameter="time",
                value=self.time,
            )

    @property
    def n_spatial(self) -> int:
        """Total number of spatial elements."""
        return int(np.prod(self.spatial))


def _normalize_indices(
    indices: NDArray[Any] | Sequence[int] | int | None,
    upper: int,
    name: str,
) -> NDArray[np.intp] | None:
    """Resolve a ``rows=``/``cols=`` selector to non-negative positions.

    A boolean selector is a mask and must match the axis length; numeric
    positions must be whole. Python's negative-index and repeat semantics are
    kept (R ``ec6efd3`` selection law, minus R's negative-exclusion rule).
    ``None`` means "the whole axis" and is returned unchanged.
    """
    if indices is None:
        return None
    arr = np.atleast_1d(np.asarray(indices))
    if arr.dtype == np.bool_:
        if arr.shape != (upper,):
            raise ValueError(
                f"{name} boolean mask length ({arr.size}) must equal {upper}"
            )
        return cast("NDArray[np.intp]", np.flatnonzero(arr))
    if arr.size == 0:
        return np.empty(0, dtype=np.intp)
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)) or np.any(arr != np.round(arr)):
            raise ValueError(f"{name} indices must be integers")
    elif not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} indices must be integers")
    out = arr.reshape(-1).astype(np.intp)
    if np.any(out < -upper) or np.any(out >= upper):
        raise ValueError(f"{name} indices must be within [{-upper}, {upper - 1}]")
    return cast("NDArray[np.intp]", np.where(out < 0, out + upper, out))


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    Data orientation is always ``timepoints x voxels``.
    """

    @abstractmethod
    def open(self) -> None:
        """Acquire resources."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""

    def __enter__(self) -> StorageBackend:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @abstractmethod
    def get_dims(self) -> BackendDims:
        """Return backend dimensions."""

    @abstractmethod
    def get_mask(self) -> NDArray[np.bool_]:
        """Return a flat boolean mask with length ``prod(spatial)``."""

    @abstractmethod
    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        """Read data in ``timepoints x voxels`` orientation."""

    @abstractmethod
    def get_metadata(self) -> dict[str, object]:
        """Return backend metadata."""

    @property
    def space(self) -> VolumeSpace:
        """Spatial identity of the columns.

        The default covers sources without geometry: the reported grid and
        mask, no affine, and a flat mask. Image backends override it.
        """
        dims = self.get_dims()
        return VolumeSpace(dims.spatial, self.get_mask(), affine=None)

    @property
    def source_run_lengths(self) -> tuple[int, ...] | None:
        """Rows contributed by each underlying file or source, when known.

        ``None`` means the backend has no natural run boundaries.
        """
        return None

    def validate(self) -> bool:
        """Check backend invariants and raise ``ConfigError`` on failure."""
        dims = self.get_dims()
        if not isinstance(dims, BackendDims):
            raise ConfigError("get_dims() must return a BackendDims instance")

        mask = self.get_mask()
        if mask.dtype != np.bool_:
            raise ConfigError("get_mask() must return a boolean array")

        expected = int(np.prod(dims.spatial))
        if mask.shape != (expected,):
            raise ConfigError(
                f"mask length ({mask.size}) must equal prod(spatial dims) ({expected})"
            )
        if np.any(np.isnan(mask.astype(float))):
            raise ConfigError("mask must not contain NaN values")
        if mask.sum() == 0:
            raise ConfigError("mask must contain at least one True value")
        return True
