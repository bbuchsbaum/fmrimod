"""Storage backend protocol for canonical fMRI datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import ConfigError


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
    ) -> NDArray[np.floating[Any]]:
        """Read data in ``timepoints x voxels`` orientation."""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return backend metadata."""

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
