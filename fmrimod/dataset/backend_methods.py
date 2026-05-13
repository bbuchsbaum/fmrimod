"""Functional facade over storage backend methods."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .backend_protocol import BackendDims, StorageBackend


def backend_open(backend: StorageBackend) -> None:
    """Open a storage backend."""
    backend.open()


def backend_close(backend: StorageBackend) -> None:
    """Close a storage backend."""
    backend.close()


def backend_get_dims(backend: StorageBackend) -> BackendDims:
    """Return backend dimensions."""
    return backend.get_dims()


def backend_get_mask(backend: StorageBackend) -> NDArray[np.bool_]:
    """Return backend mask."""
    return backend.get_mask()


def backend_get_data(
    backend: StorageBackend,
    rows: NDArray[np.intp] | None = None,
    cols: NDArray[np.intp] | None = None,
) -> NDArray[np.floating[Any]]:
    """Return backend data."""
    return backend.get_data(rows=rows, cols=cols)


def backend_get_metadata(backend: StorageBackend) -> dict[str, Any]:
    """Return backend metadata."""
    return backend.get_metadata()


def backend_get_loadings(backend: StorageBackend) -> Any:
    """Return backend loadings when implemented."""
    method = getattr(backend, "get_loadings", None)
    if method is None:
        raise NotImplementedError("backend does not expose get_loadings()")
    return method()


def backend_reconstruct_voxels(
    backend: StorageBackend,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Reconstruct voxel data when implemented by the backend."""
    method = getattr(backend, "reconstruct_voxels", None)
    if method is None:
        raise NotImplementedError("backend does not expose reconstruct_voxels()")
    return method(*args, **kwargs)


def validate_backend(backend: StorageBackend) -> bool:
    """Validate a backend."""
    return backend.validate()
