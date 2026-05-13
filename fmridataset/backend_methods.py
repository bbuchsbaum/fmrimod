"""Compatibility re-exports for backend method facades."""

from fmrimod.dataset.backend_methods import (
    backend_close,
    backend_get_data,
    backend_get_dims,
    backend_get_loadings,
    backend_get_mask,
    backend_get_metadata,
    backend_open,
    backend_reconstruct_voxels,
    validate_backend,
)

__all__ = [
    "backend_open",
    "backend_close",
    "backend_get_dims",
    "backend_get_mask",
    "backend_get_data",
    "backend_get_metadata",
    "backend_get_loadings",
    "backend_reconstruct_voxels",
    "validate_backend",
]
