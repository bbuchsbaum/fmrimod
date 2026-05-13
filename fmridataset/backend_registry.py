"""Compatibility re-exports for the backend registry."""

from fmrimod.dataset.backend_registry import (
    BackendRegistry,
    create_backend,
    get_backend_registry,
    is_backend_registered,
    list_backend_names,
    register_backend,
    unregister_backend,
)

__all__ = [
    "BackendRegistry",
    "get_backend_registry",
    "register_backend",
    "create_backend",
    "is_backend_registered",
    "list_backend_names",
    "unregister_backend",
]
