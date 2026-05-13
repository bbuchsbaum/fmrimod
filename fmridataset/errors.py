"""Compatibility re-exports for dataset errors."""

from fmrimod.dataset.errors import BackendIOError, ConfigError, FmriDatasetError

__all__ = ["FmriDatasetError", "BackendIOError", "ConfigError"]
