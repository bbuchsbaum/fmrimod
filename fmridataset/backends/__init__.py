"""Compatibility re-exports for storage backends."""

from fmrimod.dataset.backends import InMemoryLatentBackend, LatentBackend, MatrixBackend

__all__ = ["InMemoryLatentBackend", "LatentBackend", "MatrixBackend"]
