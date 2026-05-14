"""Compatibility re-export for the latent storage backend."""

from fmrimod.dataset.backends.latent_backend import InMemoryLatentBackend, LatentBackend

__all__ = ["InMemoryLatentBackend", "LatentBackend"]
