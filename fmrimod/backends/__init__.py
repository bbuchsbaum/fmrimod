"""Solver backends (numpy default, optional JAX).

The JAX backend is lazily imported to avoid requiring JAX at
import time.  Access it via ``from fmrimod.backends.jax_backend import JaxBackend``.
"""

from .numpy_backend import NumpyBackend

__all__ = ["NumpyBackend"]
