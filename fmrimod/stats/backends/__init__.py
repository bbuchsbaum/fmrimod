"""Backend registry for second-level interfaces."""

from __future__ import annotations

from .base import SecondLevelBackend
from .fmrigds_backend import FmrigdsBackend, fmrigds_backend_available
from .python_backend import PythonParityBackend


def available_second_level_backends() -> list[str]:
    """List known backend names."""
    return ["auto", "python", "fmrigds"]


def resolve_second_level_backend(name: str) -> SecondLevelBackend:
    """Resolve a backend name to an implementation instance."""
    if name in ("auto", "python"):
        return PythonParityBackend()
    if name == "fmrigds":
        return FmrigdsBackend()
    raise ValueError("backend must be one of: auto, python, fmrigds")


__all__ = [
    "SecondLevelBackend",
    "available_second_level_backends",
    "fmrigds_backend_available",
    "resolve_second_level_backend",
]
