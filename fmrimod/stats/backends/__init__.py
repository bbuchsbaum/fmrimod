"""Backend registry for second-level interfaces."""

from __future__ import annotations

from .base import SecondLevelBackend
from .fmrigds_backend import FmrigdsBackend, fmrigds_backend_available
from .python_backend import PythonParityBackend


def available_second_level_backends(*, include_oracles: bool = False) -> list[str]:
    """List supported second-level backend names.

    The default production surface is native-only. Pass
    ``include_oracles=True`` to include explicit non-production oracle or
    migration fallback backends.
    """
    backends = ["auto", "python"]
    if include_oracles:
        backends.append("fmrigds-r")
    return backends


def resolve_second_level_backend(name: str) -> SecondLevelBackend:
    """Resolve a backend name to an implementation instance."""
    if name in ("auto", "python"):
        return PythonParityBackend()
    if name in ("fmrigds-r", "r-fmrigds", "r"):
        return FmrigdsBackend()
    if name == "fmrigds":
        raise ValueError(
            "backend='fmrigds' is retired; use backend='fmrigds-r' for the R oracle"
        )
    raise ValueError("backend must be one of: auto, python, fmrigds-r")


__all__ = [
    "SecondLevelBackend",
    "available_second_level_backends",
    "fmrigds_backend_available",
    "resolve_second_level_backend",
]
