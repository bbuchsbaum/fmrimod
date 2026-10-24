"""Pickle support for objects that expose read-only ``MappingProxyType`` fields.

``MappingProxyType`` cannot be pickled, so a frozen dataclass that stores one
(to hand out a read-only view) cannot travel to a worker process or a cache.
:class:`ReadOnlyMappingPickle` swaps each proxy for a plain dict when pickling
and rebuilds the proxy when unpickling; everything else pickles as usual.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any


class _ProxiedDict:
    """Marker for a mapping that was a ``MappingProxyType`` before pickling."""

    __slots__ = ("data",)

    def __init__(self, data: dict[Any, Any]) -> None:
        self.data = data

    def __getstate__(self) -> dict[Any, Any]:
        return self.data

    def __setstate__(self, data: dict[Any, Any]) -> None:
        self.data = data


class ReadOnlyMappingPickle:
    """Mixin: make ``MappingProxyType`` attributes survive pickling."""

    def __getstate__(self) -> dict[str, Any]:
        return {
            key: _ProxiedDict(dict(value))
            if isinstance(value, MappingProxyType)
            else value
            for key, value in vars(self).items()
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        for key, value in state.items():
            if isinstance(value, _ProxiedDict):
                value = MappingProxyType(value.data)
            # ``object.__setattr__`` so frozen dataclasses can be restored.
            object.__setattr__(self, key, value)
