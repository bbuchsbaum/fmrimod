"""Progress events for native group-analysis operations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol


@dataclass(frozen=True)
class GroupProgressEvent:
    """One progress notification from a group-analysis operation."""

    stage: str
    method: str | None = None
    message: str = ""
    completed: int | None = None
    total: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class ProgressReporter(Protocol):
    """Callable progress sink used by long-running group operations."""

    def __call__(self, event: GroupProgressEvent) -> None:
        """Handle one progress event."""


ProgressCallback = Callable[[GroupProgressEvent], None]


def emit_progress(
    progress: ProgressCallback | ProgressReporter | None,
    stage: str,
    *,
    method: str | None = None,
    message: str = "",
    completed: int | None = None,
    total: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    **extra: Any,
) -> None:
    """Emit a progress event when a callback is configured."""
    if progress is None:
        return
    payload: dict[str, Any] = {}
    if metadata is not None:
        payload.update(dict(metadata))
    payload.update(extra)
    progress(
        GroupProgressEvent(
            stage=stage,
            method=method,
            message=message,
            completed=completed,
            total=total,
            metadata=payload,
        )
    )
