"""Backend protocol for second-level modeling."""

from __future__ import annotations

from typing import Protocol

from ..interfaces import GroupFitRequest, GroupFitResult


class SecondLevelBackend(Protocol):
    """Backend interface for second-level model fitting."""

    name: str

    def fit(self, request: GroupFitRequest) -> GroupFitResult:
        """Fit a second-level request and return canonical outputs."""
        ...
