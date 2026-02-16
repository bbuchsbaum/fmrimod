"""Pytest configuration for repository-wide test hooks."""

from __future__ import annotations

import warnings


warnings.filterwarnings(
    "ignore",
    message="scipy.misc is deprecated and will be removed in 2.0.0",
    category=DeprecationWarning,
)


def pytest_sessionstart(session) -> None:  # pragma: no cover - test runner hook
    """Register warning filters that are noisy for external dependencies."""
    warnings.filterwarnings(
        "ignore",
        message="scipy.misc is deprecated and will be removed in 2.0.0",
        category=DeprecationWarning,
    )
