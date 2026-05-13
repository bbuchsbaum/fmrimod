"""Exception hierarchy for canonical dataset IO."""

from __future__ import annotations


class FmriDatasetError(Exception):
    """Base exception for dataset errors."""


class BackendIOError(FmriDatasetError):
    """Raised when a storage backend encounters an IO failure."""

    def __init__(
        self,
        message: str,
        *,
        file: str | None = None,
        operation: str | None = None,
    ) -> None:
        self.file = file
        self.operation = operation
        super().__init__(message)


class ConfigError(FmriDatasetError):
    """Raised when invalid dataset configuration is provided."""

    def __init__(
        self,
        message: str,
        *,
        parameter: str | None = None,
        value: object = None,
    ) -> None:
        self.parameter = parameter
        self.value = value
        super().__init__(message)
