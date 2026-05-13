"""Exception hierarchy for native group-level analysis."""

from __future__ import annotations


class GroupAnalysisError(Exception):
    """Base exception for group-analysis errors."""


class GroupConfigError(GroupAnalysisError):
    """Raised when invalid group-analysis configuration is provided."""

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


class GroupRegistryError(GroupConfigError):
    """Raised when a group-analysis registry operation is invalid."""


class GroupSpaceError(GroupAnalysisError):
    """Raised when a group space descriptor is invalid or incompatible."""


class AdapterContractError(GroupAnalysisError):
    """Raised when a group data adapter violates the expected contract."""


class GroupSchemaError(GroupAnalysisError):
    """Raised when a serialized group-analysis schema is unsupported."""


class GroupReducerError(GroupAnalysisError):
    """Raised when a reducer cannot execute or returns invalid output."""


class UnsupportedGroupFeatureError(NotImplementedError, GroupAnalysisError):
    """Raised for explicit, documented gaps in the native group-analysis port."""

