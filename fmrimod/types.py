"""Type definitions and protocols for fmrimod."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

# Type aliases
Array = NDArray[np.float64]
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]
DataFrame = pd.DataFrame
Series = pd.Series
Categorical = pd.Categorical

# Time-related types
TimeType = Union[float, int, np.number]
OnsetType = Union[Sequence[TimeType], Array, Series]
DurationType = Union[TimeType, Sequence[TimeType], Array, Series]
AmplitudeType = Union[float, Sequence[float], Array, Series]

# Event types
EventType = Literal["categorical", "continuous", "basis", "matrix"]
ContrastType = Literal["simple", "pair", "polynomial", "interaction", "F"]

# Generic type variables
T = TypeVar("T")
EventT = TypeVar("EventT", bound="EventProtocol")
BasisT = TypeVar("BasisT", bound="BasisProtocol")
ModelT = TypeVar("ModelT", bound="ModelProtocol")


# Protocols
@runtime_checkable
class EventProtocol(Protocol):
    """Protocol for event objects."""
    
    @property
    def name(self) -> str:
        """Name of the event."""
        ...
    
    @property
    def event_type(self) -> EventType:
        """Type of event (categorical, continuous, etc.)."""
        ...
    
    @property
    def onsets(self) -> Array:
        """Event onset times."""
        ...
    
    @property
    def durations(self) -> Array:
        """Event durations."""
        ...
    
    @property
    def values(self) -> Union[Array, Categorical]:
        """Event values (categories or continuous values)."""
        ...
    
    def design_matrix(self, sampling_points: Array) -> Array:
        """Generate design matrix columns for this event."""
        ...


@runtime_checkable
class BasisProtocol(Protocol):
    """Protocol for basis function objects."""
    
    @property
    def name(self) -> str:
        """Name of the basis."""
        ...
    
    @property
    def nbasis(self) -> int:
        """Number of basis functions."""
        ...
    
    def evaluate(self, x: ArrayLike) -> Array:
        """Evaluate basis functions at given points."""
        ...
    
    def predict(self, x: ArrayLike, coef: Optional[ArrayLike] = None) -> Array:
        """Predict using basis functions."""
        ...


@runtime_checkable
class HRFProtocol(Protocol):
    """Protocol for HRF objects (from fmrimod)."""
    
    @property
    def name(self) -> str:
        """Name of the HRF."""
        ...
    
    @property
    def nbasis(self) -> int:
        """Number of basis functions in HRF."""
        ...
    
    def evaluate(self, t: ArrayLike) -> Array:
        """Evaluate HRF at time points."""
        ...


@runtime_checkable
class ContrastProtocol(Protocol):
    """Protocol for contrast objects."""
    
    @property
    def name(self) -> str:
        """Name of the contrast."""
        ...
    
    @property
    def contrast_type(self) -> ContrastType:
        """Type of contrast."""
        ...
    
    def compute_weights(self, columns: List[str]) -> Array:
        """Compute contrast weights for given columns."""
        ...


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for model objects."""
    
    def design_matrix(self) -> Array:
        """Get the design matrix."""
        ...
    
    def column_names(self) -> List[str]:
        """Get column names of design matrix."""
        ...
    
    def add_contrast(self, contrast: ContrastProtocol) -> None:
        """Add a contrast to the model."""
        ...


@runtime_checkable
class DesignMatrixProtocol(Protocol):
    """Protocol for objects that can produce design matrices."""
    
    def design_matrix(self) -> Array:
        """Get the design matrix."""
        ...


# Formula-related types
FormulaSpec = Union[str, Dict[str, Any], List[Dict[str, Any]]]


class FormulaContext:
    """Context for formula evaluation."""
    
    def __init__(self, data: Optional[DataFrame] = None, 
                 env: Optional[Dict[str, Any]] = None):
        """Initialize formula context.
        
        Parameters
        ----------
        data : DataFrame, optional
            Data frame containing variables
        env : dict, optional
            Additional environment variables
        """
        self.data = data
        self.env = env or {}
    
    def get(self, name: str) -> Any:
        """Get variable from context."""
        if self.data is not None and name in self.data.columns:
            return self.data[name]
        if name in self.env:
            return self.env[name]
        raise KeyError(f"Variable '{name}' not found in context")


# Sampling-related types  
# Note: SamplingFrame should be imported from fmrimod or fmrimod.sampling
# This is just a type alias for type hints
SamplingInfo = Any  # Type alias for SamplingFrame


# Validation types
class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_onsets(onsets: Any) -> Array:
    """Validate and convert onsets to array.
    
    Parameters
    ----------
    onsets : array-like
        Event onset times
    
    Returns
    -------
    Array
        Validated onset array
    
    Raises
    ------
    ValidationError
        If onsets are invalid
    """
    try:
        onsets = np.asarray(onsets, dtype=np.float64)
    except (ValueError, TypeError) as err:
        raise ValidationError(f"Cannot convert onsets to array: {err}") from err
    
    if onsets.ndim != 1:
        raise ValidationError(f"Onsets must be 1-dimensional, got {onsets.ndim}D")
    
    if len(onsets) == 0:
        raise ValidationError("Onsets cannot be empty")
    
    if np.any(onsets < 0):
        raise ValidationError("Onsets cannot be negative")
    
    if not np.all(np.isfinite(onsets)):
        raise ValidationError("Onsets must be finite")
    
    return onsets


def validate_durations(durations: Any, n_events: int) -> Array:
    """Validate and convert durations to array.
    
    Parameters
    ----------
    durations : scalar or array-like
        Event durations
    n_events : int
        Number of events (for broadcasting)
    
    Returns
    -------
    Array
        Validated duration array
    
    Raises
    ------
    ValidationError
        If durations are invalid
    """
    # Handle scalar durations
    if np.isscalar(durations):
        durations = np.full(n_events, durations, dtype=np.float64)
    else:
        try:
            durations = np.asarray(durations, dtype=np.float64)
        except (ValueError, TypeError) as err:
            raise ValidationError(f"Cannot convert durations to array: {err}") from err
    
    if durations.ndim != 1:
        raise ValidationError(f"Durations must be 1-dimensional, got {durations.ndim}D")
    
    if len(durations) != n_events:
        raise ValidationError(
            f"Duration length ({len(durations)}) must match number of events ({n_events})"
        )
    
    if np.any(durations < 0):
        raise ValidationError("Durations cannot be negative")
    
    if not np.all(np.isfinite(durations)):
        raise ValidationError("Durations must be finite")
    
    return durations
