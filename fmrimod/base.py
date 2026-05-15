"""Base classes for fmrimod."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .types import (
    Array,
    ContrastProtocol,
    EventType,
)


class BaseEvent(ABC):
    """Abstract base class for events."""
    
    # Subclasses must define:
    # - name: str
    # - onsets: Array
    # - durations: Array
    
    def __post_init__(self):
        """Validate inputs after initialization."""
        self._validate()
    
    @abstractmethod
    def _validate(self) -> None:
        """Validate event data."""
        pass
    
    @property
    @abstractmethod
    def event_type(self) -> EventType:
        """Type of event."""
        pass
    
    @abstractmethod
    def design_matrix(self, sampling_points: Array) -> Array:
        """Generate design matrix columns for this event."""
        pass
    
    @property
    def n_events(self) -> int:
        """Number of events."""
        return len(self.onsets)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"n_events={self.n_events}, event_type='{self.event_type}')"
        )


@dataclass
class BaseBasis(ABC):
    """Abstract base class for basis functions."""
    
    name: str
    
    @property
    @abstractmethod
    def nbasis(self) -> int:
        """Number of basis functions."""
        pass
    
    @abstractmethod
    def evaluate(self, x: Union[float, Array]) -> Array:
        """Evaluate basis functions at given points.
        
        Parameters
        ----------
        x : float or array-like
            Points at which to evaluate basis
        
        Returns
        -------
        Array
            Basis function values, shape (len(x), nbasis)
        """
        pass
    
    def predict(self, x: Union[float, Array], 
                coef: Optional[Array] = None) -> Array:
        """Predict using basis functions.
        
        Parameters
        ----------
        x : float or array-like
            Points at which to predict
        coef : array-like, optional
            Coefficients for basis functions. If None, returns basis evaluation.
        
        Returns
        -------
        Array
            Predictions
        """
        basis_eval = self.evaluate(x)
        if coef is None:
            return basis_eval
        
        coef = np.asarray(coef)
        if coef.shape[0] != self.nbasis:
            raise ValueError(
                f"Coefficient length ({coef.shape[0]}) must match "
                f"number of basis functions ({self.nbasis})"
            )
        
        return basis_eval @ coef
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', nbasis={self.nbasis})"


@dataclass
class BaseContrast(ABC):
    """Abstract base class for contrasts."""
    
    name: str
    
    @property
    @abstractmethod
    def contrast_type(self) -> str:
        """Type of contrast."""
        pass
    
    @abstractmethod
    def compute_weights(self, columns: List[str]) -> Array:
        """Compute contrast weights for given columns.
        
        Parameters
        ----------
        columns : list of str
            Column names from design matrix
        
        Returns
        -------
        Array
            Contrast weights
        """
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"type='{self.contrast_type}')"
        )


class BaseModel(ABC):
    """Abstract base class for models."""
    
    def __init__(self):
        """Initialize model."""
        self._contrasts: Dict[str, ContrastProtocol] = {}
        self._design_matrix: Optional[Array] = None
        self._column_names: Optional[List[str]] = None
    
    @abstractmethod
    def _build_design_matrix(self) -> tuple[Array, List[str]]:
        """Build the design matrix.
        
        Returns
        -------
        matrix : Array
            Design matrix
        columns : list of str
            Column names
        """
        pass
    
    def design_matrix(self) -> Array:
        """Get the design matrix.
        
        Returns
        -------
        Array
            Design matrix
        """
        if self._design_matrix is None:
            self._design_matrix, self._column_names = self._build_design_matrix()
        return self._design_matrix
    
    def column_names(self) -> List[str]:
        """Get column names of design matrix.
        
        Returns
        -------
        list of str
            Column names
        """
        if self._column_names is None:
            self._design_matrix, self._column_names = self._build_design_matrix()
        return self._column_names
    
    def add_contrast(self, contrast: ContrastProtocol) -> None:
        """Add a contrast to the model.
        
        Parameters
        ----------
        contrast : ContrastProtocol
            Contrast to add
        """
        self._contrasts[contrast.name] = contrast
    
    def get_contrast(self, name: str) -> ContrastProtocol:
        """Get contrast by name.
        
        Parameters
        ----------
        name : str
            Contrast name
        
        Returns
        -------
        ContrastProtocol
            The contrast
        
        Raises
        ------
        KeyError
            If contrast not found
        """
        if name not in self._contrasts:
            raise KeyError(f"Contrast '{name}' not found")
        return self._contrasts[name]
    
    def contrast_weights(self, name: str) -> Array:
        """Get contrast weights.
        
        Parameters
        ----------
        name : str
            Contrast name
        
        Returns
        -------
        Array
            Contrast weights
        """
        contrast = self.get_contrast(name)
        columns = self.column_names()
        return contrast.compute_weights(columns)
    
    @property
    def contrasts(self) -> Dict[str, ContrastProtocol]:
        """All contrasts."""
        return self._contrasts.copy()


# Mixin classes for common functionality
class NamedMixin:
    """Mixin for objects with names."""
    
    @property
    def name(self) -> str:
        """Object name."""
        return getattr(self, "_name", self.__class__.__name__)
    
    @name.setter
    def name(self, value: str) -> None:
        """Set object name."""
        self._name = value


class CacheMixin:
    """Mixin for caching computed values."""
    
    def __init__(self):
        """Initialize cache."""
        self._cache: Dict[str, object] = {}
    
    def _get_cached(self, key: str, compute_func, *args: object, **kwargs: object) -> object:
        """Get cached value or compute it.
        
        Parameters
        ----------
        key : str
            Cache key
        compute_func : callable
            Function to compute value if not cached
        *args, **kwargs
            Arguments for compute_func
        
        Returns
        -------
        Any
            Cached or computed value
        """
        if key not in self._cache:
            self._cache[key] = compute_func(*args, **kwargs)
        return self._cache[key]
    
    def _invalidate_cache(self, key: Optional[str] = None) -> None:
        """Invalidate cache.
        
        Parameters
        ----------
        key : str, optional
            Specific key to invalidate. If None, clear entire cache.
        """
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)


class ValidatedMixin:
    """Mixin for objects requiring validation."""
    
    def __init__(self, validate: bool = True):
        """Initialize with optional validation.
        
        Parameters
        ----------
        validate : bool
            Whether to validate on initialization
        """
        self._validated = False
        if validate:
            self.validate()
    
    @abstractmethod
    def _validate_impl(self) -> None:
        """Implementation of validation logic."""
        pass
    
    def validate(self) -> None:
        """Validate object state."""
        self._validate_impl()
        self._validated = True
    
    @property
    def is_validated(self) -> bool:
        """Whether object has been validated."""
        return self._validated