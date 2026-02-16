"""Single dispatch system to replace R's S3 generics."""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import numpy as np
import pandas as pd

from .types import (
    Array,
    BasisProtocol,
    ContrastProtocol,
    DesignMatrixProtocol,
    EventProtocol,
    ModelProtocol,
)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class GenericFunction:
    """Generic function with single dispatch based on first argument type.
    
    This mimics R's S3 generic functions but uses Python's type system.
    """
    
    def __init__(self, name: str, default: Optional[Callable] = None):
        """Initialize generic function.
        
        Parameters
        ----------
        name : str
            Name of the generic function
        default : callable, optional
            Default implementation if no method found
        """
        self.name = name
        self._methods: Dict[Type, Callable] = {}
        self._default = default or self._no_method
    
    def _no_method(self, obj, *args, **kwargs):
        """Default when no method is found."""
        type_name = type(obj).__name__
        raise NotImplementedError(
            f"No method for {self.name}() for type '{type_name}'"
        )
    
    def register(self, cls: Type) -> Callable[[F], F]:
        """Register a method for a specific type.
        
        Parameters
        ----------
        cls : type
            Type to register method for
        
        Returns
        -------
        callable
            Decorator function
        """
        def decorator(func: F) -> F:
            self._methods[cls] = func
            # Also register for subclasses
            for subclass in cls.__subclasses__():
                if subclass not in self._methods:
                    self._methods[subclass] = func
            return func
        return decorator
    
    def __call__(self, obj, *args, **kwargs):
        """Call the appropriate method based on object type."""
        # Look for exact type match first
        obj_type = type(obj)
        if obj_type in self._methods:
            return self._methods[obj_type](obj, *args, **kwargs)
        
        # Then check inheritance chain
        for base in obj_type.__mro__[1:]:
            if base in self._methods:
                return self._methods[base](obj, *args, **kwargs)
        
        # Finally check protocols
        for cls, method in self._methods.items():
            if hasattr(cls, '__subclasshook__'):
                try:
                    if issubclass(obj_type, cls):
                        return method(obj, *args, **kwargs)
                except TypeError:
                    # Python 3.9: Protocols with non-method members
                    # don't support issubclass()
                    pass
        
        # Use default
        return self._default(obj, *args, **kwargs)
    
    def methods(self) -> List[str]:
        """List registered methods."""
        return [f"{cls.__name__}" for cls in self._methods.keys()]


# Create generic functions to replace R's S3 generics

# Design matrix extraction
design_matrix = GenericFunction("design_matrix")

@design_matrix.register(DesignMatrixProtocol)
def _(obj: DesignMatrixProtocol) -> Array:
    """Extract design matrix from objects implementing the protocol."""
    return obj.design_matrix()


# Column names extraction
columns = GenericFunction("columns")

@columns.register(ModelProtocol)
def _(obj: ModelProtocol) -> List[str]:
    """Get column names from model."""
    return obj.column_names()

@columns.register(pd.DataFrame)
def _(obj: pd.DataFrame) -> List[str]:
    """Get column names from DataFrame."""
    return list(obj.columns)

@columns.register(np.ndarray)
def _(obj: np.ndarray) -> List[str]:
    """Generate column names for array."""
    if obj.ndim == 1:
        return ["V1"]
    elif obj.ndim == 2:
        return [f"V{i+1}" for i in range(obj.shape[1])]
    else:
        raise ValueError(f"Cannot get columns for {obj.ndim}D array")


# Conditions/levels extraction
conditions = GenericFunction("conditions")

@conditions.register(EventProtocol)
def _(obj: EventProtocol) -> List[str]:
    """Get conditions from event."""
    if hasattr(obj.values, 'categories'):
        # Categorical
        return list(obj.values.categories)
    else:
        # Continuous - use event name
        return [obj.name]


# Cell counting
cells = GenericFunction("cells")

@cells.register(EventProtocol)
def _(obj: EventProtocol) -> int:
    """Count cells in event."""
    if hasattr(obj.values, 'categories'):
        return len(obj.values.categories)
    else:
        return 1


# Element extraction
elements = GenericFunction("elements")

@elements.register(EventProtocol)
def _(obj: EventProtocol) -> Union[List[str], Array]:
    """Get unique elements from event."""
    if hasattr(obj.values, 'categories'):
        return list(obj.values.categories)
    else:
        return np.unique(obj.values)


# Onset extraction
onsets = GenericFunction("onsets")

@onsets.register(EventProtocol)
def _(obj: EventProtocol) -> Array:
    """Get onsets from event."""
    return obj.onsets

@onsets.register(pd.DataFrame)
def _(obj: pd.DataFrame, column: str = "onset") -> Array:
    """Get onsets from DataFrame."""
    if column not in obj.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    return obj[column].values


# Duration extraction
durations = GenericFunction("durations")

@durations.register(EventProtocol)
def _(obj: EventProtocol) -> Array:
    """Get durations from event."""
    return obj.durations

@durations.register(pd.DataFrame)
def _(obj: pd.DataFrame, column: str = "duration") -> Array:
    """Get durations from DataFrame."""
    if column not in obj.columns:
        # Default to 0 duration (impulse)
        return np.zeros(len(obj))
    return obj[column].values


# Contrast weights computation
contrast_weights = GenericFunction("contrast_weights")

@contrast_weights.register(ContrastProtocol)
def _(obj: ContrastProtocol, columns: List[str]) -> Array:
    """Compute contrast weights."""
    return obj.compute_weights(columns)


# Number of basis functions
nbasis = GenericFunction("nbasis")

@nbasis.register(BasisProtocol)
def _(obj: BasisProtocol) -> int:
    """Get number of basis functions."""
    if hasattr(obj, 'nbasis'):
        return obj.nbasis
    elif hasattr(obj, 'n_basis'):
        return obj.n_basis
    else:
        raise AttributeError(f"Object {type(obj)} has no nbasis or n_basis attribute")


# Generic construction
construct = GenericFunction("construct")

@construct.register(BasisProtocol)
def _(obj: BasisProtocol, x: Array) -> Array:
    """Construct basis matrix from basis object."""
    if hasattr(obj, 'evaluate'):
        return obj.evaluate(x)
    elif hasattr(obj, 'predict'):
        return obj.predict(x)
    elif callable(obj):
        return obj(x)
    else:
        raise AttributeError(f"Object {type(obj)} has no evaluate method")


# Register ParametricBasis directly for Python 3.9 compatibility
# (runtime_checkable Protocols with non-method members don't support issubclass)
try:
    from .basis.base import ParametricBasis as _ParametricBasis

    @nbasis.register(_ParametricBasis)
    def _(obj) -> int:
        if hasattr(obj, 'nbasis'):
            return obj.nbasis
        elif hasattr(obj, 'n_basis'):
            return obj.n_basis
        else:
            raise AttributeError(f"Object {type(obj)} has no nbasis or n_basis attribute")

    @construct.register(_ParametricBasis)
    def _(obj, x: Array) -> Array:
        return obj.evaluate(x)
except ImportError:
    pass


# Type checking generics
is_categorical = GenericFunction("is_categorical", default=lambda x: False)
is_continuous = GenericFunction("is_continuous", default=lambda x: False)

@is_categorical.register(EventProtocol)
def _(obj: EventProtocol) -> bool:
    """Check if event is categorical."""
    return obj.event_type == "categorical"

@is_continuous.register(EventProtocol)
def _(obj: EventProtocol) -> bool:
    """Check if event is continuous."""
    return obj.event_type == "continuous"

@is_categorical.register(pd.Series)
def _(obj: pd.Series) -> bool:
    """Check if Series is categorical."""
    return isinstance(obj.dtype, pd.CategoricalDtype) or obj.dtype == object

@is_continuous.register(pd.Series)
def _(obj: pd.Series) -> bool:
    """Check if Series is continuous."""
    return pd.api.types.is_numeric_dtype(obj)


# Utility function for method dispatch debugging
def list_methods(generic: GenericFunction) -> None:
    """List all methods for a generic function.
    
    Parameters
    ----------
    generic : GenericFunction
        Generic function to inspect
    """
    print(f"Methods for {generic.name}():")
    for method in generic.methods():
        print(f"  - {method}")


# HRF dispatch
def get_hrf(hrf: Union[str, Dict[str, Any], Any], **kwargs) -> Any:
    """Get HRF function by name or return as-is if already an HRF.
    
    Parameters
    ----------
    hrf : str, dict, or HRF
        HRF specification. Can be:
        - String name (e.g., 'spm', 'gamma')
        - Dictionary with 'name' and parameters
        - HRF object (returned as-is)
    **kwargs
        Additional parameters for HRF
    
    Returns
    -------
    HRF
        HRF function object
    """
    if isinstance(hrf, str):
        # Try fmrimod HRFs first
        from .hrf_dispatch import get_hrf as _get_hrf
        try:
            return _get_hrf(hrf, **kwargs)
        except ValueError:
            # If not found, try fmrimod
            try:
                from .hrf_integration import get_fmrimod
                return get_fmrimod(hrf, **kwargs)
            except (ImportError, ValueError) as err:
                raise ValueError(f"Unknown HRF '{hrf}'") from err
    elif isinstance(hrf, dict):
        # Dictionary specification - make a copy to avoid modifying original
        hrf_dict = hrf.copy()
        hrf_name = hrf_dict.pop('name', 'spm')
        params = {**hrf_dict, **kwargs}
        return get_hrf(hrf_name, **params)
    else:
        # Already an HRF object
        return hrf


# Export main generic functions
__all__ = [
    "get_hrf",
    "GenericFunction",
    "design_matrix",
    "columns",
    "conditions",
    "cells",
    "elements",
    "onsets",
    "durations",
    "contrast_weights",
    "nbasis",
    "construct",
    "is_categorical",
    "is_continuous",
    "list_methods",
]
