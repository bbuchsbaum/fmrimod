"""EventBasis: Basis function event implementation.

An :class:`EventBasis` wraps a continuous variable and expands it through
a set of basis functions (polynomials, splines, etc.) to capture
non-linear relationships. The expanded values form multiple columns in
the design matrix.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from ..base import BaseEvent
from .matrix import EventMatrix

from ..types import (
    Array,
    BasisProtocol,
    DurationType,
    EventType,
    OnsetType,
    validate_durations,
    validate_onsets,
)


class EventBasis(BaseEvent):
    """Event with basis function expansion.
    
    This represents continuous events that are expanded using basis functions
    (polynomials, splines, etc.) to capture non-linear relationships.
    
    Parameters
    ----------
    name : str
        Name of the event
    onsets : array-like
        Event onset times
    values : array-like
        Values to be expanded by basis functions
    basis : BasisProtocol
        Basis function to apply
    durations : scalar or array-like, optional
        Event durations. If scalar, used for all events. Default is 0.
    
    Attributes
    ----------
    values : Array
        Original values before basis expansion
    basis : BasisProtocol
        Basis function object
    expanded_values : Array
        Values after basis expansion, shape (n_events, n_basis)
    
    Examples
    --------
    >>> from fmrimod.basis import Poly
    >>> 
    >>> # Polynomial expansion of continuous variable
    >>> event = EventBasis(
    ...     name='rating',
    ...     onsets=[1, 2, 3, 4],
    ...     values=[1, 2, 3, 4],
    ...     basis=Poly(degree=2)
    ... )
    >>> 
    >>> # Spline expansion
    >>> from fmrimod.basis import BSpline
    >>> event = EventBasis(
    ...     name='time_in_block',
    ...     onsets=np.arange(10),
    ...     values=np.arange(10),
    ...     basis=BSpline(df=4)
    ... )
    """
    
    # Instance attributes are defined in __init__
    
    def __init__(
        self,
        name: str,
        onsets: OnsetType,
        values: Union[Array, pd.Series],
        basis: BasisProtocol,
        durations: DurationType = 0,
    ):
        """Initialize EventBasis."""
        self.name = name
        self.basis = basis
        self._onsets = onsets
        self._values = values
        self._durations = durations
        
        # Initialize base attributes
        self.onsets = None
        self.durations = None
        self.values = None
        self.expanded_values = None
        
        # Trigger validation and setup
        self.__post_init__()
    
    def _validate(self) -> None:
        """Validate and process event data."""
        # Validate onsets
        self.onsets = validate_onsets(self._onsets)
        
        # Validate durations
        self.durations = validate_durations(self._durations, len(self.onsets))
        
        # Process values
        if isinstance(self._values, pd.Series):
            self.values = self._values.values.astype(np.float64)
        else:
            self.values = np.asarray(self._values, dtype=np.float64)
        
        # Validate values
        if self.values.ndim != 1:
            raise ValueError(
                f"Values must be 1-dimensional, got {self.values.ndim}D"
            )
        
        if len(self.values) != len(self.onsets):
            raise ValueError(
                f"Length mismatch: {len(self.values)} values "
                f"but {len(self.onsets)} onsets"
            )
        
        if not np.all(np.isfinite(self.values)):
            raise ValueError("Values must be finite (no NaN or inf)")
        
        # Validate basis
        if not hasattr(self.basis, 'evaluate'):
            raise TypeError("Basis must have an 'evaluate' method")
        
        # Expand values using basis functions
        self.expanded_values = self._expand_values()
    
    def _expand_values(self) -> Array:
        """Expand values using basis functions.
        
        Returns
        -------
        Array
            Expanded values, shape (n_events, n_basis)
        """
        # Evaluate basis at each value
        expanded = self.basis.evaluate(self.values)
        
        # Ensure 2D
        if expanded.ndim == 1:
            expanded = expanded.reshape(-1, 1)
        
        return expanded
    
    @property
    def event_type(self) -> EventType:
        """Type of event."""
        return "basis"
    
    @property
    def n_basis(self) -> int:
        """Number of basis functions."""
        return self.expanded_values.shape[1]
    
    @property
    def basis_names(self) -> List[str]:
        """Names for each basis function."""
        if hasattr(self.basis, 'basis_names'):
            return self.basis.basis_names
        else:
            return [f"{self.name}_basis{i+1}" for i in range(self.n_basis)]
    
    def design_matrix(self, sampling_points: Array) -> Array:
        """Generate design matrix columns for this event.
        
        Creates multiple columns, one for each basis function.
        
        Parameters
        ----------
        sampling_points : Array
            Time points to evaluate at
        
        Returns
        -------
        Array
            Design matrix columns, shape (n_timepoints, n_basis)
        """
        n_points = len(sampling_points)
        
        # Initialize design matrix
        X = np.zeros((n_points, self.n_basis))
        
        # Fill in expanded values at event times
        for onset, duration, exp_values in zip(
            self.onsets, self.durations, self.expanded_values
        ):
            if duration == 0:
                # Impulse event - find nearest sampling point
                idx = np.argmin(np.abs(sampling_points - onset))
                X[idx, :] += exp_values
            else:
                # Extended event - fill duration
                mask = (sampling_points >= onset) & (
                    sampling_points < onset + duration
                )
                X[mask, :] += exp_values
        
        return X
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame representation.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with onset, duration, original value, and basis columns
        """
        data = {
            'onset': self.onsets,
            'duration': self.durations,
            f'{self.name}_value': self.values,
        }
        
        # Add basis columns
        for i, basis_name in enumerate(self.basis_names):
            data[basis_name] = self.expanded_values[:, i]
        
        return pd.DataFrame(data)
    
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name: str,
        basis: BasisProtocol,
        onset_col: str = 'onset',
        value_col: Optional[str] = None,
        duration_col: str = 'duration',
        **kwargs
    ) -> EventBasis:
        """Create EventBasis from DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing event data
        name : str
            Name for the event
        basis : BasisProtocol
            Basis function to apply
        onset_col : str
            Column name for onsets
        value_col : str, optional
            Column name for values. If None, uses name.
        duration_col : str
            Column name for durations
        **kwargs
            Additional arguments passed to EventBasis
        
        Returns
        -------
        EventBasis
            New EventBasis instance
        """
        if value_col is None:
            value_col = name
        
        durations = df[duration_col] if duration_col in df else 0
        
        return cls(
            name=name,
            onsets=df[onset_col].values,
            values=df[value_col].values,
            basis=basis,
            durations=durations,
            **kwargs
        )
    
    def to_matrix(self) -> EventMatrix:
        """Convert to EventMatrix representation.
        
        Returns
        -------
        EventMatrix
            Multi-column event with expanded values
        """
        from .matrix import EventMatrix
        
        return EventMatrix(
            name=f"{self.name}_expanded",
            onsets=self.onsets,
            values=self.expanded_values,
            durations=self.durations,
            column_names=self.basis_names
        )
    
    def change_basis(self, new_basis: BasisProtocol) -> EventBasis:
        """Create new EventBasis with different basis functions.
        
        Parameters
        ----------
        new_basis : BasisProtocol
            New basis function to apply
        
        Returns
        -------
        EventBasis
            New EventBasis with changed basis
        """
        return EventBasis(
            name=self.name,
            onsets=self.onsets,
            values=self.values,  # Use original values
            basis=new_basis,
            durations=self.durations
        )
    
    def __repr__(self) -> str:
        """String representation."""
        basis_name = getattr(self.basis, 'name', self.basis.__class__.__name__)
        
        return (
            f"EventBasis(name='{self.name}', "
            f"n_events={self.n_events}, "
            f"basis={basis_name}, "
            f"n_basis={self.n_basis})"
        )
