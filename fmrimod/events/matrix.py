"""EventMatrix: Multi-column event implementation.

An :class:`EventMatrix` represents events with multiple simultaneous
continuous values (e.g., multi-dimensional parametric modulators,
pre-computed basis-function coefficients, or motion parameters at each
event). Each column is convolved independently.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd

from ..base import BaseEvent
from ..types import (
    Array,
    DurationType,
    EventType,
    OnsetType,
    validate_durations,
    validate_onsets,
)
from .variable import EventVariable


# Scoped/strict divergence (bd-01KRNN0H73CCYGFJSJ30JPVFTW): under scoped
# mypy (--follow-imports=skip) BaseEvent is opaque Any so this subclass
# REQUIRES the [misc] ignore; full-strict resolves BaseEvent and flags it
# unused. The __post_init__ no-untyped-call leg was resolved by typing
# BaseEvent.__post_init__ in base.py, so the [misc] ignore is now the
# sole residue. Scoped is the epic gate so the ignore stays.
class EventMatrix(BaseEvent):  # type: ignore[misc]
    """Multi-column event representation.
    
    This represents events with multiple simultaneous values, such as
    multi-dimensional parametric modulators or pre-computed design columns.
    
    Parameters
    ----------
    name : str
        Name of the event
    onsets : array-like
        Event onset times
    values : array-like
        Matrix of values, shape (n_events, n_columns)
    durations : scalar or array-like, optional
        Event durations. If scalar, used for all events. Default is 0.
    column_names : list of str, optional
        Names for each column. If None, uses generic names.
    
    Attributes
    ----------
    values : Array
        Matrix of event values
    n_columns : int
        Number of columns in the matrix
    column_names : list of str
        Names of columns
    
    Examples
    --------
    >>> # Multi-dimensional parametric modulator
    >>> event = EventMatrix(
    ...     name='motion',
    ...     onsets=[1, 2, 3, 4],
    ...     values=[[0.1, 0.2, 0.3],
    ...             [0.2, 0.1, 0.4],
    ...             [0.3, 0.3, 0.2],
    ...             [0.1, 0.4, 0.3]],
    ...     column_names=['x', 'y', 'z']
    ... )
    >>> 
    >>> # Pre-computed regressors
    >>> event = EventMatrix(
    ...     name='custom',
    ...     onsets=np.arange(10),
    ...     values=np.random.randn(10, 3),
    ...     durations=1.0
    ... )
    """
    
    # Instance attributes are defined in __init__
    
    def __init__(
        self,
        name: str,
        onsets: OnsetType,
        values: Union[Array, pd.DataFrame],
        durations: DurationType = 0,
        column_names: Optional[List[str]] = None,
    ):
        """Initialize EventMatrix."""
        self.name = name
        self.column_names: Optional[List[str]] = column_names
        self._onsets = onsets
        self._values = values
        self._durations = durations

        # Initialize base attributes — typed as Optional so post-init narrowing
        # is explicit. _validate() reassigns these to concrete arrays.
        self.onsets: Optional[Array] = None
        self.durations: Optional[Array] = None
        self.values: Optional[Array] = None

        # Trigger validation and setup
        self.__post_init__()

    def _validate(self) -> None:
        """Validate and process event data."""
        # Validate onsets
        self.onsets = validate_onsets(self._onsets)
        assert self.onsets is not None

        # Validate durations
        self.durations = validate_durations(self._durations, len(self.onsets))

        # Process matrix values
        if isinstance(self._values, pd.DataFrame):
            values = cast(Array, self._values.values.astype(np.float64))
            if self.column_names is None:
                self.column_names = list(self._values.columns)
        else:
            values = np.asarray(self._values, dtype=np.float64)

        # Ensure 2D
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        elif values.ndim != 2:
            raise ValueError(
                f"Values must be 1D or 2D, got {values.ndim}D"
            )

        # Validate shape
        if values.shape[0] != len(self.onsets):
            raise ValueError(
                f"Shape mismatch: {values.shape[0]} rows "
                f"but {len(self.onsets)} onsets"
            )

        if values.shape[1] == 0:
            raise ValueError("Values must contain at least one column")

        # Validate finite
        if not np.all(np.isfinite(values)):
            raise ValueError("Values must be finite (no NaN or inf)")
        self.values = values
        
        # Set default column names if needed
        if self.column_names is None:
            self.column_names = [
                f"{self.name}_{i+1}" for i in range(self.n_columns)
            ]
        elif len(self.column_names) != self.n_columns:
            raise ValueError(
                f"Number of column names ({len(self.column_names)}) "
                f"must match number of columns ({self.n_columns})"
            )
        elif pd.Index(self.column_names).nunique() != len(self.column_names):
            raise ValueError("column_names must be unique")
    
    @property
    def event_type(self) -> EventType:
        """Type of event."""
        return "matrix"
    
    @property
    def n_columns(self) -> int:
        """Number of columns."""
        assert self.values is not None
        return int(self.values.shape[1])

    def get_column(self, index: Union[int, str]) -> Array:
        """Get a specific column.

        Parameters
        ----------
        index : int or str
            Column index or name

        Returns
        -------
        Array
            Column values
        """
        assert self.column_names is not None and self.values is not None
        if isinstance(index, str):
            if index not in self.column_names:
                raise KeyError(f"Column '{index}' not found")
            index = self.column_names.index(index)

        return self.values[:, index]
    
    def design_matrix(self, sampling_points: Array) -> Array:
        """Generate design matrix columns for this event.
        
        Creates multiple columns, one for each column in the value matrix.
        
        Parameters
        ----------
        sampling_points : Array
            Time points to evaluate at
        
        Returns
        -------
        Array
            Design matrix columns, shape (n_timepoints, n_columns)
        """
        n_points = len(sampling_points)
        
        # Initialize design matrix
        X = np.zeros((n_points, self.n_columns))
        
        # Fill in values at event times
        assert self.onsets is not None and self.durations is not None and self.values is not None
        for onset, duration, values in zip(self.onsets, self.durations, self.values):
            if duration == 0:
                # Impulse event - find nearest sampling point
                idx = np.argmin(np.abs(sampling_points - onset))
                X[idx, :] += values
            else:
                # Extended event - fill duration
                mask = (sampling_points >= onset) & (
                    sampling_points < onset + duration
                )
                X[mask, :] += values
        
        return X
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame representation.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with onset, duration, and value columns
        """
        data: Dict[str, Any] = {
            'onset': self.onsets,
            'duration': self.durations,
        }

        # Add value columns
        assert self.column_names is not None and self.values is not None
        for i, col_name in enumerate(self.column_names):
            data[col_name] = self.values[:, i]

        return pd.DataFrame(data)
    
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name: str,
        onset_col: str = 'onset',
        value_cols: Optional[List[str]] = None,
        duration_col: str = 'duration',
        **kwargs: object
    ) -> EventMatrix:
        """Create EventMatrix from DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing event data
        name : str
            Name for the event
        onset_col : str
            Column name for onsets
        value_cols : list of str, optional
            Column names for values. If None, uses all numeric columns
            except onset and duration.
        duration_col : str
            Column name for durations
        **kwargs
            Additional arguments passed to EventMatrix
        
        Returns
        -------
        EventMatrix
            New EventMatrix instance
        """
        if value_cols is None:
            # Find all numeric columns except onset and duration
            exclude = [onset_col, duration_col]
            value_cols = [
                col for col in df.select_dtypes(include=[np.number]).columns
                if col not in exclude
            ]
        
        if not value_cols:
            raise ValueError("No value columns found")
        
        durations = df[duration_col] if duration_col in df else 0
        
        return cls(
            name=name,
            onsets=np.asarray(df[onset_col].values, dtype=np.float64),
            values=df[value_cols].values,
            durations=durations,
            column_names=value_cols,
            **cast("dict[str, Any]", kwargs),
        )
    
    def split_columns(self) -> Dict[str, EventVariable]:
        """Split into separate EventVariable objects by column.
        
        Returns
        -------
        dict
            Dictionary mapping column names to EventVariable objects
        """
        from .variable import EventVariable

        result: Dict[str, EventVariable] = {}
        assert self.column_names is not None and self.values is not None
        assert self.onsets is not None and self.durations is not None
        for i, col_name in enumerate(self.column_names):
            result[col_name] = EventVariable(
                name=col_name,
                onsets=self.onsets,
                values=self.values[:, i],
                durations=self.durations,
                center=False,  # Already processed
                scale=False
            )
        return result
    
    def apply_transform(self, transform: Union[Array, Callable[..., Any]]) -> EventMatrix:
        """Apply transformation to values.
        
        Parameters
        ----------
        transform : array or callable
            If array, matrix multiply (values @ transform).
            If callable, apply element-wise.
        
        Returns
        -------
        EventMatrix
            New EventMatrix with transformed values
        """
        assert self.values is not None
        assert self.onsets is not None and self.durations is not None
        if callable(transform):
            new_values = transform(self.values)
        else:
            transform = np.asarray(transform)
            new_values = self.values @ transform
        
        return EventMatrix(
            name=f"{self.name}_transformed",
            onsets=self.onsets,
            values=new_values,
            durations=self.durations
        )
    
    def __repr__(self) -> str:
        """String representation."""
        assert self.column_names is not None
        col_str = ", ".join(self.column_names[:3])
        if self.n_columns > 3:
            col_str += ", ..."
        
        return (
            f"EventMatrix(name='{self.name}', "
            f"n_events={self.n_events}, "
            f"n_columns={self.n_columns}, "
            f"columns=[{col_str}])"
        )
