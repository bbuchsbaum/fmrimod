"""EventVariable: Continuous event implementation.

An :class:`EventVariable` represents experimental events with continuous
numeric values (e.g., subjective ratings, reaction times, parametric
modulators). Values are optionally centered and/or scaled before being
placed into the design matrix.
"""

from __future__ import annotations

from typing import List, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..base import BaseEvent
if TYPE_CHECKING:
    from .factor import EventFactor

from ..types import (
    Array,
    DurationType,
    EventType,
    OnsetType,
    validate_durations,
    validate_onsets,
)


class EventVariable(BaseEvent):
    """Continuous event with numeric values.
    
    This represents events with continuous values (ratings, reaction times,
    parametric modulators, etc.) in an fMRI experiment.
    
    Parameters
    ----------
    name : str
        Name of the event
    onsets : array-like
        Event onset times
    values : array-like
        Continuous values for each event
    durations : scalar or array-like, optional
        Event durations. If scalar, used for all events. Default is 0.
    center : bool, optional
        Whether to center values (subtract mean). Default is True.
    scale : bool, optional
        Whether to scale values (divide by std). Default is False.
    
    Attributes
    ----------
    values : Array
        Numeric values (possibly centered/scaled)
    raw_values : Array
        Original values before centering/scaling
    
    Examples
    --------
    >>> # Simple continuous event
    >>> event = EventVariable(
    ...     name='rating',
    ...     onsets=[1, 2, 3, 4],
    ...     values=[7.5, 3.2, 8.1, 5.5]
    ... )
    >>> 
    >>> # With scaling and custom durations
    >>> event = EventVariable(
    ...     name='reaction_time',
    ...     onsets=[1, 2, 3, 4],
    ...     values=[0.5, 0.8, 0.3, 0.6],
    ...     durations=[1, 1, 2, 1],
    ...     center=True,
    ...     scale=True
    ... )
    """
    
    # Instance attributes are defined in __init__
    
    def __init__(
        self,
        name: str,
        onsets: OnsetType,
        values: Union[Array, pd.Series],
        durations: DurationType = 0,
        center: bool = True,
        scale: bool = False,
    ):
        """Initialize EventVariable."""
        self.name = name
        self.center = center
        self.scale = scale
        self._onsets = onsets
        self._values = values
        self._durations = durations
        
        # Initialize base attributes
        self.onsets = None
        self.durations = None
        self.values = None
        self.raw_values = None
        
        # Trigger validation and setup
        self.__post_init__()
    
    def _validate(self) -> None:
        """Validate and process event data."""
        # Validate onsets
        self.onsets = validate_onsets(self._onsets)
        
        # Validate durations
        self.durations = validate_durations(self._durations, len(self.onsets))
        
        # Process continuous values
        if isinstance(self._values, pd.Series):
            self.raw_values = self._values.values.astype(np.float64)
        else:
            self.raw_values = np.asarray(self._values, dtype=np.float64)
        
        # Validate values
        if self.raw_values.ndim != 1:
            raise ValueError(
                f"Values must be 1-dimensional, got {self.raw_values.ndim}D"
            )
        
        if len(self.raw_values) != len(self.onsets):
            raise ValueError(
                f"Length mismatch: {len(self.raw_values)} values "
                f"but {len(self.onsets)} onsets"
            )
        
        if not np.all(np.isfinite(self.raw_values)):
            raise ValueError("Values must be finite (no NaN or inf)")
        
        # Apply centering and scaling
        self.values = self._transform_values(self.raw_values)
    
    def _transform_values(self, values: Array) -> Array:
        """Apply centering and/or scaling to values.
        
        Parameters
        ----------
        values : Array
            Raw values to transform
        
        Returns
        -------
        Array
            Transformed values
        """
        result = values.copy()
        
        if self.center:
            result = result - np.mean(result)
        
        if self.scale:
            std = np.std(result)
            if std > 0:
                result = result / std
            else:
                # All values are the same
                result = np.zeros_like(result)
        
        return result
    
    @property
    def event_type(self) -> EventType:
        """Type of event."""
        return "continuous"
    
    @property
    def mean(self) -> float:
        """Mean of raw values."""
        return float(np.mean(self.raw_values))
    
    @property
    def std(self) -> float:
        """Standard deviation of raw values."""
        return float(np.std(self.raw_values))
    
    @property
    def min(self) -> float:
        """Minimum of raw values."""
        return float(np.min(self.raw_values))
    
    @property
    def max(self) -> float:
        """Maximum of raw values."""
        return float(np.max(self.raw_values))
    
    def design_matrix(self, sampling_points: Array) -> Array:
        """Generate design matrix column for this event.
        
        Creates a single column with the continuous values at event times.
        
        Parameters
        ----------
        sampling_points : Array
            Time points to evaluate at
        
        Returns
        -------
        Array
            Design matrix column, shape (n_timepoints, 1)
        """
        n_points = len(sampling_points)
        
        # Initialize design matrix column
        X = np.zeros((n_points, 1))
        
        # Fill in values at event times
        for onset, duration, value in zip(self.onsets, self.durations, self.values):
            if duration == 0:
                # Impulse event - find nearest sampling point
                idx = np.argmin(np.abs(sampling_points - onset))
                X[idx, 0] = value
            else:
                # Extended event - fill duration
                mask = (sampling_points >= onset) & (
                    sampling_points < onset + duration
                )
                X[mask, 0] = value
        
        return X
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame representation.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with onset, duration, value, and raw_value columns
        """
        return pd.DataFrame({
            'onset': self.onsets,
            'duration': self.durations,
            self.name: self.values,
            f'{self.name}_raw': self.raw_values
        })
    
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name: str,
        onset_col: str = 'onset',
        value_col: Optional[str] = None,
        duration_col: str = 'duration',
        **kwargs
    ) -> EventVariable:
        """Create EventVariable from DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing event data
        name : str
            Name for the event
        onset_col : str
            Column name for onsets
        value_col : str, optional
            Column name for values. If None, uses name.
        duration_col : str
            Column name for durations
        **kwargs
            Additional arguments passed to EventVariable
        
        Returns
        -------
        EventVariable
            New EventVariable instance
        """
        if value_col is None:
            value_col = name
        
        durations = df[duration_col] if duration_col in df else 0
        
        return cls(
            name=name,
            onsets=df[onset_col].values,
            values=df[value_col].values,
            durations=durations,
            **kwargs
        )
    
    def bin_values(self, n_bins: int = 3, 
                   labels: Optional[List[str]] = None) -> EventFactor:
        """Convert to categorical by binning values.
        
        Parameters
        ----------
        n_bins : int
            Number of bins
        labels : list of str, optional
            Labels for bins. If None, uses Low/Medium/High style.
        
        Returns
        -------
        EventFactor
            Categorical event with binned values
        """
        from .factor import EventFactor
        
        # Create bins
        bins = np.quantile(self.raw_values, np.linspace(0, 1, n_bins + 1))
        bins[0] -= 1e-10  # Ensure leftmost edge includes minimum
        bins[-1] += 1e-10  # Ensure rightmost edge includes maximum
        
        # Bin the values
        binned = pd.cut(self.raw_values, bins=bins, labels=labels)
        
        return EventFactor(
            name=f"{self.name}_binned",
            onsets=self.onsets,
            values=binned,
            durations=self.durations
        )
    
    def __repr__(self) -> str:
        """String representation."""
        transform_str = []
        if self.center:
            transform_str.append("centered")
        if self.scale:
            transform_str.append("scaled")
        
        transform = ", ".join(transform_str) if transform_str else "none"
        
        return (
            f"EventVariable(name='{self.name}', "
            f"n_events={self.n_events}, "
            f"range=[{self.min:.2f}, {self.max:.2f}], "
            f"transform={transform})"
        )
