"""EventFactor: Categorical event implementation.

An :class:`EventFactor` represents experimental events that take on
discrete categorical values (e.g., stimulus condition, task type). Each
unique value is called a "level", and the design matrix contains one
indicator column per level.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

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


class EventFactor(BaseEvent):
    """Categorical event with discrete levels.
    
    This represents events that take on discrete values (conditions, 
    categories, etc.) in an fMRI experiment.
    
    Parameters
    ----------
    name : str
        Name of the event
    onsets : array-like
        Event onset times
    values : array-like
        Categorical values for each event
    durations : scalar or array-like, optional
        Event durations. If scalar, used for all events. Default is 0.
    levels : list, optional
        Explicit ordering of factor levels. If None, inferred from data.
    contrasts : dict, optional
        Contrast coding for levels
    
    Attributes
    ----------
    values : pd.Categorical
        Categorical representation of event values
    levels : list
        Ordered factor levels
    n_levels : int
        Number of unique levels
    
    Examples
    --------
    >>> # Simple categorical event
    >>> event = EventFactor(
    ...     name='condition',
    ...     onsets=[1, 2, 3, 4],
    ...     values=['A', 'B', 'A', 'B']
    ... )
    >>> 
    >>> # With explicit levels and durations
    >>> event = EventFactor(
    ...     name='condition',
    ...     onsets=[1, 2, 3, 4],
    ...     values=['easy', 'hard', 'easy', 'medium'],
    ...     levels=['easy', 'medium', 'hard'],
    ...     durations=1.5
    ... )
    """
    
    # Instance attributes are defined in __init__
    
    def __init__(
        self,
        name: str,
        onsets: OnsetType,
        values: Union[List, Array, pd.Series],
        durations: DurationType = 0,
        levels: Optional[List[str]] = None,
        contrasts: Optional[Dict[str, Array]] = None,
    ):
        """Initialize EventFactor."""
        self.name = name
        self.levels = levels
        self.contrasts = contrasts
        self._onsets = onsets
        self._values = values
        self._durations = durations
        
        # Initialize base attributes
        self.onsets = None
        self.durations = None
        self._categorical_values = None
        
        # Trigger validation and setup
        self.__post_init__()
    
    def _validate(self) -> None:
        """Validate and process event data."""
        # Validate onsets
        self.onsets = validate_onsets(self._onsets)
        
        # Validate durations
        self.durations = validate_durations(self._durations, len(self.onsets))
        
        # Process categorical values
        if isinstance(self._values, pd.Categorical):
            self._categorical_values = self._values
        else:
            # Convert to categorical
            self._categorical_values = pd.Categorical(
                self._values,
                categories=self.levels,
                ordered=self.levels is not None
            )
        
        # Update levels if not provided
        if self.levels is None:
            self.levels = list(self._categorical_values.categories)
        
        # Validate lengths match
        if len(self._categorical_values) != len(self.onsets):
            raise ValueError(
                f"Length mismatch: {len(self._categorical_values)} values "
                f"but {len(self.onsets)} onsets"
            )

        na_mask = pd.isna(self._categorical_values)
        if np.any(na_mask):
            raw_values = np.asarray(self._values, dtype=object)
            missing_levels = sorted(
                {str(v) for v in raw_values[na_mask] if not pd.isna(v)}
            )
            if missing_levels:
                raise ValueError(
                    "Factor values include levels not present in provided levels: "
                    + ", ".join(missing_levels)
                )
            raise ValueError("Factor values cannot contain missing values")
    
    @property
    def event_type(self) -> EventType:
        """Type of event."""
        return "categorical"
    
    @property
    def values(self) -> pd.Categorical:
        """Event values."""
        return self._categorical_values
    
    @property
    def n_levels(self) -> int:
        """Number of unique levels."""
        return len(self.levels)
    
    def get_level_indices(self, level: str) -> Array:
        """Get indices where events have a specific level.
        
        Parameters
        ----------
        level : str
            Level to find
        
        Returns
        -------
        Array
            Boolean array indicating where level occurs
        """
        return self.values == level
    
    def get_level_onsets(self, level: str) -> Array:
        """Get onset times for a specific level.
        
        Parameters
        ----------
        level : str
            Level to get onsets for
        
        Returns
        -------
        Array
            Onset times for the level
        """
        indices = self.get_level_indices(level)
        return self.onsets[indices]
    
    def get_level_durations(self, level: str) -> Array:
        """Get durations for a specific level.
        
        Parameters
        ----------
        level : str
            Level to get durations for
        
        Returns
        -------
        Array
            Durations for the level
        """
        indices = self.get_level_indices(level)
        return self.durations[indices]
    
    def split_by_level(self) -> Dict[str, EventFactor]:
        """Split into separate EventFactor objects by level.
        
        Returns
        -------
        dict
            Dictionary mapping level names to EventFactor objects
        """
        result = {}
        for level in self.levels:
            indices = self.get_level_indices(level)
            if np.any(indices):
                result[level] = EventFactor(
                    name=f"{self.name}_{level}",
                    onsets=self.onsets[indices],
                    values=[level] * np.sum(indices),
                    durations=self.durations[indices],
                    levels=[level]
                )
        return result
    
    def design_matrix(self, sampling_points: Array) -> Array:
        """Generate design matrix columns for this event.
        
        Creates indicator columns for each level (dummy coding).
        
        Parameters
        ----------
        sampling_points : Array
            Time points to evaluate at
        
        Returns
        -------
        Array
            Design matrix columns, shape (n_timepoints, n_levels)
        """
        n_points = len(sampling_points)
        n_levels = self.n_levels
        
        # Initialize design matrix
        X = np.zeros((n_points, n_levels))
        
        # For each level, create indicator column
        for i, level in enumerate(self.levels):
            level_onsets = self.get_level_onsets(level)
            level_durations = self.get_level_durations(level)
            
            # Create boxcar function for this level
            for onset, duration in zip(level_onsets, level_durations):
                if duration == 0:
                    # Zero duration - create impulse at nearest sampling point
                    nearest_idx = np.argmin(np.abs(sampling_points - onset))
                    X[nearest_idx, i] += 1.0
                else:
                    # Non-zero duration - create boxcar
                    mask = (sampling_points >= onset) & (
                        sampling_points < onset + duration
                    )
                    X[mask, i] += 1.0
        
        return X
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame representation.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with onset, duration, and value columns
        """
        return pd.DataFrame({
            'onset': self.onsets,
            'duration': self.durations,
            self.name: self.values
        })
    
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name: str,
        onset_col: str = 'onset',
        value_col: Optional[str] = None,
        duration_col: str = 'duration',
        **kwargs: object
    ) -> EventFactor:
        """Create EventFactor from DataFrame.
        
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
            Additional arguments passed to EventFactor
        
        Returns
        -------
        EventFactor
            New EventFactor instance
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
    
    def __repr__(self) -> str:
        """String representation."""
        level_str = ", ".join(self.levels[:3])
        if self.n_levels > 3:
            level_str += ", ..."
        
        return (
            f"EventFactor(name='{self.name}', "
            f"n_events={self.n_events}, "
            f"levels=[{level_str}])"
        )
