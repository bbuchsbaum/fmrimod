"""Covariate support for fMRI models.

Covariates are terms that are added directly to the design matrix without
HRF convolution. They are useful for including nuisance variables, motion
parameters, physiological measurements, or other regressors.

The main user-facing entry point is the :func:`covariate` function, which
creates a :class:`CovariateTerm` that can be passed to
:func:`~fmrimod.core.event_model.event_model`. Internally, covariate
values are wrapped in :class:`CovariateEvent` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .base import BaseEvent
from .formula.base import Term
from .types import Array

if TYPE_CHECKING:
    from .design.event_model import EventModel


class CovariateTerm(Term):
    """A term representing covariates that bypass HRF convolution.
    
    Covariates are added directly to the design matrix without convolution.
    This is useful for:
    - Motion parameters
    - Physiological measurements (heart rate, respiration)
    - Scanner drift corrections
    - Global signal regressors
    - Behavioral measures that directly correlate with BOLD
    
    Parameters
    ----------
    covariates : str or list of str
        Covariate variable name(s)
    prefix : str, optional
        Prefix to add to covariate names in design matrix
    subset : array-like or callable, optional
        Boolean mask or function to subset the data
    name : str, optional
        Custom name for the term
    
    Examples
    --------
    >>> # Single covariate
    >>> term = CovariateTerm('motion_x')
    >>> 
    >>> # Multiple covariates with prefix
    >>> term = CovariateTerm(['motion_x', 'motion_y', 'motion_z'], 
    ...                      prefix='motion')
    >>> 
    >>> # With subsetting
    >>> term = CovariateTerm('heart_rate', subset=valid_hr_mask)
    """
    
    def __init__(
        self,
        covariates: Union[str, List[str]],
        prefix: Optional[str] = None,
        subset: Optional[Union[Array, callable]] = None,
        name: Optional[str] = None
    ):
        """Initialize covariate term."""
        # Normalize to list
        if isinstance(covariates, str):
            covariates = [covariates]
        
        self.covariates = covariates
        self.prefix = prefix
        self.subset = subset
        self._is_covariate = True
        
        # Generate name if not provided
        if name is None:
            if self.prefix:
                name = f"{self.prefix}_{'_'.join(self.covariates)}"
            else:
                # Use parent's default naming for consistency
                name = None
        
        # Initialize parent Term with events matching covariates
        super().__init__(
            events=covariates,
            hrf=None,  # Covariates are never convolved
            name=name
        )
    
    @property
    def is_covariate(self) -> bool:
        """Whether this is a covariate term."""
        return True
    
    def __repr__(self) -> str:
        """String representation."""
        parts = [f"CovariateTerm({', '.join(self.covariates)}"]
        if self.prefix:
            parts.append(f"prefix={self.prefix}")
        parts.append(")")
        return ", ".join(parts)


class CovariateEvent(BaseEvent):
    """Event representation for covariate data.
    
    This is a special event type that holds covariate data
    for direct inclusion in the design matrix.
    
    Parameters
    ----------
    name : str
        Covariate name
    values : array-like
        Covariate values (must match number of timepoints)
    sampling_points : array-like, optional
        Time points corresponding to values
    
    Examples
    --------
    >>> # Motion parameter
    >>> motion_x = CovariateEvent('motion_x', motion_values)
    >>> 
    >>> # With explicit time points
    >>> hr = CovariateEvent('heart_rate', hr_values, sampling_points=times)
    """
    
    def __init__(
        self,
        name: str,
        values: Array,
        sampling_points: Optional[Array] = None
    ):
        """Initialize covariate event."""
        self.name = name
        self.values = np.asarray(values)
        self.sampling_points = sampling_points
        self._n_timepoints = len(self.values)
        
        # CovariateEvent doesn't have traditional onsets/durations
        # but we need them for the base class
        self.onsets = np.array([])  # Empty for covariates
        self.durations = np.array([])  # Empty for covariates
        
        # Trigger validation
        self.__post_init__()
    
    @property
    def event_type(self) -> str:
        """Event type identifier."""
        return "covariate"
    
    @property
    def n_timepoints(self) -> int:
        """Number of timepoints."""
        return self._n_timepoints
    
    def design_matrix(self, sampling_points: Array) -> Array:
        """Get design matrix representation.
        
        For covariates, this returns the values directly if they match
        the sampling points, or interpolates if necessary.
        
        Parameters
        ----------
        sampling_points : array-like
            Time points to evaluate at
        
        Returns
        -------
        array-like
            Covariate values (n_timepoints, 1)
        """
        # If we have explicit sampling points, check if interpolation is needed
        if self.sampling_points is not None:
            if not np.array_equal(sampling_points, self.sampling_points):
                # Interpolate to match requested sampling points
                from scipy.interpolate import interp1d
                f = interp1d(
                    self.sampling_points, 
                    self.values,
                    kind='linear',
                    fill_value='extrapolate'
                )
                values = f(sampling_points)
            else:
                values = self.values
        else:
            # Assume values match the requested sampling points
            if len(self.values) != len(sampling_points):
                raise ValueError(
                    f"Covariate '{self.name}' has {len(self.values)} values "
                    f"but {len(sampling_points)} sampling points were requested. "
                    "Provide explicit sampling_points if interpolation is needed."
                )
            values = self.values
        
        # Ensure 2D
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        
        return values
    
    def __repr__(self) -> str:
        """String representation."""
        return f"CovariateEvent(name='{self.name}', n_timepoints={self.n_timepoints})"
    
    def _validate(self) -> None:
        """Validate covariate event data."""
        # Values should be numeric
        if self.values.dtype.kind not in 'biufc':
            raise ValueError(f"Covariate values must be numeric, got {self.values.dtype}")
        
        # Values should be finite
        if not np.all(np.isfinite(self.values)):
            raise ValueError("Covariate values must be finite (no NaN or inf)")
        
        # If sampling_points provided, should match length
        if self.sampling_points is not None:
            if len(self.sampling_points) != len(self.values):
                raise ValueError(
                    f"Sampling points length ({len(self.sampling_points)}) "
                    f"must match values length ({len(self.values)})"
                )


def covariate(
    *variables: str,
    data: Optional[pd.DataFrame] = None,
    prefix: Optional[str] = None,
    subset: Optional[Union[Array, callable]] = None,
    name: Optional[str] = None
) -> CovariateTerm:
    """Create a covariate term for non-convolved regressors.
    
    Covariates are added directly to the design matrix without HRF convolution.
    This is useful for nuisance regressors, motion parameters, physiological
    signals, or other variables that should not be convolved.
    
    Parameters
    ----------
    *variables : str
        Variable names to include as covariates
    data : DataFrame, optional
        Data containing the variables (for validation)
    prefix : str, optional
        Prefix to add to variable names in design matrix
    subset : array-like or callable, optional
        Boolean mask or function to subset the data
    name : str, optional
        Custom name for the covariate term
    
    Returns
    -------
    CovariateTerm
        Covariate term specification
    
    Examples
    --------
    >>> # Motion parameters
    >>> motion_term = covariate('motion_x', 'motion_y', 'motion_z',
    ...                        prefix='motion')
    >>> 
    >>> # Physiological regressors
    >>> physio_term = covariate('heart_rate', 'respiration',
    ...                        data=physio_df,
    ...                        name='physiological')
    >>> 
    >>> # In event model
    >>> model = event_model(
    ...     formula="hrf(stimulus) + covariate(motion_x, motion_y)",
    ...     data=df,
    ...     tr=2.0,
    ...     n_scans=200
    ... )
    """
    if not variables:
        raise ValueError("At least one variable must be specified")
    
    # Validate variables exist in data if provided
    if data is not None:
        missing = [v for v in variables if v not in data.columns]
        if missing:
            raise ValueError(
                f"Variables not found in data: {missing}"
            )
    
    return CovariateTerm(
        covariates=list(variables),
        prefix=prefix,
        subset=subset,
        name=name
    )


def create_covariate_events(
    data: pd.DataFrame,
    covariate_names: List[str],
    sampling_info: Any,
    prefix: Optional[str] = None
) -> Dict[str, CovariateEvent]:
    """Create covariate events from data.
    
    Parameters
    ----------
    data : DataFrame
        Data containing covariate values
    covariate_names : list of str
        Names of covariate columns
    sampling_info : SamplingInfo
        Sampling information
    prefix : str, optional
        Prefix for event names
    
    Returns
    -------
    dict
        Dictionary mapping event names to CovariateEvent objects
    """
    events = {}
    
    # Get expected number of timepoints
    n_timepoints = len(sampling_info.samples)
    
    for cov_name in covariate_names:
        if cov_name not in data.columns:
            raise ValueError(f"Covariate '{cov_name}' not found in data")
        
        values = data[cov_name].values
        
        # Validate length
        if len(values) != n_timepoints:
            raise ValueError(
                f"Covariate '{cov_name}' has {len(values)} values but "
                f"sampling frame expects {n_timepoints}"
            )
        
        # Create event name with prefix if provided
        event_name = f"{prefix}_{cov_name}" if prefix else cov_name
        
        # Create covariate event
        events[event_name] = CovariateEvent(
            name=event_name,
            values=values,
            sampling_points=sampling_info.samples
        )
    
    return events