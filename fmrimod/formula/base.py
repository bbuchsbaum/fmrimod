"""Core Term and builder classes for formula specification.

This module defines the :class:`Term` dataclass -- the fundamental
building block for specifying which events, HRFs, and basis functions
are combined into design-matrix columns -- and the
:class:`EventModelBuilder` fluent interface for constructing models
programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..types import (
    Array,
    BasisProtocol,
    EventProtocol,
    HRFProtocol,
    ModelProtocol,
    validate_onsets,
    validate_durations,
)
from ..base import BaseEvent, CacheMixin


@dataclass
class Term:
    """Represents a term in an event model.
    
    A term specifies how one or more events are transformed and combined
    to create columns in the design matrix. This is the core building block
    for all formula interfaces.
    
    Parameters
    ----------
    events : str or list of str
        Event variable name(s). If multiple, creates an interaction.
    hrf : str or HRFProtocol, optional
        HRF specification
    basis : BasisProtocol, optional
        Basis function for continuous variables
    normalize : bool, default False
        If True, peak-normalize each regressor column after convolution
        so that max(abs(col)) == 1.
    summate : bool, default True
        If True, overlapping HRF responses are summed. If False, max is
        used instead. Passed through to fmrimod.regressor().
    kwargs : dict
        Additional parameters
    
    Examples
    --------
    >>> # Single event
    >>> term = Term('condition')
    >>> 
    >>> # Interaction
    >>> term = Term(['condition', 'block'])
    >>> 
    >>> # With transformations
    >>> term = Term('condition').with_hrf('spm_canonical')
    """
    
    events: Union[str, List[str]]
    hrf: Optional[Union[str, HRFProtocol]] = None
    basis: Optional[BasisProtocol] = None
    name: Optional[str] = None
    normalize: bool = False
    summate: bool = True
    _kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Normalize events to list."""
        if isinstance(self.events, str):
            self.events = [self.events]
        
        # Generate default name if not provided
        if self.name is None:
            if len(self.events) == 1:
                self.name = self.events[0]
            else:
                self.name = ":".join(self.events)
    
    def with_hrf(self, hrf: Union[str, HRFProtocol]) -> Term:
        """Set HRF for this term.
        
        Parameters
        ----------
        hrf : str or HRFProtocol
            HRF specification
        
        Returns
        -------
        Term
            Modified term (returns self for chaining)
        """
        self.hrf = hrf
        return self
    
    def with_basis(self, basis: BasisProtocol) -> Term:
        """Set basis function for this term.
        
        Parameters
        ----------
        basis : BasisProtocol
            Basis function
        
        Returns
        -------
        Term
            Modified term (returns self for chaining)
        """
        self.basis = basis
        return self
    
    def with_name(self, name: str) -> Term:
        """Set custom name for this term.
        
        Parameters
        ----------
        name : str
            Term name
        
        Returns
        -------
        Term
            Modified term (returns self for chaining)
        """
        self.name = name
        return self
    
    def set(self, **kwargs) -> Term:
        """Set additional parameters.
        
        Parameters
        ----------
        **kwargs
            Additional parameters
        
        Returns
        -------
        Term
            Modified term (returns self for chaining)
        """
        self._kwargs.update(kwargs)
        return self
    
    @property
    def is_interaction(self) -> bool:
        """Whether this is an interaction term."""
        return len(self.events) > 1
    
    @property
    def kwargs(self) -> Dict[str, Any]:
        """All parameters including HRF, basis, normalize, and summate."""
        params = self._kwargs.copy()
        if self.hrf is not None:
            params['hrf'] = self.hrf
        if self.basis is not None:
            params['basis'] = self.basis
        params['normalize'] = self.normalize
        params['summate'] = self.summate
        return params
    
    def __matmul__(self, transform):
        """Apply transformation using @ operator.

        Supports chaining like: ``term @ basis @ hrf``.

        Parameters
        ----------
        transform : Transform
            A DSL Transform (HRFTransform, BasisTransform, etc.)

        Returns
        -------
        Term
            Transformed term
        """
        # Import here to avoid circular dependency
        from .dsl import Transform
        if isinstance(transform, Transform):
            return transform.apply(self)
        return NotImplemented

    def __repr__(self) -> str:
        """String representation."""
        parts = []

        # Events
        if self.is_interaction:
            event_str = " * ".join(self.events)
            parts.append(f"({event_str})")
        else:
            parts.append(self.events[0])

        # Transformations
        if self.basis is not None:
            parts.append(f"basis={self.basis.name}")
        if self.hrf is not None:
            hrf_name = self.hrf if isinstance(self.hrf, str) else self.hrf.name
            parts.append(f"hrf={hrf_name}")
        
        return f"Term({' | '.join(parts)})"


class EventModelBuilder:
    """Fluent builder for constructing event models step-by-step.

    Provides an explicit, type-safe alternative to the string-formula
    interface. All setter methods return ``self`` so calls can be
    chained.

    Examples
    --------
    >>> import pandas as pd
    >>> from fmrimod.formula.base import EventModelBuilder, Term
    >>> df = pd.DataFrame({
    ...     'onset': [0, 3, 6, 9],
    ...     'condition': ['A', 'B', 'A', 'B'],
    ...     'duration': [1, 1, 1, 1],
    ... })
    >>> model = (EventModelBuilder()
    ...     .set_data(df)
    ...     .set_sampling(sampling_frame)
    ...     .add_term(Term('condition').with_hrf('spm_canonical'))
    ...     .build())

    See Also
    --------
    event_model : Factory function that accepts builders directly.
    Term : Individual term specification.
    """
    
    def __init__(self):
        """Initialize builder."""
        self._data: Optional[pd.DataFrame] = None
        self._sampling: Optional[Any] = None  # Will be SamplingFrame
        self._terms: List[Term] = []
        self._contrasts: Dict[str, Any] = {}
        self._onset_column: str = 'onset'
        self._duration_column: Optional[str] = 'duration'
        self._kwargs: Dict[str, Any] = {}
    
    def set_data(self, data: pd.DataFrame) -> EventModelBuilder:
        """Set data frame containing event information.
        
        Parameters
        ----------
        data : DataFrame
            Event data
        
        Returns
        -------
        EventModelBuilder
            Self for chaining
        """
        self._data = data
        return self
    
    def set_sampling(self, sampling: Any) -> EventModelBuilder:
        """Set sampling frame.
        
        Parameters
        ----------
        sampling : SamplingFrame
            Sampling information
        
        Returns
        -------
        EventModelBuilder
            Self for chaining
        """
        self._sampling = sampling
        return self
    
    def set_onset_column(self, column: str) -> EventModelBuilder:
        """Set column name for onset times.
        
        Parameters
        ----------
        column : str
            Column name
        
        Returns
        -------
        EventModelBuilder
            Self for chaining
        """
        self._onset_column = column
        return self
    
    def set_duration_column(self, column: Optional[str]) -> EventModelBuilder:
        """Set column name for durations.
        
        Parameters
        ----------
        column : str or None
            Column name, or None for impulse events
        
        Returns
        -------
        EventModelBuilder
            Self for chaining
        """
        self._duration_column = column
        return self
    
    def add_term(self, term: Term) -> EventModelBuilder:
        """Add a term to the model.
        
        Parameters
        ----------
        term : Term
            Term to add
        
        Returns
        -------
        EventModelBuilder
            Self for chaining
        """
        self._terms.append(term)
        return self
    
    def add_terms(self, *terms: Term) -> EventModelBuilder:
        """Add multiple terms to the model.
        
        Parameters
        ----------
        *terms : Term
            Terms to add
        
        Returns
        -------
        EventModelBuilder
            Self for chaining
        """
        self._terms.extend(terms)
        return self
    
    def add_contrast(self, name: str, specification: Any) -> EventModelBuilder:
        """Add a contrast to the model.
        
        Parameters
        ----------
        name : str
            Contrast name
        specification : str or dict or ContrastProtocol
            Contrast specification
        
        Returns
        -------
        EventModelBuilder
            Self for chaining
        """
        self._contrasts[name] = specification
        return self
    
    def set(self, **kwargs) -> EventModelBuilder:
        """Set additional parameters.
        
        Parameters
        ----------
        **kwargs
            Additional parameters for EventModel
        
        Returns
        -------
        EventModelBuilder
            Self for chaining
        """
        self._kwargs.update(kwargs)
        return self
    
    def build(self) -> ModelProtocol:
        """Build the event model.
        
        Returns
        -------
        EventModel
            Constructed event model
        
        Raises
        ------
        ValueError
            If required components are missing
        """
        # Validate required components
        if not self._terms:
            raise ValueError("No terms added to model")
        if self._data is None and not hasattr(self, '_events'):
            raise ValueError("No data set for model")
        if self._sampling is None:
            raise ValueError("No sampling frame set for model")
        
        # Import here to avoid circular dependency
        from ..design.event_model import event_model
        
        # Use event_model constructor which handles all the setup
        return event_model(
            self._terms,
            data=self._data,
            sampling_info=self._sampling,
            onset_column=self._onset_column,
            duration_column=self._duration_column,
            **self._kwargs
        )
    
    def __enter__(self) -> EventModelBuilder:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Could auto-build here if desired
        pass


# Convenience functions for creating terms
def term(*events: str, **kwargs) -> Term:
    """Create a term.
    
    Parameters
    ----------
    *events : str
        Event variable name(s)
    **kwargs
        Additional parameters
    
    Returns
    -------
    Term
        New term
    
    Examples
    --------
    >>> # Single event
    >>> t = term('condition')
    >>> 
    >>> # Interaction
    >>> t = term('condition', 'block')
    >>> 
    >>> # With parameters
    >>> t = term('condition', hrf='gamma')
    """
    if len(events) == 1:
        return Term(events[0], **kwargs)
    else:
        return Term(list(events), **kwargs)


def interaction(*events: str, **kwargs) -> Term:
    """Create an interaction term.
    
    Parameters
    ----------
    *events : str
        Event variable names to interact
    **kwargs
        Additional parameters
    
    Returns
    -------
    Term
        Interaction term
    
    Examples
    --------
    >>> t = interaction('condition', 'block')
    """
    if len(events) < 2:
        raise ValueError("Interaction requires at least 2 events")
    return Term(list(events), **kwargs)
