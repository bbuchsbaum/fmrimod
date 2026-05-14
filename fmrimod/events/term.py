"""Event term combination system for interactions and complex events.

An :class:`EventTerm` groups one or more event objects (``EventFactor``,
``EventVariable``, ``EventMatrix``, ``EventBasis``) into a single
model term. When multiple events are combined, the term represents
their interaction (Cartesian product of factor levels, element-wise
product of continuous values).
"""

from __future__ import annotations

from itertools import product
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import BaseEvent, CacheMixin
from ..types import Array


class EventTerm(CacheMixin):
    """A model term combining one or more event objects.

    ``EventTerm`` is the bridge between high-level ``Term`` specifications
    (from the formula system) and the concrete event objects that produce
    design-matrix columns. A single-event term simply delegates to the
    event's ``design_matrix()`` method; a multi-event term computes the
    appropriate interaction (indicator product for categoricals,
    element-wise product for continuous, or a mixed combination).

    Parameters
    ----------
    events : list of BaseEvent
        One or more event objects to combine. At least one is required.
    name : str, optional
        Human-readable name. If ``None``, generated automatically
        (single event: event name; interaction: names joined by ``":"``).
    interaction : bool, default False
        Explicit interaction flag. Automatically set to ``True`` when
        more than one event is provided.

    Attributes
    ----------
    events : list of BaseEvent
        Component event objects.
    name : str
        Term name.
    interaction : bool
        Whether this is an interaction term.

    Examples
    --------
    >>> from fmrimod.events import EventFactor, EventVariable
    >>> cond = EventFactor('cond', onsets=[0, 2, 4], values=['A','B','A'])
    >>> term = EventTerm([cond])
    >>> term.design_matrix(np.arange(0, 10, 2.0)).shape
    (5, 2)

    >>> block = EventFactor('block', onsets=[0, 2, 4], values=['1','1','2'])
    >>> ix = EventTerm([cond, block], interaction=True)
    >>> ix.name
    'cond:block'

    See Also
    --------
    create_interaction : Convenience constructor for interaction terms.
    """
    
    def __init__(
        self,
        events: List[BaseEvent],
        name: Optional[str] = None,
        interaction: bool = False,
    ):
        """Initialize EventTerm."""
        super().__init__()  # Initialize cache
        self.events = events
        self.name = name
        self.interaction = interaction
        
        if not self.events:
            raise ValueError("EventTerm requires at least one event")
        
        # Generate name if not provided
        if self.name is None:
            if self.interaction and len(self.events) > 1:
                self.name = ":".join(e.name for e in self.events)
            else:
                self.name = self.events[0].name
        
        # Set interaction flag if multiple events
        if len(self.events) > 1:
            self.interaction = True
    
    @property
    def n_events(self) -> int:
        """Number of component events."""
        return len(self.events)
    
    @property
    def event_names(self) -> List[str]:
        """Names of component events."""
        return [e.name for e in self.events]
    
    @property
    def is_categorical(self) -> bool:
        """Whether all events are categorical."""
        return all(e.event_type == "categorical" for e in self.events)
    
    @property
    def is_continuous(self) -> bool:
        """Whether all events are continuous."""
        return all(e.event_type in ["continuous", "basis", "matrix"] for e in self.events)
    
    @property
    def is_mixed(self) -> bool:
        """Whether term contains both categorical and continuous events."""
        types = set(e.event_type for e in self.events)
        return "categorical" in types and any(
            t in types for t in ["continuous", "basis", "matrix"]
        )
    
    def get_levels(self) -> List[Tuple[str, ...]]:
        """Get all level combinations for categorical events.
        
        Returns
        -------
        list of tuple
            Each tuple contains one level combination
        """
        if not self.is_categorical:
            return []
        
        # Get levels for each categorical event
        all_levels = []
        for event in self.events:
            if hasattr(event, 'levels'):
                all_levels.append(event.levels)
        
        # Generate all combinations
        if self.interaction:
            return list(product(*all_levels))
        else:
            return [(level,) for level in all_levels[0]]
    
    def get_column_names(self) -> List[str]:
        """Get column names for design matrix.
        
        Returns
        -------
        list of str
            Column names
        """
        if self.is_categorical:
            # For categorical, use level combinations
            levels = self.get_levels()
            if self.interaction:
                return [":".join(level_combo) for level_combo in levels]
            else:
                return [str(level) for level, in levels]
        
        elif len(self.events) == 1:
            # Single non-categorical event
            event = self.events[0]
            if event.event_type == "matrix":
                return event.column_names
            elif event.event_type == "basis":
                return event.basis_names
            else:
                return [event.name]
        
        else:
            # Multiple events (interaction)
            if self.is_continuous:
                return [self.name]
            else:
                # Mixed or complex - generate names
                n_cols = self._get_n_columns()
                return [f"{self.name}_{i+1}" for i in range(n_cols)]
    
    def _get_n_columns(self) -> int:
        """Get number of columns in design matrix."""
        if self.is_categorical:
            return len(self.get_levels())
        elif len(self.events) == 1:
            event = self.events[0]
            if event.event_type == "matrix":
                return event.n_columns
            elif event.event_type == "basis":
                return event.n_basis
            else:
                return 1
        else:
            # Interaction - multiply column counts
            n_cols = 1
            for event in self.events:
                if event.event_type == "categorical":
                    n_cols *= len(event.levels)
                elif event.event_type == "matrix":
                    n_cols *= event.n_columns
                elif event.event_type == "basis":
                    n_cols *= event.n_basis
                # Continuous contributes factor of 1
            return n_cols
    
    def design_matrix(self, sampling_points: Array) -> Array:
        """Generate design matrix for this term.
        
        Parameters
        ----------
        sampling_points : Array
            Time points to evaluate at
        
        Returns
        -------
        Array
            Design matrix columns
        """
        return self._get_cached(
            'design_matrix',
            self._compute_design_matrix,
            sampling_points
        )
    
    def _compute_design_matrix(self, sampling_points: Array) -> Array:
        """Compute design matrix (internal)."""
        if len(self.events) == 1:
            # Single event - just use its design matrix
            return self.events[0].design_matrix(sampling_points)
        
        # Multiple events - compute interaction
        matrices = []
        for event in self.events:
            dm = event.design_matrix(sampling_points)
            matrices.append(dm)
        
        # Combine matrices based on event types
        if self.is_categorical:
            # Categorical interaction - multiply indicators
            return self._categorical_interaction(matrices)
        elif self.is_continuous:
            # Continuous interaction - element-wise multiply
            return self._continuous_interaction(matrices)
        else:
            # Mixed interaction
            return self._mixed_interaction(matrices)
    
    def _categorical_interaction(self, matrices: List[Array]) -> Array:
        """Compute interaction for categorical events."""
        n_points = matrices[0].shape[0]
        
        # Get all level combinations
        levels = self.get_levels()
        n_cols = len(levels)
        
        # Initialize result
        X = np.zeros((n_points, n_cols))
        
        # For each level combination, multiply indicators
        for i, level_combo in enumerate(levels):
            col = np.ones(n_points)
            for j, level in enumerate(level_combo):
                event = self.events[j]
                level_idx = event.levels.index(level)
                col *= matrices[j][:, level_idx]
            X[:, i] = col
        
        return X
    
    def _continuous_interaction(self, matrices: List[Array]) -> Array:
        """Compute interaction for continuous events."""
        # Element-wise multiplication
        result = matrices[0]
        for mat in matrices[1:]:
            # Handle different numbers of columns
            if result.shape[1] == 1 and mat.shape[1] > 1:
                result = result * mat
            elif result.shape[1] > 1 and mat.shape[1] == 1:
                result = result * mat
            elif result.shape[1] == mat.shape[1]:
                result = result * mat
            else:
                # Outer product for different column counts
                n_points = result.shape[0]
                n_cols = result.shape[1] * mat.shape[1]
                new_result = np.zeros((n_points, n_cols))
                
                col = 0
                for i in range(result.shape[1]):
                    for j in range(mat.shape[1]):
                        new_result[:, col] = result[:, i] * mat[:, j]
                        col += 1
                
                result = new_result
        
        return result
    
    def _mixed_interaction(self, matrices: List[Array]) -> Array:
        """Compute interaction for mixed event types."""
        from itertools import product as iter_product

        n_points = matrices[0].shape[0]

        # Separate categorical and continuous events with their matrices
        cat_events = []
        cat_mats = []
        cont_mats = []

        for i, event in enumerate(self.events):
            if event.event_type == "categorical":
                cat_events.append(event)
                cat_mats.append(matrices[i])
            else:
                cont_mats.append(matrices[i])

        # Get all level combinations from categorical events
        if cat_events:
            all_levels = [e.levels for e in cat_events]
            level_combos = list(iter_product(*all_levels))
        else:
            level_combos = [()]

        # Compute continuous part (element-wise multiply if multiple)
        if cont_mats:
            cont_part = cont_mats[0]
            for mat in cont_mats[1:]:
                if cont_part.shape[1] == 1 and mat.shape[1] > 1:
                    cont_part = cont_part * mat
                elif cont_part.shape[1] > 1 and mat.shape[1] == 1:
                    cont_part = cont_part * mat
                elif cont_part.shape[1] == mat.shape[1]:
                    cont_part = cont_part * mat
                else:
                    new = np.zeros((n_points, cont_part.shape[1] * mat.shape[1]))
                    col = 0
                    for ci in range(cont_part.shape[1]):
                        for cj in range(mat.shape[1]):
                            new[:, col] = cont_part[:, ci] * mat[:, cj]
                            col += 1
                    cont_part = new
        else:
            cont_part = None

        # Combine: for each categorical level combo, multiply indicator with continuous
        cols = []
        for combo in level_combos:
            # Build categorical indicator by multiplying level-specific columns
            cat_indicator = np.ones(n_points)
            for j, (event, level) in enumerate(zip(cat_events, combo)):
                level_idx = event.levels.index(level)
                cat_indicator *= cat_mats[j][:, level_idx]

            if cont_part is not None:
                for c in range(cont_part.shape[1]):
                    cols.append((cat_indicator * cont_part[:, c]).reshape(-1, 1))
            else:
                cols.append(cat_indicator.reshape(-1, 1))

        if not cols:
            return np.zeros((n_points, 0))

        return np.hstack(cols)
    
    def __repr__(self) -> str:
        """Rich string representation."""
        event_names = [e.name for e in self.events]
        n_events = len(self.events[0].onsets) if self.events and hasattr(self.events[0], 'onsets') else 0

        # Collect variable info
        var_types = []
        for e in self.events:
            if e.event_type == "categorical":
                n_levels = len(e.levels) if hasattr(e, 'levels') else 0
                var_types.append(f"{e.name}[{n_levels} levels]")
            elif e.event_type == "continuous":
                var_types.append(f"{e.name}[continuous]")
            elif e.event_type == "matrix":
                n_cols = e.n_columns if hasattr(e, 'n_columns') else 0
                var_types.append(f"{e.name}[matrix:{n_cols}cols]")
            else:
                var_types.append(f"{e.name}[{e.event_type}]")

        name_str = self.name or " x ".join(event_names)
        vars_str = ", ".join(var_types)
        n_cols = self._get_n_columns()

        parts = [f"EventTerm '{name_str}'"]
        parts.append(f"  Variables: {vars_str}")
        parts.append(f"  Events: {n_events}, Columns: {n_cols}")

        if self.events and hasattr(self.events[0], 'onsets') and len(self.events[0].onsets) > 0:
            onsets = self.events[0].onsets
            parts.append(f"  Onset range: [{onsets.min():.1f}, {onsets.max():.1f}]s")

        return "\n".join(parts)
    
    def cells(self, drop_empty: bool = True) -> pd.DataFrame:
        """Extract cells (factor combinations) from this term.
        
        Parameters
        ----------
        drop_empty : bool
            Whether to drop cells with zero count
            
        Returns
        -------
        pd.DataFrame
            DataFrame with factor levels as columns and count attribute
        """
        from .cells import cells_event_term
        return cells_event_term(self, drop_empty)
    
    def conditions(
        self,
        drop_empty: bool = True,
        expand_basis: bool = False
    ) -> List[str]:
        """Extract condition names from this term.
        
        Parameters
        ----------
        drop_empty : bool
            Whether to drop conditions with no observations
        expand_basis : bool
            Whether to expand names for basis functions
            
        Returns
        -------
        List[str]
            List of condition names
        """
        from .cells import conditions_event_term
        return conditions_event_term(self, drop_empty, expand_basis)


def create_interaction(*events: BaseEvent, name: Optional[str] = None) -> EventTerm:
    """Create an interaction term from two or more events.

    This is a convenience wrapper around ``EventTerm(..., interaction=True)``.

    Parameters
    ----------
    *events : BaseEvent
        Two or more event objects to interact.
    name : str, optional
        Custom name for the interaction term. If ``None``, names are
        joined with ``":"``.

    Returns
    -------
    EventTerm
        Interaction term with ``interaction=True``.

    Examples
    --------
    >>> from fmrimod.events import EventFactor
    >>> cond = EventFactor('cond', [0, 2, 4], ['A','B','A'])
    >>> block = EventFactor('block', [0, 2, 4], ['1','1','2'])
    >>> ix = create_interaction(cond, block)
    >>> ix.name
    'cond:block'
    """
    return EventTerm(list(events), name=name, interaction=True)