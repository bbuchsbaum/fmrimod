"""Event table functionality for extracting event combinations.

This module implements the :func:`event_table` generic function
(``singledispatch``-based) that extracts tables of unique event
combinations from ``EventTerm``, ``EventFactor``, ``EventVariable``,
``EventMatrix``, ``EventBasis``, and ``EventModel`` objects.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..types import EventProtocol
from .basis import EventBasis
from .factor import EventFactor
from .matrix import EventMatrix
from .term import EventTerm
from .variable import EventVariable


@singledispatch
def event_table(x, **kwargs) -> pd.DataFrame:
    """Extract event table from an object.
    
    This generic function extracts a table containing all unique
    combinations of event levels/values for a given object.
    
    Parameters
    ----------
    x : object
        Object to extract event table from
    **kwargs
        Additional arguments depending on object type
    
    Returns
    -------
    pd.DataFrame
        Table of event combinations
    
    Raises
    ------
    NotImplementedError
        If no method exists for the object type
    
    Examples
    --------
    >>> # From EventTerm
    >>> term = EventTerm([event1, event2])
    >>> table = event_table(term)
    >>> 
    >>> # From EventModel
    >>> model = event_model("condition * block", data=df)
    >>> table = event_table(model)
    """
    raise NotImplementedError(
        f"No event_table method for type {type(x).__name__}"
    )


@event_table.register(EventTerm)
def _(x: EventTerm, **kwargs) -> pd.DataFrame:
    """Extract event table from EventTerm.
    
    Parameters
    ----------
    x : EventTerm
        Event term object
    **kwargs
        Additional arguments (ignored)
    
    Returns
    -------
    pd.DataFrame
        Table with columns for each event in the term
    """
    # For single events, return their unique values
    if len(x.events) == 1:
        event = x.events[0]
        return _event_to_table(event)
    
    # For interactions, get all combinations
    tables = []
    for event in x.events:
        table = _event_to_table(event)
        tables.append(table)
    
    # Create full factorial combination
    # First, get all unique rows from each table
    unique_tables = []
    for table in tables:
        unique_table = table.drop_duplicates()
        unique_tables.append(unique_table)
    
    # Now create Cartesian product
    if len(unique_tables) == 2:
        # Simple case: two events
        result = _cartesian_product_2(unique_tables[0], unique_tables[1])
    else:
        # General case: multiple events
        result = unique_tables[0]
        for i in range(1, len(unique_tables)):
            result = _cartesian_product_2(result, unique_tables[i])
    
    return result


def _event_to_table(event: EventProtocol) -> pd.DataFrame:
    """Convert a single event to a table.
    
    Parameters
    ----------
    event : EventProtocol
        Event object
    
    Returns
    -------
    pd.DataFrame
        Table representation of the event
    """
    if event.event_type == "categorical":
        # For categorical, return unique levels
        levels = event.levels
        return pd.DataFrame({event.name: levels})
    
    elif event.event_type == "continuous":
        # For continuous, return unique values (or summary)
        values = np.unique(event.values)
        # If too many unique values, return summary
        if len(values) > 20:
            # Return summary statistics instead
            return pd.DataFrame({
                event.name: ["continuous"],
                f"{event.name}_min": [np.min(values)],
                f"{event.name}_max": [np.max(values)],
                f"{event.name}_mean": [np.mean(values)]
            })
        else:
            return pd.DataFrame({event.name: values})
    
    elif event.event_type == "matrix":
        # For matrix events, return column information
        n_cols = event.values.shape[1] if len(event.values.shape) > 1 else 1
        col_names = event.column_names or [f"{event.name}_{i+1}" for i in range(n_cols)]
        # Return a summary row for each column
        data = {event.name: col_names}
        return pd.DataFrame(data)
    
    elif event.event_type == "basis":
        # For basis events, return basis function information
        n_basis = getattr(event, "n_basis", getattr(event, "nbasis", None))
        if n_basis is None:
            raise AttributeError("EventBasis has no n_basis/nbasis attribute")
        basis_names = list(event.basis_names)
        if len(basis_names) != n_basis:
            basis_names = [f"basis_{i+1}" for i in range(n_basis)]
        return pd.DataFrame({event.name: basis_names})
    
    else:
        # Unknown type - return name only
        return pd.DataFrame({event.name: [f"<{event.event_type}>"]})


def _cartesian_product_2(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Compute Cartesian product of two DataFrames.
    
    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames to combine
    
    Returns
    -------
    pd.DataFrame
        Cartesian product
    """
    # Add temporary key for cross join
    df1 = df1.assign(_key=1)
    df2 = df2.assign(_key=1)
    
    # Merge on key and drop it
    result = pd.merge(df1, df2, on='_key').drop('_key', axis=1)
    
    return result


# Register for EventModel when implemented
def _register_event_model():
    """Register event_table method for EventModel."""
    from ..design.event_model import EventModel
    
    @event_table.register(EventModel)
    def _(x: EventModel, **kwargs) -> pd.DataFrame:
        """Extract event table from EventModel.
        
        Combines event tables from all terms in the model.
        """
        tables = []
        
        # Get event table for each term
        for term in x._create_event_terms():
            table = event_table(term)
            # Add term identifier
            table['_term'] = term.name or 'term'
            tables.append(table)
        
        # Combine all tables
        if tables:
            result = pd.concat(tables, ignore_index=True)
        else:
            result = pd.DataFrame()
        
        return result


# Register for individual event types
@event_table.register(EventFactor)
def _(x: EventFactor, **kwargs) -> pd.DataFrame:
    """Extract event table from EventFactor."""
    return pd.DataFrame({x.name: x.levels})


@event_table.register(EventVariable)
def _(x: EventVariable, **kwargs) -> pd.DataFrame:
    """Extract event table from EventVariable."""
    values = np.unique(x.values)
    if len(values) > 20:
        # Summary for many values
        return pd.DataFrame({
            x.name: ["continuous"],
            f"{x.name}_range": [f"[{np.min(values):.2f}, {np.max(values):.2f}]"]
        })
    return pd.DataFrame({x.name: values})


@event_table.register(EventMatrix)
def _(x: EventMatrix, **kwargs) -> pd.DataFrame:
    """Extract event table from EventMatrix."""
    return pd.DataFrame({
        'column': x.column_names,
        'type': 'matrix'
    })


@event_table.register(EventBasis)
def _(x: EventBasis, **kwargs) -> pd.DataFrame:
    """Extract event table from EventBasis."""
    n_basis = getattr(x, "n_basis", getattr(x, "nbasis", None))
    if n_basis is None:
        raise AttributeError("EventBasis has no n_basis/nbasis attribute")

    basis_names = list(x.basis_names)
    if len(basis_names) != n_basis:
        basis_names = [f"basis_{i+1}" for i in range(n_basis)]
    return pd.DataFrame({
        'basis_function': basis_names,
        'type': x.basis.name if hasattr(x.basis, 'name') else 'basis'
    })


# Re-export
__all__ = ['event_table']
