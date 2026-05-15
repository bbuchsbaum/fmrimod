"""Cell and condition extraction for event terms.

Provides :func:`cells_event_term` and :func:`conditions_event_term` for
extracting the factor-level combinations ("cells") and human-readable
condition names from an ``EventTerm``, as well as model-level wrappers
:func:`cells_event_model` and :func:`conditions_event_model`.
"""

from itertools import product
from typing import Any, List, cast

import numpy as np
import pandas as pd

from ..naming import continuous_token, level_token, make_cond_tag


def cells_event_term(event_term: Any, drop_empty: bool = True) -> pd.DataFrame:
    """Extract cells (factor combinations) from an event term.
    
    For categorical events, returns all possible combinations of factor levels
    with their counts. For continuous events, returns a single row.
    
    Parameters
    ----------
    event_term : EventTerm
        Event term to extract cells from
    drop_empty : bool
        Whether to drop cells with zero count
        
    Returns
    -------
    pd.DataFrame
        DataFrame with factor levels as columns and count attribute
    """
    # Check cache
    cache_key = f"cells_{drop_empty}"
    if hasattr(event_term, '_cache') and cache_key in event_term._cache:
        return cast(pd.DataFrame, event_term._cache[cache_key])
    
    # Separate categorical and continuous events
    categorical_events = []
    continuous_events = []
    
    for event in event_term.events:
        if event.event_type == "categorical":
            categorical_events.append(event)
        else:
            continuous_events.append(event)
    
    # Handle continuous-only case
    if not categorical_events:
        # Create single row with continuous variable names
        data = {}
        for event in continuous_events:
            if event.event_type == "matrix":
                # Multiple columns
                for col_name in event.column_names:
                    data[col_name] = [""]
            else:
                # Single column
                data[event.name] = [""]
        
        result = pd.DataFrame(data)
        # Set count to number of observations
        if event_term.events:
            count = np.array([len(event_term.events[0].onsets)])
        else:
            count = np.array([0])
        result.attrs['count'] = count
        
        # Cache result
        if not hasattr(event_term, '_cache'):
            event_term._cache = {}
        event_term._cache[cache_key] = result
        
        return result
    
    # Handle categorical case
    # Get observed combinations
    observed_data = {}
    for event in categorical_events:
        # Get factor values
        if hasattr(event, 'factor_matrix'):
            # Get the category indices
            cat_indices = event.factor_matrix.argmax(axis=1)
            # Convert to factor labels
            observed_data[event.name] = pd.Categorical(
                [event.levels[i] for i in cat_indices],
                categories=event.levels
            )
        else:
            # Direct factor values
            observed_data[event.name] = pd.Categorical(
                event.values,
                categories=event.levels
            )
    
    obs_df = pd.DataFrame(observed_data)
    
    # Create grid of all possible combinations
    levels_dict = {event.name: event.levels for event in categorical_events}
    grid_data = {}
    
    # Use itertools.product for efficient combination generation
    all_combinations = list(product(*[levels_dict[name] for name in levels_dict]))
    
    for i, name in enumerate(levels_dict):
        grid_data[name] = pd.Categorical(
            [combo[i] for combo in all_combinations],
            categories=levels_dict[name]
        )
    
    grid_df = pd.DataFrame(grid_data)
    
    # Count occurrences
    # Create keys for matching
    obs_keys = obs_df.apply(lambda row: '\x01'.join(str(x) for x in row), axis=1)
    grid_keys = grid_df.apply(lambda row: '\x01'.join(str(x) for x in row), axis=1)
    
    # Count matches
    counts = np.zeros(len(grid_df), dtype=int)
    for obs_key in obs_keys:
        match_idx = grid_keys[grid_keys == obs_key].index
        if len(match_idx) > 0:
            counts[match_idx[0]] += 1
    
    # Add count as attribute
    grid_df.attrs['count'] = counts
    
    # Filter if requested
    if drop_empty:
        keep_idx = counts > 0
        result = grid_df[keep_idx].reset_index(drop=True)
        result.attrs['count'] = counts[keep_idx]
    else:
        result = grid_df
    
    # Cache result
    if not hasattr(event_term, '_cache'):
        event_term._cache = {}
    event_term._cache[cache_key] = result
    
    return result


def conditions_event_term(
    event_term: Any,
    drop_empty: bool = True,
    expand_basis: bool = False
) -> List[str]:
    """Extract condition names from an event term.
    
    Generates descriptive names for all conditions represented by the term.
    For categorical events, creates names for each level. For continuous events,
    uses column names.
    
    Parameters
    ----------
    event_term : EventTerm
        Event term to extract conditions from
    drop_empty : bool
        Whether to drop conditions with no observations (currently ignored)
    expand_basis : bool
        Whether to expand names for basis functions
        
    Returns
    -------
    List[str]
        List of condition names
    """
    # Check cache
    cache_key = f"conditions_{drop_empty}_{expand_basis}"
    if hasattr(event_term, '_cache') and cache_key in event_term._cache:
        return cast("List[str]", event_term._cache[cache_key])
    
    # Shortcut for single continuous event with one column
    if (len(event_term.events) == 1 and 
        event_term.events[0].event_type != "categorical"):
        event = event_term.events[0]
        
        if event.event_type == "variable":
            base_cond_tags = [continuous_token(event.name)]
        elif event.event_type == "matrix":
            base_cond_tags = [continuous_token(col) for col in event.column_names]
        elif event.event_type == "basis":
            # For basis events, use the event name
            base_cond_tags = [continuous_token(event.name)]
        else:
            base_cond_tags = [continuous_token(event.name)]
        
        # Handle basis expansion
        if expand_basis and hasattr(event_term, 'nbasis') and event_term.nbasis > 1:
            final_cond_tags = []
            for tag in base_cond_tags:
                for b in range(event_term.nbasis):
                    final_cond_tags.append(f"{tag}_b{b+1}")
        else:
            final_cond_tags = base_cond_tags
        
        # Cache and return
        if not hasattr(event_term, '_cache'):
            event_term._cache = {}
        event_term._cache[cache_key] = final_cond_tags
        return final_cond_tags
    
    # Generate tokens for each component
    comp_tokens_list = []
    
    for event in event_term.events:
        if event.event_type == "categorical":
            # Level tokens for categorical
            tokens = [level_token(event.name, level) for level in event.levels]
        elif event.event_type == "matrix":
            # Column names for matrix
            tokens = [continuous_token(col) for col in event.column_names]
        elif event.event_type == "basis":
            # Single token for basis (expansion handled later)
            tokens = [continuous_token(event.name)]
        else:
            # Single continuous variable
            tokens = [continuous_token(event.name)]
        
        if tokens:  # Only add non-empty token lists
            comp_tokens_list.append(tokens)
    
    # Handle empty case
    if not comp_tokens_list:
        result: List[str] = []
        if not hasattr(event_term, '_cache'):
            event_term._cache = {}
        event_term._cache[cache_key] = result
        return result
    
    # Combine tokens using product
    if len(comp_tokens_list) == 1:
        # Single component - just use its tokens
        base_cond_tags = comp_tokens_list[0]
    else:
        # Multiple components - create all combinations
        all_combinations = list(product(*comp_tokens_list))
        base_cond_tags = [make_cond_tag(combo) for combo in all_combinations]
    
    # Handle basis expansion
    if expand_basis and hasattr(event_term, 'nbasis') and event_term.nbasis > 1:
        final_cond_tags = []
        for tag in base_cond_tags:
            for b in range(event_term.nbasis):
                final_cond_tags.append(f"{tag}_b{b+1}")
    else:
        final_cond_tags = base_cond_tags
    
    # Cache and return
    if not hasattr(event_term, '_cache'):
        event_term._cache = {}
    event_term._cache[cache_key] = final_cond_tags
    
    return final_cond_tags


def cells_event_model(model: Any, drop_empty: bool = True) -> List[pd.DataFrame]:
    """Extract cells from all terms in an event model.
    
    Parameters
    ----------
    model : EventModel
        Event model to extract cells from
    drop_empty : bool
        Whether to drop empty cells
        
    Returns
    -------
    List[pd.DataFrame]
        List of cell DataFrames, one per term
    """
    event_terms = model._create_event_terms()
    return [cells_event_term(term, drop_empty) for term in event_terms]


def conditions_event_model(
    model: Any,
    drop_empty: bool = True,
    expand_basis: bool = False
) -> List[List[str]]:
    """Extract conditions from all terms in an event model.
    
    Parameters
    ----------
    model : EventModel
        Event model to extract conditions from
    drop_empty : bool
        Whether to drop empty conditions
    expand_basis : bool
        Whether to expand basis functions
        
    Returns
    -------
    List[List[str]]
        List of condition lists, one per term
    """
    event_terms = model._create_event_terms()
    return [conditions_event_term(term, drop_empty, expand_basis) 
            for term in event_terms]