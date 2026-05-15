"""Utility functions for working with events and onsets."""

from typing import Callable, Dict, Hashable, Optional, Union

import numpy as np

from ..types import Array, EventProtocol


def split_onsets(event: EventProtocol, 
                 by: Union[str, Callable[[object], object]],
                 values: Optional[Array] = None) -> Dict[Hashable, Array]:
    """Split event onsets by a grouping criterion.
    
    Splits the onsets of an event into groups based on a
    categorical variable or custom function. Useful for
    condition-specific analysis or block extraction.
    
    Parameters
    ----------
    event : EventProtocol
        Event object with onsets
    by : str or callable
        If str, name of attribute to group by (e.g., 'values' for conditions).
        If callable, function that takes event values and returns group labels.
    values : array-like, optional
        Values to use for grouping. If None, uses event.values.
    
    Returns
    -------
    Dict[Hashable, Array]
        Dictionary mapping group labels to onset arrays
    
    Examples
    --------
    >>> # Split by event conditions
    >>> face_event = EventFactor('face_type', onsets, values=['happy', 'sad', 'neutral'])
    >>> onset_groups = split_onsets(face_event, by='values')
    >>> onset_groups.keys()
    dict_keys(['happy', 'sad', 'neutral'])
    >>> 
    >>> # Split by custom criterion
    >>> def early_late(onsets):
    ...     midpoint = np.median(onsets)
    ...     return ['early' if o < midpoint else 'late' for o in onsets]
    >>> 
    >>> timing_groups = split_onsets(event, by=early_late)
    >>> 
    >>> # Split by external variable
    >>> blocks = ['A', 'A', 'B', 'B', 'A', 'B']
    >>> block_onsets = split_onsets(event, by=lambda x: blocks, values=blocks)
    """
    # Get onsets
    onsets = np.asarray(event.onsets)
    
    # Determine grouping values
    if isinstance(by, str):
        if by == 'values':
            if values is None:
                if hasattr(event, 'values'):
                    values = event.values
                else:
                    raise ValueError("Event has no 'values' attribute")
            group_labels = values
        else:
            # Try to get attribute
            if hasattr(event, by):
                group_labels = getattr(event, by)
            else:
                raise AttributeError(f"Event has no attribute '{by}'")
    elif callable(by):
        # Apply function to get grouping
        if values is None:
            if hasattr(event, 'values'):
                values = event.values
            else:
                values = onsets
        group_labels = by(values)
    else:
        raise TypeError("'by' must be a string or callable")
    
    # Ensure same length
    group_labels = np.asarray(group_labels)
    if len(group_labels) != len(onsets):
        raise ValueError(
            f"Length mismatch: {len(group_labels)} group labels "
            f"but {len(onsets)} onsets"
        )
    
    # Group onsets
    groups = {}
    for label in np.unique(group_labels):
        mask = group_labels == label
        groups[label] = onsets[mask]
    
    return groups


def split_by_block(event: EventProtocol,
                   block_var: Union[str, Array],
                   block_onsets: Optional[Array] = None,
                   block_durations: Optional[Array] = None) -> Dict[Hashable, EventProtocol]:
    """Split an event by experimental blocks.
    
    Divides an event into separate events for each block,
    preserving the event structure but filtering onsets/values
    to each block period.
    
    Parameters
    ----------
    event : EventProtocol
        Event to split
    block_var : str or array-like
        If str, name of block variable in event.
        If array, block labels for each event.
    block_onsets : array-like, optional
        Start times of blocks. If provided with block_durations,
        events are assigned to blocks based on timing.
    block_durations : array-like, optional
        Duration of each block.
    
    Returns
    -------
    Dict[Hashable, EventProtocol]
        Dictionary mapping block labels to event objects
    
    Examples
    --------
    >>> # Split by block labels
    >>> event = EventFactor('condition', onsets, values, block=['A', 'A', 'B', 'B'])
    >>> blocks = split_by_block(event, 'block')
    >>> blocks['A'].onsets
    array([1.0, 2.5])
    >>> 
    >>> # Split by block timing
    >>> block_starts = [0, 100, 200]
    >>> block_lengths = [100, 100, 100]
    >>> blocks = split_by_block(event, ['block1', 'block2', 'block3'],
    ...                        block_onsets=block_starts,
    ...                        block_durations=block_lengths)
    """
    from ..events import EventBasis, EventFactor, EventMatrix, EventVariable
    
    # Get event attributes
    onsets = np.asarray(event.onsets)
    durations = np.asarray(event.durations) if hasattr(event, 'durations') else None
    values = event.values if hasattr(event, 'values') else None
    
    # Determine block assignment
    if isinstance(block_var, str):
        # Get from event attribute
        if hasattr(event, block_var):
            block_labels = np.asarray(getattr(event, block_var))
        else:
            raise AttributeError(f"Event has no attribute '{block_var}'")
    else:
        block_labels = np.asarray(block_var)
    
    # If block timing provided, assign events to blocks
    if block_onsets is not None and block_durations is not None:
        block_onsets = np.asarray(block_onsets)
        block_durations = np.asarray(block_durations)
        
        if len(block_labels) != len(block_onsets):
            raise ValueError(
                f"Length mismatch: {len(block_labels)} block labels "
                f"but {len(block_onsets)} block onsets"
            )
        
        # Assign each onset to a block
        assigned_blocks = []
        for onset in onsets:
            # Find which block this onset belongs to
            block_idx = None
            for i, (start, dur) in enumerate(zip(block_onsets, block_durations)):
                if start <= onset < start + dur:
                    block_idx = i
                    break
            
            if block_idx is not None:
                assigned_blocks.append(block_labels[block_idx])
            else:
                # Onset outside any block
                assigned_blocks.append(None)
        
        block_labels = np.array(assigned_blocks)
    else:
        # Direct block labels
        if len(block_labels) != len(onsets):
            raise ValueError(
                f"Length mismatch: {len(block_labels)} block labels "
                f"but {len(onsets)} onsets"
            )
    
    # Create event for each block
    blocks = {}
    unique_blocks = [b for b in np.unique(block_labels) if b is not None]
    
    for block in unique_blocks:
        mask = block_labels == block
        block_onsets = onsets[mask]
        
        # Skip empty blocks
        if len(block_onsets) == 0:
            continue
        
        # Create appropriate event type
        if isinstance(event, EventFactor):
            block_values = values[mask] if values is not None else None
            block_durations = durations[mask] if durations is not None else None
            
            blocks[block] = EventFactor(
                name=f"{event.name}_{block}",
                onsets=block_onsets,
                values=block_values,
                durations=block_durations
            )
            
        elif isinstance(event, EventVariable):
            block_values = values[mask] if values is not None else None
            block_durations = durations[mask] if durations is not None else None
            
            blocks[block] = EventVariable(
                name=f"{event.name}_{block}",
                onsets=block_onsets,
                values=block_values,
                durations=block_durations
            )
            
        elif isinstance(event, EventMatrix):
            block_values = values[mask] if values is not None else None
            block_durations = durations[mask] if durations is not None else None
            
            blocks[block] = EventMatrix(
                name=f"{event.name}_{block}",
                onsets=block_onsets,
                values=block_values,
                durations=block_durations,
                column_names=(
                    event.column_names if hasattr(event, "column_names") else None
                ),
            )
            
        elif isinstance(event, EventBasis):
            block_values = values[mask] if values is not None else None
            block_durations = durations[mask] if durations is not None else None
            
            blocks[block] = EventBasis(
                name=f"{event.name}_{block}",
                onsets=block_onsets,
                values=block_values,
                basis=event.basis,
                durations=block_durations
            )
        else:
            # Generic event copy
            block_event = type(event)(
                name=f"{event.name}_{block}",
                onsets=block_onsets
            )
            # Copy other attributes
            if values is not None:
                block_event.values = values[mask]
            if durations is not None:
                block_event.durations = durations[mask]
            
            blocks[block] = block_event
    
    return blocks
