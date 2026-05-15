"""Base event class and utilities."""

from typing import Dict, Union

from .basis import EventBasis
from .factor import EventFactor
from .matrix import EventMatrix
from .variable import EventVariable

# Event type mapping for construction
EVENT_TYPES = {
    'factor': EventFactor,
    'categorical': EventFactor,
    'variable': EventVariable,
    'continuous': EventVariable,
    'matrix': EventMatrix,
    'basis': EventBasis,
}


def create_event(event_type: str, **kwargs: object) -> Union[EventFactor, EventVariable, EventMatrix, EventBasis]:
    """Create an event of the specified type.
    
    Parameters
    ----------
    event_type : str
        Type of event: 'factor', 'variable', 'matrix', or 'basis'
    **kwargs
        Arguments passed to the event constructor
    
    Returns
    -------
    Event
        Event instance of the specified type
    
    Examples
    --------
    >>> event = create_event('factor', 
    ...                      name='condition',
    ...                      onsets=[1, 2, 3],
    ...                      values=['A', 'B', 'A'])
    """
    if event_type not in EVENT_TYPES:
        raise ValueError(
            f"Unknown event type '{event_type}'. "
            f"Must be one of: {list(EVENT_TYPES.keys())}"
        )
    
    event_class = EVENT_TYPES[event_type]
    return event_class(**kwargs)


def events_from_dataframe(
    df,
    event_specs: Dict[str, Dict],
    onset_col: str = 'onset',
    duration_col: str = 'duration'
) -> Dict[str, Union[EventFactor, EventVariable, EventMatrix, EventBasis]]:
    """Create multiple events from a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing event data
    event_specs : dict
        Specification for each event. Keys are event names, values are
        dicts with 'type' and other parameters.
    onset_col : str
        Column name for onsets
    duration_col : str
        Column name for durations
    
    Returns
    -------
    dict
        Dictionary mapping event names to event objects
    
    Examples
    --------
    >>> specs = {
    ...     'condition': {'type': 'factor'},
    ...     'rating': {'type': 'variable', 'center': True},
    ...     'motion': {'type': 'matrix', 'value_cols': ['x', 'y', 'z']}
    ... }
    >>> events = events_from_dataframe(df, specs)
    """
    events = {}
    
    for name, spec in event_specs.items():
        spec = spec.copy()
        event_type = spec.pop('type')
        
        # Get appropriate constructor
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown event type '{event_type}' for '{name}'")
        
        event_class = EVENT_TYPES[event_type]
        
        # Create event from dataframe
        events[name] = event_class.from_dataframe(
            df,
            name=name,
            onset_col=onset_col,
            duration_col=duration_col,
            **spec
        )
    
    return events