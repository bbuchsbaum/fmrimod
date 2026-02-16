"""Event classes for fmrimod.

This module provides event representations for fMRI design matrices:

- EventFactor: Categorical events (conditions, blocks, etc.)
- EventVariable: Continuous events (ratings, RTs, etc.)
- EventMatrix: Multi-column events (motion parameters, etc.)
- EventBasis: Basis function expanded events
"""

from .factor import EventFactor
from .variable import EventVariable
from .matrix import EventMatrix
from .basis import EventBasis
from .term import EventTerm
from .base import create_event, events_from_dataframe
from .event_table import event_table

__all__ = [
    "EventFactor",
    "EventVariable",
    "EventMatrix",
    "EventBasis",
    "EventTerm",
    "create_event",
    "events_from_dataframe",
    "event_table",
]