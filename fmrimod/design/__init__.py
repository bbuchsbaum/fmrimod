"""Design matrix functionality for fmrimod."""

from .columns import DesignColumn, DesignColumns
from .design_matrix import design_matrix
from .diff import (
    BaselineDiff,
    ColumnsDiff,
    Composite,
    DesignDiff,
    DesignDiffPart,
    EventDiff,
    HRFDiff,
    HRFKindChange,
    HRFParameterChange,
    NoDiff,
    SamplingDiff,
    TermAdded,
    TermChanged,
    TermFieldChange,
    TermRemoved,
    design_diff,
)
from .realized import ColumnKind, DesignSource, RealizedDesign

__all__ = [
    "BaselineDiff",
    "ColumnsDiff",
    "ColumnKind",
    "Composite",
    "DesignColumn",
    "DesignColumns",
    "DesignDiff",
    "DesignDiffPart",
    "DesignSource",
    "EventDiff",
    "HRFDiff",
    "HRFKindChange",
    "HRFParameterChange",
    "NoDiff",
    "RealizedDesign",
    "SamplingDiff",
    "TermAdded",
    "TermChanged",
    "TermFieldChange",
    "TermRemoved",
    "design_diff",
    "design_matrix",
]
