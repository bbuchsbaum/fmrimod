"""Design matrix functionality for fmrimod."""

from .columns import DesignColumn, DesignColumns
from .design_matrix import design_matrix
from .realized import ColumnKind, DesignSource, RealizedDesign

__all__ = [
    "ColumnKind",
    "DesignColumn",
    "DesignColumns",
    "DesignSource",
    "RealizedDesign",
    "design_matrix",
]
