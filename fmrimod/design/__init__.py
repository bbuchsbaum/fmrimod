"""Design matrix functionality for fmrimod."""

from .columns import DesignColumn, DesignColumns
from .design_matrix import design_matrix

__all__ = ["DesignColumn", "DesignColumns", "design_matrix"]
