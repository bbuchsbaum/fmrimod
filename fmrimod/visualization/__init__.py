"""Visualization tools for fmrimod.

Every figure draws under one design system (:mod:`.style`): colour tokens,
a matplotlib theme and shared idioms from the R plot overhaul.
"""

from .design_map import (
    correlation_map,
    design_map,
    plot_baseline_model,
    plot_design_matrix,
    plot_sampling_frame,
)
from .fit_plots import plot_fit, plot_fmri_model
from .style import cmap, colour_map, palette, set_theme, theme

__all__ = [
    "cmap",
    "colour_map",
    "correlation_map",
    "design_map",
    "palette",
    "plot_baseline_model",
    "plot_design_matrix",
    "plot_fit",
    "plot_fmri_model",
    "plot_sampling_frame",
    "set_theme",
    "theme",
]
