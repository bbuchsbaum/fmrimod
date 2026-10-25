"""Visualization tools for fmrimod.

Every figure draws under one design system (:mod:`.style`): colour tokens,
a matplotlib theme and shared idioms from the R plot overhaul.
"""

from .baseline_plot import plot_baseline_model, plot_sampling_frame
from .design_stats import design_correlations, design_vif
from .event_plot import plot_design_matrix, plot_event_model
from .fit_plots import plot_fit, plot_fmri_model
from .heatmaps import correlation_map, design_map, plot_model_summary
from .style import cmap, colour_map, palette, set_theme, theme

__all__ = [
    "cmap",
    "colour_map",
    "correlation_map",
    "design_correlations",
    "design_map",
    "design_vif",
    "palette",
    "plot_baseline_model",
    "plot_design_matrix",
    "plot_event_model",
    "plot_fit",
    "plot_fmri_model",
    "plot_model_summary",
    "plot_sampling_frame",
    "set_theme",
    "theme",
]
