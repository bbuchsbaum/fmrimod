"""Contrast specification and computation for fMRI design matrices.

This module provides tools for defining and computing statistical contrasts
for hypothesis testing in fMRI analyses.
"""

from .contrast_spec import (
    contrast,
    unit_contrast,
    pair_contrast,
    column_contrast,
    poly_contrast,
    oneway_contrast,
    interaction_contrast,
    contrast_set,
    pairwise_contrasts,
    one_against_all_contrast,
    sliding_window_contrasts,
)
from .contrast_weights import contrast_weights
from .fcontrast import Fcontrasts, plot_Fcontrasts
from .plot_contrasts import plot_contrasts
from .basis_filter import filter_basis, apply_basis_filter

__all__ = [
    # Core contrast constructors
    'contrast',
    'unit_contrast',
    'pair_contrast',
    'column_contrast',
    'poly_contrast',
    'oneway_contrast',
    'interaction_contrast',
    # Batch constructors
    'contrast_set',
    'pairwise_contrasts',
    'one_against_all_contrast',
    'sliding_window_contrasts',
    # Weight computation
    'contrast_weights',
    # F-contrasts
    'Fcontrasts',
    'plot_Fcontrasts',
    # Visualization
    'plot_contrasts',
    # Basis filtering
    'filter_basis',
    'apply_basis_filter',
]