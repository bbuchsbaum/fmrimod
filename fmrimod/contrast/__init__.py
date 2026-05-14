"""Contrast specification and computation for fMRI design matrices.

This module provides tools for defining and computing statistical contrasts
for hypothesis testing in fMRI analyses.
"""

from .basis_filter import apply_basis_filter, filter_basis
from .contrast_spec import (
    column_contrast,
    contrast,
    contrast_set,
    interaction_contrast,
    one_against_all_contrast,
    oneway_contrast,
    pair_contrast,
    pairwise_contrasts,
    poly_contrast,
    sliding_window_contrasts,
    unit_contrast,
)
from .contrast_weights import contrast_weights
from .errors import DesignProvenanceError
from .factorial import (
    generate_interaction_contrast,
    generate_main_effect_contrast,
)
from .fcontrast import Fcontrasts, plot_Fcontrasts
from .omnibus import OmnibusContrast
from .plot_contrasts import plot_contrasts
from .semantic import ConditionRef, SemanticContrast, condition

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
    'generate_main_effect_contrast',
    'generate_interaction_contrast',
    # Weight computation
    'contrast_weights',
    # F-contrasts
    'Fcontrasts',
    'plot_Fcontrasts',
    # Typed contrast intent
    'OmnibusContrast',
    'ConditionRef',
    'SemanticContrast',
    'condition',
    'DesignProvenanceError',
    # Visualization
    'plot_contrasts',
    # Basis filtering
    'filter_basis',
    'apply_basis_filter',
]
