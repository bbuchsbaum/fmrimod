"""Utility functions for fmrimod.

Merges utilities from both the HRF and design subsystems.
"""

# From HRF subsystem (formerly fmrimod)
from .misc import (
    recycle_or_error,
    list_available_hrfs,
    single_trial_regressor,
    hrf_toeplitz,
)
from .cache import cached_hrf_eval, clear_hrf_cache

# From design subsystem (formerly fmrimod)
from .term_utils import (
    term_indices,
    term_matrices,
    baseline_terms,
    split_by_term,
)

from .event_utils import (
    split_onsets,
    split_by_block,
)

from .generics import (
    blockids,
    blocklens,
    term_names,
    longnames,
    shortnames,
    cells,
    conditions,
    condition_map,
    onsets,
    durations,
    evaluate,
    acquisition_onsets,
    amplitudes,
    elements,
    labels,
    levels,
    samples,
    global_onsets,
    shift,
    columns,
    nbasis,
    is_categorical,
    is_continuous,
    event_terms,
    construct,
    events,
    event_conditions,
    contrasts,
)

__all__ = [
    # HRF utilities
    "recycle_or_error",
    "list_available_hrfs",
    "single_trial_regressor",
    "hrf_toeplitz",
    "cached_hrf_eval",
    "clear_hrf_cache",
    # Term utilities
    "term_indices",
    "term_matrices",
    "baseline_terms",
    "split_by_term",
    # Event utilities
    "split_onsets",
    "split_by_block",
    # Generic functions
    "blockids",
    "blocklens",
    "term_names",
    "longnames",
    "shortnames",
    "cells",
    "conditions",
    "condition_map",
    "onsets",
    "durations",
    "evaluate",
    "acquisition_onsets",
    "amplitudes",
    "elements",
    "labels",
    "levels",
    "samples",
    "global_onsets",
    "shift",
    "columns",
    "nbasis",
    "is_categorical",
    "is_continuous",
    "event_terms",
    "construct",
    "events",
    "event_conditions",
    "contrasts",
]
