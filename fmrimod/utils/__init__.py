"""Utility functions for fmrimod.

Merges utilities from both the HRF and design subsystems.
"""

# From HRF subsystem (formerly fmrimod)
from .cache import cached_hrf_eval, clear_hrf_cache
from .event_utils import (
    split_by_block,
    split_onsets,
)
from .generics import (
    acquisition_onsets,
    amplitudes,
    blockids,
    blocklens,
    cells,
    columns,
    condition_map,
    conditions,
    construct,
    contrasts,
    durations,
    elements,
    evaluate,
    event_conditions,
    event_terms,
    events,
    global_onsets,
    is_categorical,
    is_continuous,
    labels,
    levels,
    longnames,
    nbasis,
    onsets,
    samples,
    shift,
    shortnames,
    term_names,
)
from .misc import (
    hrf_toeplitz,
    list_available_hrfs,
    recycle_or_error,
    single_trial_regressor,
)

# From design subsystem (formerly fmrimod)
from .term_utils import (
    baseline_terms,
    split_by_term,
    term_indices,
    term_matrices,
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
