"""Compatibility re-exports for fMRI series helpers."""

from __future__ import annotations

import sys

from fmrimod.dataset.series import (
    FmriSeries,
    as_matrix,
    as_tibble,
    fmri_series,
    is_fmri_series,
    new_fmri_series,
    resolve_selector,
    resolve_timepoints,
    series,
    to_dataframe,
)

__all__ = [
    "FmriSeries",
    "fmri_series",
    "new_fmri_series",
    "is_fmri_series",
    "as_matrix",
    "as_tibble",
    "to_dataframe",
    "resolve_selector",
    "resolve_timepoints",
    "series",
]

_parent = sys.modules.get(__package__)
if _parent is not None:
    setattr(_parent, "fmri_series", fmri_series)
