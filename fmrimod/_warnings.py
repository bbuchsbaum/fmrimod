"""Shared warning suppression helpers for noisy third-party dependencies."""

from __future__ import annotations

import warnings
from contextlib import contextmanager

SCIPY_MISC_DEPRECATION = "scipy.misc is deprecated and will be removed in 2.0.0"
PYFMRIHRF_PARAM_WARNING = "Parameters .* ignored for pre-defined HRF .*"
PYARROW_DEPRECATION = "Pyarrow will become a required dependency of pandas"


@contextmanager
def suppress_fmrimod_warnings():
    """Temporarily suppress known noisy fmrimod warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=SCIPY_MISC_DEPRECATION,
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=PYFMRIHRF_PARAM_WARNING,
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=PYARROW_DEPRECATION,
            category=DeprecationWarning,
        )
        yield


def call_safely(func, *args, **kwargs):
    """Call a callable while suppressing noisy fmrimod warnings."""
    with suppress_fmrimod_warnings():
        return func(*args, **kwargs)
