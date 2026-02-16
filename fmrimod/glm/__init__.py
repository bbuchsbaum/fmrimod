"""GLM fitting engine for fMRI data.

Provides the core OLS/WLS solver, fitting strategies, pluggable engine
registry, and the ``fmri_lm()`` entry point for running GLMs on fMRI
datasets.
"""

from .fmri_lm import fmri_lm, FmriLm
from .solver import fast_preproject, fast_lm_matrix, Projection
from .contrasts import contrast_t, contrast_f, ContrastResult
from .engine import (
    FittingEngine,
    EngineResult,
    register_engine,
    get_engine,
    list_engines,
)

__all__ = [
    "fmri_lm",
    "FmriLm",
    "fast_preproject",
    "fast_lm_matrix",
    "Projection",
    "contrast_t",
    "contrast_f",
    "ContrastResult",
    "FittingEngine",
    "EngineResult",
    "register_engine",
    "get_engine",
    "list_engines",
]
