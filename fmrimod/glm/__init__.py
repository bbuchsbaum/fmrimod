"""GLM fitting engine for fMRI data.

Provides the core OLS/WLS solver, fitting strategies, pluggable engine
registry, and the ``fmri_lm()`` entry point for running GLMs on fMRI
datasets.
"""

from .fmri_lm import fmri_lm, FmriLm
from .solver import fast_preproject, fast_lm_matrix, Projection
from .contrasts import contrast_t, contrast_f, ContrastResult
from .compat import (
    LowRankControl,
    SoftProjection,
    apply_soft_projection,
    compute_lm_contrasts,
    compute_lm_contrasts_from_suffstats,
    estimate,
    fit_contrasts,
    fit_glm_from_suffstats,
    fit_glm_on_transformed_series,
    fit_glm_with_config,
    flip_sign,
    fmri_ols_fit,
    fmri_rlm,
    hrf_smoothing_kernel,
    lowrank_control,
    paired_diff_block,
    soft_projection,
    t_to_beta_se,
)
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
    "SoftProjection",
    "soft_projection",
    "apply_soft_projection",
    "compute_lm_contrasts",
    "compute_lm_contrasts_from_suffstats",
    "fit_contrasts",
    "fit_glm_from_suffstats",
    "fit_glm_on_transformed_series",
    "fit_glm_with_config",
    "fmri_ols_fit",
    "fmri_rlm",
    "LowRankControl",
    "lowrank_control",
    "paired_diff_block",
    "flip_sign",
    "t_to_beta_se",
    "hrf_smoothing_kernel",
    "estimate",
    "FittingEngine",
    "EngineResult",
    "register_engine",
    "get_engine",
    "list_engines",
]
