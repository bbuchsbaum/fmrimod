"""GLM fitting engine for fMRI data.

Provides the core OLS/WLS solver, fitting strategies, pluggable engine
registry, and the ``fmri_lm()`` entry point for running GLMs on fMRI
datasets.
"""

from .combine import CombinedFmriLm, combine_contrasts, combine_runs
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
from .contrasts import ContrastResult, contrast_f, contrast_t
from .engine import (
    EngineResult,
    FittingEngine,
    get_engine,
    list_engines,
    register_engine,
)
from .fmri_lm import FmriLm, fmri_lm
from .solver import Projection, fast_lm_matrix, fast_preproject

__all__ = [
    "fmri_lm",
    "FmriLm",
    "fast_preproject",
    "fast_lm_matrix",
    "Projection",
    "contrast_t",
    "contrast_f",
    "ContrastResult",
    "combine_runs",
    "combine_contrasts",
    "CombinedFmriLm",
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
