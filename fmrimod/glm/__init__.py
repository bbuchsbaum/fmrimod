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
    fit_glm_from_matrix,
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
from .contrasts import (
    ContrastExplanation,
    ContrastIntent,
    ContrastResult,
    contrast_f,
    contrast_t,
)
from .engine import (
    DEFAULT_ENGINE_OPTIONS,
    ChunkwiseEngineOptions,
    EngineResult,
    FittingEngine,
    RunwiseEngineOptions,
    SketchEngineOptions,
    get_engine,
    list_engines,
    register_engine,
    resolve_engine,
)
from .fmri_lm import (
    CompleteFitProvenance,
    FitProvenance,
    FmriLm,
    FmriModelLike,
    IncompleteFitProvenanceError,
    fmri_lm,
)
from .solver import Projection, fast_lm_matrix, fast_preproject
from .spatial import SpatialContext

__all__ = [
    "fmri_lm",
    "FmriLm",
    "FitProvenance",
    "CompleteFitProvenance",
    "IncompleteFitProvenanceError",
    "fast_preproject",
    "fast_lm_matrix",
    "Projection",
    "contrast_t",
    "contrast_f",
    "ContrastExplanation",
    "ContrastIntent",
    "ContrastResult",
    "combine_runs",
    "combine_contrasts",
    "CombinedFmriLm",
    "SpatialContext",
    "SoftProjection",
    "soft_projection",
    "apply_soft_projection",
    "compute_lm_contrasts",
    "compute_lm_contrasts_from_suffstats",
    "fit_contrasts",
    "fit_glm_from_matrix",
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
    "FmriModelLike",
    "EngineResult",
    "RunwiseEngineOptions",
    "ChunkwiseEngineOptions",
    "SketchEngineOptions",
    "DEFAULT_ENGINE_OPTIONS",
    "register_engine",
    "get_engine",
    "resolve_engine",
    "list_engines",
]
