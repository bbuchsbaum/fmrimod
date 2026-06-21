"""Low-rank / sketch-based GLM methods.

Provides randomised sketching matrices, Nyström landmark extension,
and a sketch-based GLM solver for large-scale fMRI data.
"""

from .engine import LowRankConfig, fit_sketched
from .nystrom import (
    LandmarkWeights,
    build_landmark_weights,
    extend_betas,
    select_landmarks,
)
from .rrr import ReducedRankConfig, ReducedRankResult, fit_reduced_rank
from .sketch import SketchKind, make_sketch, normalize_sketch_kind, sketch_data

__all__ = [
    "SketchKind",
    "normalize_sketch_kind",
    "make_sketch",
    "sketch_data",
    "LandmarkWeights",
    "build_landmark_weights",
    "extend_betas",
    "select_landmarks",
    "LowRankConfig",
    "fit_sketched",
    "ReducedRankConfig",
    "ReducedRankResult",
    "fit_reduced_rank",
]
