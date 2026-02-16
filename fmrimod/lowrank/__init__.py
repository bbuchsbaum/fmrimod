"""Low-rank / sketch-based GLM methods.

Provides randomised sketching matrices, Nyström landmark extension,
and a sketch-based GLM solver for large-scale fMRI data.
"""

from .sketch import SketchKind, make_sketch, sketch_data
from .nystrom import (
    LandmarkWeights,
    build_landmark_weights,
    extend_betas,
    select_landmarks,
)
from .engine import LowRankConfig, fit_sketched

__all__ = [
    "SketchKind",
    "make_sketch",
    "sketch_data",
    "LandmarkWeights",
    "build_landmark_weights",
    "extend_betas",
    "select_landmarks",
    "LowRankConfig",
    "fit_sketched",
]
