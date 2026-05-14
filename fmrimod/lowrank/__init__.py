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
from .sketch import SketchKind, make_sketch, sketch_data

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
