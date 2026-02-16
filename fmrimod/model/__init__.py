"""Model specification for fMRI GLM analysis."""

from .fmri_model import FmriModel, create_fmri_model
from .config import FmriLmConfig, RobustOptions, AROptions, VolumeWeightOptions, SoftSubspaceOptions

__all__ = [
    "FmriModel",
    "create_fmri_model",
    "FmriLmConfig",
    "RobustOptions",
    "AROptions",
    "VolumeWeightOptions",
    "SoftSubspaceOptions",
]
