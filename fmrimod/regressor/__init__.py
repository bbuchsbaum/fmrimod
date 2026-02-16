"""Regressor module for fMRI event modeling."""

from .core import Regressor, RegressorSet, regressor, regressor_set, null_regressor
from .neural_input import neural_input
from .design import regressor_design

__all__ = [
    "Regressor",
    "RegressorSet",
    "regressor",
    "regressor_set",
    "null_regressor",
    "neural_input",
    "regressor_design",
]