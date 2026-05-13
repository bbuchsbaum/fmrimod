"""Regressor module for fMRI event modeling."""

from .core import Regressor, RegressorSet, null_regressor, regressor, regressor_set
from .design import regressor_design
from .neural_input import NeuralInput, neural_input

__all__ = [
    "NeuralInput",
    "Regressor",
    "RegressorSet",
    "regressor",
    "regressor_set",
    "null_regressor",
    "neural_input",
    "regressor_design",
]