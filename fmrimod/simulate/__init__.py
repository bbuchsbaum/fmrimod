"""Simulation tools for fMRI data."""

from .bold import simulate_bold
from .compat import simulate_bold_signal, simulate_fmri_matrix, simulate_noise_vector
from .dataset import simulate_simple_dataset
from .noise import ar_noise, white_noise

__all__ = [
    "simulate_bold",
    "simulate_simple_dataset",
    "simulate_bold_signal",
    "simulate_noise_vector",
    "simulate_fmri_matrix",
    "ar_noise",
    "white_noise",
]
