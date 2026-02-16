"""Simulation tools for fMRI data."""

from .bold import simulate_bold
from .noise import ar_noise, white_noise

__all__ = ["simulate_bold", "ar_noise", "white_noise"]
