"""Reusable fixtures for parity cases that should avoid network access."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cross_testing.fitlins_parity import make_synthetic_glm


Array = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticGlmInputs:
    """Synthetic first-level GLM arrays shared by harness canaries."""

    X: Array
    Y: Array
    contrast: Array


def synthetic_ols_inputs(
    *,
    n_timepoints: int = 160,
    n_regressors: int = 8,
    n_voxels: int = 256,
    noise_sd: float = 1.0,
    seed: int = 17,
) -> SyntheticGlmInputs:
    """Return deterministic OLS arrays for fmrimod-vs-Nilearn comparisons."""

    X, Y, _beta_true, contrast = make_synthetic_glm(
        n_timepoints=n_timepoints,
        n_regressors=n_regressors,
        n_voxels=n_voxels,
        noise_sd=noise_sd,
        seed=seed,
    )
    return SyntheticGlmInputs(X=X, Y=Y, contrast=contrast)
