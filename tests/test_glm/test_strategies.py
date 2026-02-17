"""Targeted tests for GLM strategy helpers."""

import numpy as np
import pytest

from fmrimod.glm.strategies import fit_run_ols
from fmrimod.model.config import FmriLmConfig, SoftSubspaceOptions, VolumeWeightOptions


def test_fit_run_ols_rejects_soft_subspace_auto_mode():
    """fit_run_ols should not silently accept `soft_subspace.lam='auto'`."""
    X = np.ones((8, 2), dtype=np.float64)
    Y = np.ones((8, 3), dtype=np.float64)
    nuisance = np.ones((8, 1), dtype=np.float64)

    config = FmriLmConfig(
        soft_subspace=SoftSubspaceOptions(
            enabled=True, nuisance_matrix=nuisance, lam="auto"
        )
    )

    with pytest.raises(NotImplementedError, match="soft subspace"):
        fit_run_ols(X, Y, config)


def test_fit_run_ols_rejects_mismatched_volume_weight_length():
    """Volume weights must align with run rows to avoid opaque broadcast errors."""
    X = np.ones((8, 2), dtype=np.float64)
    Y = np.ones((8, 3), dtype=np.float64)

    config = FmriLmConfig(
        volume_weights=VolumeWeightOptions(enabled=True, weights=np.ones(7))
    )

    with pytest.raises(ValueError, match="weights length .* rows"):
        fit_run_ols(X, Y, config)
