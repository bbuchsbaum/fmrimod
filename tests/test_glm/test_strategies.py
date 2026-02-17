"""Targeted tests for GLM strategy helpers."""

import numpy as np
import pytest

from fmrimod.glm.strategies import fit_run_ols
from fmrimod.model.config import FmriLmConfig, SoftSubspaceOptions


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
