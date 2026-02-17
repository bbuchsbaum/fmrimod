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


def test_fit_run_ols_rejects_mismatched_length_after_censoring():
    """Censoring should not mask volume-weight length mismatches."""
    X = np.ones((8, 2), dtype=np.float64)
    Y = np.ones((8, 3), dtype=np.float64)
    censor = np.array([False, False, True, False, False, False, False, False], dtype=bool)

    config = FmriLmConfig(
        volume_weights=VolumeWeightOptions(enabled=True, weights=np.ones(5))
    )

    with pytest.raises(ValueError, match="weights length .* rows"):
        fit_run_ols(X, Y, config, censor)


def test_fit_run_ols_accepts_binary_integer_censor_parity():
    """Binary 0/1 censor vectors should behave like boolean masks (fmrireg parity)."""
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(8), rng.standard_normal((8, 2))]).astype(np.float64)
    Y = rng.standard_normal((8, 3)).astype(np.float64)
    censor_bool = np.array([False, True, False, False, True, False, False, False], dtype=bool)
    censor_int = censor_bool.astype(np.int64)
    config = FmriLmConfig()

    result_bool, proj_bool, X_bool, Y_bool = fit_run_ols(X, Y, config, censor_bool)
    result_int, proj_int, X_int, Y_int = fit_run_ols(X, Y, config, censor_int)

    np.testing.assert_array_equal(X_int, X_bool)
    np.testing.assert_array_equal(Y_int, Y_bool)
    np.testing.assert_allclose(result_int.betas, result_bool.betas, atol=1e-12)
    np.testing.assert_allclose(result_int.rss, result_bool.rss, atol=1e-12)
    assert proj_int.rank == proj_bool.rank


def test_fit_run_ols_rejects_nonbinary_numeric_censor():
    """Numeric censor vectors must be binary when used as mask-style inputs."""
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(6), rng.standard_normal((6, 1))]).astype(np.float64)
    Y = rng.standard_normal((6, 2)).astype(np.float64)
    censor_bad = np.array([0, 1, 2, 0, 1, 0], dtype=np.int64)

    with pytest.raises(ValueError, match="Censor vector must be boolean or binary"):
        fit_run_ols(X, Y, FmriLmConfig(), censor_bad)
