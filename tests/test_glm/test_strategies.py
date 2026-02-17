"""Targeted tests for GLM strategy helpers."""

import numpy as np
import pytest

from fmrimod.glm.strategies import fit_run_ols, _pool_run_results
from fmrimod.glm.preprocess import apply_volume_weights
from fmrimod.glm.solver import fast_preproject, fast_lm_matrix, LmResult, Projection
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


def test_fit_run_ols_explicit_weights_matches_manual_weighted_solution():
    """Explicit weights path should match manual weighted solve exactly."""
    rng = np.random.default_rng(7)
    n = 80
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))]).astype(np.float64)
    beta = np.array([[1.0], [2.0]], dtype=np.float64)
    Y = (X @ beta + rng.standard_normal((n, 1)) * 0.5).astype(np.float64)
    weights = np.linspace(0.2, 1.0, n, dtype=np.float64)

    config = FmriLmConfig(
        volume_weights=VolumeWeightOptions(enabled=True, weights=weights)
    )
    result_runwise, proj_runwise, X_used, Y_used = fit_run_ols(X, Y, config)

    Xw, Yw = apply_volume_weights(X, Y, weights)
    proj_manual = fast_preproject(Xw)
    result_manual = fast_lm_matrix(Xw, Yw, proj_manual, return_fitted=True)

    np.testing.assert_allclose(X_used, Xw, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(Y_used, Yw, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(result_runwise.betas, result_manual.betas, atol=1e-12)
    np.testing.assert_allclose(result_runwise.sigma2, result_manual.sigma2, atol=1e-12)
    np.testing.assert_allclose(proj_runwise.XtXinv, proj_manual.XtXinv, atol=1e-12)


def test_fit_run_ols_weighted_bool_vs_integer_censor_parity():
    """With explicit weights, bool and 0/1 censor vectors should be equivalent."""
    rng = np.random.default_rng(21)
    n = 30
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))]).astype(np.float64)
    Y = rng.standard_normal((n, 4)).astype(np.float64)
    weights = np.linspace(0.3, 1.0, n, dtype=np.float64)

    censor_bool = np.zeros(n, dtype=bool)
    censor_bool[[2, 7, 19]] = True
    censor_int = censor_bool.astype(np.int64)

    config = FmriLmConfig(
        volume_weights=VolumeWeightOptions(enabled=True, weights=weights)
    )

    res_bool, proj_bool, X_bool, Y_bool = fit_run_ols(X, Y, config, censor_bool)
    res_int, proj_int, X_int, Y_int = fit_run_ols(X, Y, config, censor_int)

    np.testing.assert_array_equal(X_int, X_bool)
    np.testing.assert_array_equal(Y_int, Y_bool)
    np.testing.assert_allclose(res_int.betas, res_bool.betas, atol=1e-12)
    np.testing.assert_allclose(res_int.rss, res_bool.rss, atol=1e-12)
    np.testing.assert_allclose(res_int.sigma2, res_bool.sigma2, atol=1e-12)
    np.testing.assert_allclose(proj_int.XtXinv, proj_bool.XtXinv, atol=1e-12)


def test_pool_run_results_inverse_variance_weighting_parity():
    """Runwise pooling should inverse-variance weight noisy runs down."""
    low_noise = LmResult(
        betas=np.array([[1.0]], dtype=np.float64),
        rss=np.array([10.0], dtype=np.float64),
        sigma2=np.array([1.0], dtype=np.float64),
        dfres=10.0,
        rank=1,
    )
    high_noise = LmResult(
        betas=np.array([[10.0]], dtype=np.float64),
        rss=np.array([1000.0], dtype=np.float64),
        sigma2=np.array([100.0], dtype=np.float64),
        dfres=10.0,
        rank=1,
    )
    proj = Projection(
        Pinv=np.zeros((1, 1), dtype=np.float64),
        XtXinv=np.array([[1.0]], dtype=np.float64),
        dfres=10.0,
        rank=1,
        is_full_rank=True,
        ill_conditioned=False,
    )

    pooled = _pool_run_results([low_noise, high_noise], [proj, proj])
    expected = (1.0 / 1.0 * 1.0 + 1.0 / 100.0 * 10.0) / (1.0 / 1.0 + 1.0 / 100.0)

    np.testing.assert_allclose(pooled["betas"][0, 0], expected, atol=1e-12)
