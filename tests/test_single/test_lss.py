"""Tests for vectorized LSS single-trial estimation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fmrimod.single.lss import _lss_beta_vec, _lss_beta_vec_with_se, lss_single_trial
from fmrimod.single._types import SingleTrialResult


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_data(rng):
    """Simple synthetic data with known betas."""
    n, n_trials, V = 100, 10, 20
    true_betas = rng.standard_normal((n_trials, V))
    X = rng.standard_normal((n, n_trials))
    noise = rng.standard_normal((n, V)) * 0.1
    Y = X @ true_betas + noise
    return Y, X, true_betas


class TestLssBetaVec:
    """Test the core vectorized LSS kernel."""

    def test_shape(self, rng):
        n, T, V = 80, 15, 30
        C = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        betas = _lss_beta_vec(C, Y)
        assert betas.shape == (T, V)

    def test_single_trial(self, rng):
        """With 1 trial, LSS == OLS."""
        n, V = 50, 10
        C = rng.standard_normal((n, 1))
        Y = rng.standard_normal((n, V))
        betas_lss = _lss_beta_vec(C, Y)
        betas_ols = np.linalg.lstsq(C, Y, rcond=None)[0]
        assert_allclose(betas_lss, betas_ols, atol=1e-10)

    def test_recovery_low_noise(self, simple_data):
        Y, X, true_betas = simple_data
        betas = _lss_beta_vec(X, Y)
        corr = np.corrcoef(betas.ravel(), true_betas.ravel())[0, 1]
        assert corr > 0.80

    def test_single_voxel(self, rng):
        n, T = 60, 8
        C = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, 1))
        betas = _lss_beta_vec(C, Y)
        assert betas.shape == (T, 1)

    def test_orthogonal_regressors(self, rng):
        """With orthogonal regressors, LSS should match OLS exactly."""
        n = 100
        T = 5
        # Build orthogonal design via QR
        Q, _ = np.linalg.qr(rng.standard_normal((n, T)))
        C = Q[:, :T]
        Y = rng.standard_normal((n, 10))
        betas_lss = _lss_beta_vec(C, Y)
        betas_ols = np.linalg.lstsq(C, Y, rcond=None)[0]
        assert_allclose(betas_lss, betas_ols, atol=1e-8)


class TestLssBetaVecWithSe:
    def test_shape(self, rng):
        n, T, V = 80, 10, 20
        C = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        betas, se, sigma2 = _lss_beta_vec_with_se(C, Y)
        assert betas.shape == (T, V)
        assert se.shape == (T, V)
        assert sigma2.shape == (T, V)

    def test_se_positive(self, rng):
        n, T, V = 80, 10, 20
        C = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        _, se, _ = _lss_beta_vec_with_se(C, Y)
        assert np.all(se >= 0)


class TestLssSingleTrial:
    def test_returns_result(self, simple_data):
        Y, X, _ = simple_data
        result = lss_single_trial(Y, X)
        assert isinstance(result, SingleTrialResult)
        assert result.method == "lss"
        assert result.betas.shape == (X.shape[1], Y.shape[1])

    def test_with_confounds(self, rng):
        n, T, V = 80, 10, 20
        X = rng.standard_normal((n, T))
        confounds = rng.standard_normal((n, 3))
        Y = rng.standard_normal((n, V))
        result = lss_single_trial(Y, X, confounds=confounds)
        assert result.betas.shape == (T, V)

    def test_with_se(self, rng):
        n, T, V = 80, 10, 20
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        result = lss_single_trial(Y, X, return_se=True)
        assert result.se is not None
        assert result.se.shape == (T, V)

    def test_trial_labels(self, rng):
        n, T, V = 50, 5, 10
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        labels = [f"trial_{i}" for i in range(T)]
        result = lss_single_trial(Y, X, trial_labels=labels)
        assert result.trial_labels == labels

    def test_1d_Y(self, rng):
        n, T = 50, 5
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal(n)
        result = lss_single_trial(Y, X)
        assert result.betas.shape[0] == T
