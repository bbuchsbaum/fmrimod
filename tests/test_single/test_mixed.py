"""Tests for mixed model single-trial estimation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fmrimod.single.mixed import mixed_single_trial
from fmrimod.single._types import MixedExtras, SingleTrialResult


@pytest.fixture
def rng():
    return np.random.default_rng(99)


class TestMixedSingleTrial:
    def test_basic(self, rng):
        n, T, V = 100, 20, 50
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        result = mixed_single_trial(Y, X)
        assert isinstance(result, SingleTrialResult)
        assert result.method == "mixed"
        assert result.betas.shape == (T, V)

    def test_variance_components(self, rng):
        n, T, V = 100, 20, 50
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        result = mixed_single_trial(Y, X)
        assert isinstance(result.extra, MixedExtras)
        assert result.extra.sigma2_e > 0
        assert result.extra.sigma2_u >= 0
        assert result.extra.lambda_ratio >= 0

    def test_shrinkage_improves_mse(self, rng):
        """Mixed model shrinkage should have lower MSE than LSS."""
        from fmrimod.single.lss import lss_single_trial
        n, T, V = 100, 20, 50
        sigma2_u, sigma2_e = 0.3, 1.0
        true_betas = rng.standard_normal((T, V)) * np.sqrt(sigma2_u)
        X = rng.standard_normal((n, T))
        Y = X @ true_betas + rng.standard_normal((n, V)) * np.sqrt(sigma2_e)
        result_mixed = mixed_single_trial(Y, X)
        result_lss = lss_single_trial(Y, X)
        mse_mixed = np.mean((result_mixed.betas - true_betas) ** 2)
        mse_lss = np.mean((result_lss.betas - true_betas) ** 2)
        assert mse_mixed < mse_lss

    def test_with_confounds(self, rng):
        n, T, V = 100, 15, 30
        X = rng.standard_normal((n, T))
        confounds = rng.standard_normal((n, 3))
        Y = rng.standard_normal((n, V))
        result = mixed_single_trial(Y, X, confounds=confounds)
        assert result.betas.shape == (T, V)

    def test_trial_labels(self, rng):
        n, T, V = 50, 5, 10
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        labels = [f"trial_{i}" for i in range(T)]
        result = mixed_single_trial(Y, X, trial_labels=labels)
        assert result.trial_labels == labels

    def test_1d_Y(self, rng):
        n, T = 50, 5
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal(n)
        result = mixed_single_trial(Y, X)
        assert result.betas.shape == (T, 1)

    def test_size_mismatch_raises(self, rng):
        X = rng.standard_normal((100, 10))
        Y = rng.standard_normal((80, 20))
        with pytest.raises(ValueError, match="timepoints"):
            mixed_single_trial(Y, X)
