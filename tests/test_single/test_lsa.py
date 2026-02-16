"""Tests for LSA (Least Squares All) single-trial estimation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fmrimod.single.lsa import lsa_single_trial
from fmrimod.single._types import SingleTrialResult


@pytest.fixture
def rng():
    return np.random.default_rng(123)


class TestLsaSingleTrial:
    def test_basic(self, rng):
        n, T, V = 80, 10, 20
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        result = lsa_single_trial(Y, X)
        assert isinstance(result, SingleTrialResult)
        assert result.method == "lsa"
        assert result.betas.shape == (T, V)

    def test_recovery(self, rng):
        """LSA should recover betas well with low noise."""
        n, T, V = 100, 10, 20
        true_betas = rng.standard_normal((T, V))
        X = rng.standard_normal((n, T))
        Y = X @ true_betas + rng.standard_normal((n, V)) * 0.1
        result = lsa_single_trial(Y, X)
        corr = np.corrcoef(result.betas.ravel(), true_betas.ravel())[0, 1]
        assert corr > 0.95

    def test_matches_ols(self, rng):
        """LSA should give same results as direct OLS."""
        n, T, V = 80, 10, 15
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        result = lsa_single_trial(Y, X)
        betas_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
        assert_allclose(result.betas, betas_ols, atol=1e-8)

    def test_with_confounds(self, rng):
        n, T, V = 80, 10, 15
        X = rng.standard_normal((n, T))
        confounds = rng.standard_normal((n, 3))
        Y = rng.standard_normal((n, V))
        result = lsa_single_trial(Y, X, confounds=confounds)
        assert result.betas.shape == (T, V)

    def test_with_se(self, rng):
        n, T, V = 80, 10, 15
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        result = lsa_single_trial(Y, X, return_se=True)
        assert result.se is not None
        assert result.se.shape == (T, V)

    def test_trial_labels(self, rng):
        n, T, V = 50, 5, 10
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        labels = [f"trial_{i}" for i in range(T)]
        result = lsa_single_trial(Y, X, trial_labels=labels)
        assert result.trial_labels == labels
