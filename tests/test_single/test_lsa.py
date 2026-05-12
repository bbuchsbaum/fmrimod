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

    def test_1d_Y(self, rng):
        n, T = 50, 5
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal(n)
        result = lsa_single_trial(Y, X)
        assert result.betas.shape == (T, 1)

    def test_baseline_regressors_match_direct_ols(self, rng):
        n, T, V = 80, 6, 4
        X = rng.standard_normal((n, T))
        baseline = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
        beta = rng.standard_normal((T, V))
        beta_base = rng.standard_normal((baseline.shape[1], V))
        Y = X @ beta + baseline @ beta_base + 0.05 * rng.standard_normal((n, V))

        result = lsa_single_trial(Y, X, baseline_regressors=baseline)
        expected = np.linalg.lstsq(np.column_stack([X, baseline]), Y, rcond=None)[0][
            :T
        ]

        assert_allclose(result.betas, expected, atol=1e-10)

    def test_include_intercept_matches_explicit_intercept(self, rng):
        n, T, V = 70, 5, 3
        X = rng.standard_normal((n, T))
        Y = 2.0 + X @ rng.standard_normal((T, V)) + rng.standard_normal((n, V))

        implicit = lsa_single_trial(Y, X, include_intercept=True)
        explicit = lsa_single_trial(Y, X, baseline_regressors=np.ones((n, 1)))

        assert_allclose(implicit.betas, explicit.betas, atol=1e-12)

    def test_rejects_bad_baseline_rows(self, rng):
        n, T, V = 50, 4, 2
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        with pytest.raises(ValueError, match="baseline_regressors has 49 rows"):
            lsa_single_trial(Y, X, baseline_regressors=np.ones((n - 1, 1)))
