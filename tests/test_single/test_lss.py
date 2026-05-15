"""Tests for vectorized LSS single-trial estimation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fmrimod.single.lss import _lss_beta_vec, _lss_beta_vec_with_se, lss_single_trial
from fmrimod.single._types import LssExtras, SingleTrialResult


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


def _naive_lss(
    Y: np.ndarray,
    X: np.ndarray,
    adjustment: np.ndarray | None = None,
) -> np.ndarray:
    """Reference per-trial LSS implementation using direct least squares."""
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    out = np.empty((X.shape[1], Y.shape[1]))
    for trial in range(X.shape[1]):
        pieces = [X[:, [trial]]]
        other = np.delete(X, trial, axis=1)
        if other.shape[1] > 0:
            pieces.append(other.sum(axis=1, keepdims=True))
        if adjustment is not None:
            pieces.append(adjustment)
        design = np.column_stack(pieces)
        out[trial] = np.linalg.lstsq(design, Y, rcond=None)[0][0]
    return out


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

    def test_baseline_regressors_match_naive_glm(self, rng):
        n, T, V = 90, 6, 4
        X = rng.standard_normal((n, T))
        baseline = np.column_stack(
            [
                np.ones(n),
                np.linspace(-1.0, 1.0, n),
            ]
        )
        Y = (
            X @ rng.standard_normal((T, V))
            + baseline @ rng.standard_normal((2, V))
            + 0.05 * rng.standard_normal((n, V))
        )

        result = lss_single_trial(Y, X, baseline_regressors=baseline)
        expected = _naive_lss(Y, X, adjustment=baseline)

        assert_allclose(result.betas, expected, atol=1e-10)
        assert isinstance(result.extra, LssExtras)
        assert result.extra.adjustment_rank == 2
        assert result.residual_df == n - 2 - 2

    def test_baseline_and_confounds_match_naive_glm(self, rng):
        n, T, V = 100, 5, 3
        X = rng.standard_normal((n, T))
        baseline = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
        confounds = rng.standard_normal((n, 3))
        adjustment = np.column_stack([baseline, confounds])
        Y = (
            X @ rng.standard_normal((T, V))
            + adjustment @ rng.standard_normal((adjustment.shape[1], V))
            + 0.05 * rng.standard_normal((n, V))
        )

        result = lss_single_trial(
            Y,
            X,
            baseline_regressors=baseline,
            confounds=confounds,
        )
        expected = _naive_lss(Y, X, adjustment=adjustment)

        assert_allclose(result.betas, expected, atol=1e-10)
        assert isinstance(result.extra, LssExtras)
        assert result.extra.adjustment_rank == adjustment.shape[1]

    def test_include_intercept_matches_explicit_intercept(self, rng):
        n, T, V = 60, 4, 2
        X = rng.standard_normal((n, T))
        Y = 5.0 + X @ rng.standard_normal((T, V)) + rng.standard_normal((n, V))

        implicit = lss_single_trial(Y, X, include_intercept=True)
        explicit = lss_single_trial(Y, X, baseline_regressors=np.ones((n, 1)))

        assert_allclose(implicit.betas, explicit.betas, atol=1e-12)
        assert isinstance(implicit.extra, LssExtras)
        assert implicit.extra.adjustment_rank == 1

    def test_rank_deficient_adjustment_does_not_overproject(self, rng):
        n, T, V = 70, 5, 3
        X = rng.standard_normal((n, T))
        intercept = np.ones((n, 1))
        trend = np.linspace(-1.0, 1.0, n)[:, np.newaxis]
        duplicated = np.column_stack([intercept, intercept, trend])
        unique = np.column_stack([intercept, trend])
        Y = X @ rng.standard_normal((T, V)) + unique @ rng.standard_normal((2, V))

        result_dup = lss_single_trial(Y, X, baseline_regressors=duplicated)
        result_unique = lss_single_trial(Y, X, baseline_regressors=unique)

        assert_allclose(result_dup.betas, result_unique.betas, atol=1e-10)
        assert isinstance(result_dup.extra, LssExtras)
        assert result_dup.extra.adjustment_rank == 2

    def test_zero_trial_regressor_warns_but_returns_finite(self, rng):
        n, T, V = 50, 3, 2
        X = rng.standard_normal((n, T))
        X[:, 2] = 0.0
        Y = rng.standard_normal((n, V))

        with pytest.warns(RuntimeWarning, match="Trial regressor 'bad' appears to be zero"):
            result = lss_single_trial(Y, X, trial_labels=["a", "b", "bad"])

        assert np.all(np.isfinite(result.betas))

    def test_rejects_projector_with_adjustment_regressors(self, rng):
        from fmrimod.single import build_nuisance_projector

        n, T, V = 50, 3, 2
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        projector = build_nuisance_projector(np.ones((n, 1)))

        with pytest.raises(ValueError, match="Provide either nuisance_projector"):
            lss_single_trial(
                Y,
                X,
                nuisance_projector=projector,
                include_intercept=True,
            )
