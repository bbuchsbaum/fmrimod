"""Tests for trial-wise beta extraction (OLS and LSS)."""

import numpy as np
import pytest

from fmrimod.betas.extraction import (
    BetaMethod,
    BetaResult,
    estimate_betas,
    estimate_betas_lss,
    estimate_betas_ols,
    _build_trial_regressors,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_trial_data(rng):
    """Create simple trial regressors and data with known betas."""
    n_time, n_trials, V = 200, 10, 20
    trial_regs = np.zeros((n_time, n_trials))
    # Non-overlapping boxcar regressors
    for i in range(n_trials):
        start = i * 15
        trial_regs[start : start + 5, i] = 1.0

    true_betas = rng.standard_normal(n_trials) * 3
    Y = trial_regs @ true_betas[:, np.newaxis] * np.ones((1, V))
    Y += rng.standard_normal((n_time, V)) * 0.5
    return trial_regs, Y, true_betas


class TestBuildTrialRegressors:
    def test_shape(self):
        onsets = np.array([0.0, 5.0, 10.0])
        durations = np.array([2.0, 2.0, 2.0])
        hrf = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        regs = _build_trial_regressors(onsets, durations, hrf, n_time=20, tr=1.0)
        assert regs.shape == (20, 3)

    def test_nonzero_after_onset(self):
        onsets = np.array([5.0])
        durations = np.array([1.0])
        hrf = np.array([0.0, 1.0, 0.5])
        regs = _build_trial_regressors(onsets, durations, hrf, n_time=20, tr=1.0)
        # Should be zero before onset
        assert np.all(regs[:5] == 0.0)
        # Should have nonzero values after onset
        assert np.any(regs[5:] != 0.0)


class TestEstimateBetasOLS:
    def test_recovery(self, simple_trial_data, rng):
        trial_regs, Y, true_betas = simple_trial_data
        result = estimate_betas_ols(trial_regs, Y)

        assert isinstance(result, BetaResult)
        assert result.method == "ols"
        assert result.betas.shape == (10, 20)

        # Mean beta across voxels should be close to true values
        mean_betas = result.betas.mean(axis=1)
        np.testing.assert_allclose(mean_betas, true_betas, atol=0.5)

    def test_with_confounds(self, simple_trial_data, rng):
        trial_regs, Y, _ = simple_trial_data
        confounds = np.column_stack([
            np.ones(200),
            np.linspace(0, 1, 200),
        ])
        result = estimate_betas_ols(trial_regs, Y, confounds=confounds)
        assert result.betas.shape == (10, 20)


class TestEstimateBetasLSS:
    def test_shape(self, simple_trial_data):
        trial_regs, Y, _ = simple_trial_data
        result = estimate_betas_lss(trial_regs, Y)

        assert isinstance(result, BetaResult)
        assert result.method == "lss"
        assert result.betas.shape == (10, 20)

    def test_recovery(self, simple_trial_data):
        trial_regs, Y, true_betas = simple_trial_data
        result = estimate_betas_lss(trial_regs, Y)

        mean_betas = result.betas.mean(axis=1)
        np.testing.assert_allclose(mean_betas, true_betas, atol=0.5)

    def test_with_confounds(self, simple_trial_data, rng):
        trial_regs, Y, _ = simple_trial_data
        confounds = np.ones((200, 1))
        result = estimate_betas_lss(trial_regs, Y, confounds=confounds)
        assert result.betas.shape == (10, 20)


class TestEstimateBetasDispatcher:
    def test_ols_dispatch(self, simple_trial_data):
        trial_regs, Y, _ = simple_trial_data
        result = estimate_betas(trial_regs, Y, method="ols")
        assert result.method == "ols"

    def test_lss_dispatch(self, simple_trial_data):
        trial_regs, Y, _ = simple_trial_data
        result = estimate_betas(trial_regs, Y, method="lss")
        assert result.method == "lss"

    def test_with_labels(self, simple_trial_data):
        trial_regs, Y, _ = simple_trial_data
        labels = [f"trial_{i}" for i in range(10)]
        result = estimate_betas(trial_regs, Y, trial_labels=labels)
        assert result.trial_labels == labels

    def test_invalid_method(self, simple_trial_data):
        trial_regs, Y, _ = simple_trial_data
        with pytest.raises(ValueError):
            estimate_betas(trial_regs, Y, method="bogus")
