"""Tests for OASIS closed-form single-trial estimation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fmrimod.single._types import OasisConfig, SingleTrialResult
from fmrimod.single.oasis import oasis_single_trial


@pytest.fixture
def rng():
    return np.random.default_rng(77)


class TestOasisK1:
    """OASIS with K=1 (single basis function)."""

    def test_basic(self, rng):
        n, T, V = 80, 10, 20
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        config = OasisConfig(K=1)
        result = oasis_single_trial(Y, X, config=config)
        assert isinstance(result, SingleTrialResult)
        assert result.method == "oasis"
        assert result.betas.shape == (T, V)

    def test_recovery(self, rng):
        n, T, V = 100, 10, 20
        true_betas = rng.standard_normal((T, V))
        X = rng.standard_normal((n, T))
        Y = X @ true_betas + rng.standard_normal((n, V)) * 0.1
        config = OasisConfig(K=1)
        result = oasis_single_trial(Y, X, config=config)
        corr = np.corrcoef(result.betas.ravel(), true_betas.ravel())[0, 1]
        assert corr > 0.90

    def test_agrees_with_lss(self, rng):
        """OASIS K=1 should agree closely with LSS for well-conditioned data."""
        from fmrimod.single.lss import lss_single_trial
        n, T, V = 100, 8, 15
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        result_oasis = oasis_single_trial(Y, X, config=OasisConfig(K=1))
        result_lss = lss_single_trial(Y, X)
        assert_allclose(result_oasis.betas, result_lss.betas, atol=1e-6)

    def test_with_ridge(self, rng):
        n, T, V = 80, 10, 20
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        config = OasisConfig(K=1, ridge_mode="fractional", ridge_x=0.01)
        result = oasis_single_trial(Y, X, config=config)
        assert result.betas.shape == (T, V)

    def test_with_confounds(self, rng):
        n, T, V = 80, 10, 20
        X = rng.standard_normal((n, T))
        confounds = rng.standard_normal((n, 3))
        Y = rng.standard_normal((n, V))
        config = OasisConfig(K=1)
        result = oasis_single_trial(Y, X, confounds=confounds, config=config)
        assert result.betas.shape == (T, V)

    def test_with_se(self, rng):
        n, T, V = 80, 10, 20
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        config = OasisConfig(K=1, return_se=True)
        result = oasis_single_trial(Y, X, config=config)
        assert result.se is not None
        assert result.se.shape == (T, V)
        assert np.all(result.se >= 0)

    def test_trial_labels(self, rng):
        n, T, V = 50, 5, 10
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        labels = [f"t{i}" for i in range(T)]
        result = oasis_single_trial(Y, X, config=OasisConfig(K=1),
                                    trial_labels=labels)
        assert result.trial_labels == labels

    def test_small_block_cols_matches_large_block_cols(self, rng):
        n, T, V = 90, 7, 11
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))

        small = oasis_single_trial(Y, X, config=OasisConfig(K=1, block_cols=3))
        large = oasis_single_trial(Y, X, config=OasisConfig(K=1, block_cols=1000))

        assert_allclose(small.betas, large.betas, atol=1e-12)

    def test_confounds_match_residualized_lss(self, rng):
        from fmrimod.single._project import project_nuisance
        from fmrimod.single.lss import lss_single_trial

        n, T, V = 90, 7, 5
        X = rng.standard_normal((n, T))
        confounds = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
        Y = (
            X @ rng.standard_normal((T, V))
            + confounds @ rng.standard_normal((2, V))
            + rng.standard_normal((n, V)) * 0.05
        )

        result = oasis_single_trial(Y, X, confounds=confounds, config=OasisConfig(K=1))
        Y_res, X_res = project_nuisance(confounds, Y, X)
        expected = lss_single_trial(Y_res, X_res)

        assert_allclose(result.betas, expected.betas, atol=1e-8)

    def test_input_validation(self, rng):
        Y = rng.standard_normal((20, 3))
        X = rng.standard_normal((20, 4))

        with pytest.raises(ValueError, match="not divisible"):
            oasis_single_trial(Y, X[:, :3], config=OasisConfig(K=2))
        with pytest.raises(ValueError, match="finite"):
            oasis_single_trial(np.array([[np.nan], [1.0]]), X[:2, :1])
        with pytest.raises(ValueError, match="confounds has 19 rows"):
            oasis_single_trial(Y, X, confounds=np.ones((19, 1)))


class TestOasisConfig:
    def test_accepts_valid_options(self):
        cfg = OasisConfig(
            K=2,
            ridge_mode="absolute",
            ridge_x=0.1,
            ridge_b=0.2,
            block_cols=32,
        )
        assert cfg.K == 2
        assert cfg.block_cols == 32

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"K": 0}, "K must be a positive integer"),
            ({"ridge_x": -0.1}, "ridge_x must be non-negative"),
            ({"ridge_b": -0.1}, "ridge_b must be non-negative"),
            ({"ridge_mode": "bogus"}, "ridge_mode must be one of"),
            ({"block_cols": 0}, "block_cols must be a positive integer"),
        ],
    )
    def test_rejects_invalid_options(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            OasisConfig(**kwargs)


class TestOasisKn:
    """OASIS with K>1 (multi-basis)."""

    def test_basic_k2(self, rng):
        n, N, K, V = 100, 10, 2, 15
        X = rng.standard_normal((n, N * K))
        Y = rng.standard_normal((n, V))
        config = OasisConfig(K=K)
        result = oasis_single_trial(Y, X, config=config)
        # K>1 returns all basis coefficients: (N*K, V)
        assert result.betas.shape == (N * K, V)

    def test_basic_k3(self, rng):
        n, N, K, V = 100, 8, 3, 10
        X = rng.standard_normal((n, N * K))
        Y = rng.standard_normal((n, V))
        config = OasisConfig(K=K)
        result = oasis_single_trial(Y, X, config=config)
        assert result.betas.shape == (N * K, V)

    def test_with_ridge_kn(self, rng):
        n, N, K, V = 100, 10, 2, 15
        X = rng.standard_normal((n, N * K))
        Y = rng.standard_normal((n, V))
        config = OasisConfig(K=K, ridge_mode="absolute", ridge_x=0.1, ridge_b=0.05)
        result = oasis_single_trial(Y, X, config=config)
        assert result.betas.shape == (N * K, V)

    def test_with_confounds_kn(self, rng):
        n, N, K, V = 100, 10, 2, 15
        X = rng.standard_normal((n, N * K))
        confounds = rng.standard_normal((n, 4))
        Y = rng.standard_normal((n, V))
        config = OasisConfig(K=K)
        result = oasis_single_trial(Y, X, confounds=confounds, config=config)
        assert result.betas.shape == (N * K, V)
