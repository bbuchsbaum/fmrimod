"""Tests for the estimate_single_trial dispatcher and cross-method equivalence."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fmrimod.single import estimate_single_trial
from fmrimod.single._types import SingleTrialMethod


@pytest.fixture
def rng():
    return np.random.default_rng(88)


class TestDispatcher:
    def test_lss(self, rng):
        n, T, V = 80, 10, 15
        Y = rng.standard_normal((n, V))
        X = rng.standard_normal((n, T))
        result = estimate_single_trial(Y, X, method="lss")
        assert result.method == "lss"

    def test_top_level_single_trial_exports(self, rng):
        import fmrimod

        n, T, V = 50, 5, 4
        Y = rng.standard_normal((n, V))
        X = rng.standard_normal((n, T))

        result = fmrimod.estimate_single_trial(Y, X, method="lss")
        direct = fmrimod.lss_single_trial(Y, X)

        assert result.method == "lss"
        assert_allclose(result.betas, direct.betas)

    def test_lsa(self, rng):
        n, T, V = 80, 10, 15
        Y = rng.standard_normal((n, V))
        X = rng.standard_normal((n, T))
        result = estimate_single_trial(Y, X, method="lsa")
        assert result.method == "lsa"

    def test_oasis(self, rng):
        n, T, V = 80, 10, 15
        Y = rng.standard_normal((n, V))
        X = rng.standard_normal((n, T))
        result = estimate_single_trial(Y, X, method="oasis")
        assert result.method == "oasis"

    def test_mixed(self, rng):
        n, T, V = 80, 10, 15
        Y = rng.standard_normal((n, V))
        X = rng.standard_normal((n, T))
        result = estimate_single_trial(Y, X, method="mixed")
        assert result.method == "mixed"

    def test_unknown_raises(self, rng):
        Y = rng.standard_normal((50, 10))
        X = rng.standard_normal((50, 5))
        with pytest.raises(ValueError):
            estimate_single_trial(Y, X, method="bogus")

    def test_accepts_single_trial_method_enum(self, rng):
        Y = rng.standard_normal((50, 10))
        X = rng.standard_normal((50, 5))
        result = estimate_single_trial(Y, X, method=SingleTrialMethod.LSS)
        assert result.method is SingleTrialMethod.LSS

    def test_lss_baseline_regressors_are_dispatched(self, rng):
        n, T, V = 70, 4, 3
        Y = rng.standard_normal((n, V))
        X = rng.standard_normal((n, T))
        baseline = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])

        result = estimate_single_trial(
            Y,
            X,
            method="lss",
            baseline_regressors=baseline,
        )

        assert result.extra["adjustment_rank"] == 2

    def test_baseline_regressors_rejected_for_methods_without_adjustment_surface(self, rng):
        n, T, V = 70, 4, 3
        Y = rng.standard_normal((n, V))
        X = rng.standard_normal((n, T))

        with pytest.raises(ValueError, match="only for method='lss' or method='lsa'"):
            estimate_single_trial(
                Y,
                X,
                method="oasis",
                baseline_regressors=np.ones((n, 1)),
            )


class TestCrossMethodEquivalence:
    """LSS, LSA, and OASIS K=1 should agree on well-conditioned data."""

    def test_lss_vs_oasis_k1(self, rng):
        n, T, V = 100, 8, 15
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        res_lss = estimate_single_trial(Y, X, method="lss")
        res_oasis = estimate_single_trial(Y, X, method="oasis")
        assert_allclose(res_lss.betas, res_oasis.betas, atol=1e-6)

    def test_lsa_vs_ols(self, rng):
        """LSA should match direct OLS."""
        n, T, V = 80, 10, 15
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        res_lsa = estimate_single_trial(Y, X, method="lsa")
        betas_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
        assert_allclose(res_lsa.betas, betas_ols, atol=1e-8)


class TestPrewhitening:
    def test_prewhiten_ar1(self, rng):
        from fmrimod.single._prewhiten import PrewhitenConfig
        n, T, V = 100, 8, 15
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        pw = PrewhitenConfig(method="ar", p=1)
        result = estimate_single_trial(Y, X, method="lss", prewhiten=pw)
        assert result.betas.shape == (T, V)

    def test_prewhiten_none(self, rng):
        from fmrimod.single._prewhiten import PrewhitenConfig
        n, T, V = 80, 8, 15
        X = rng.standard_normal((n, T))
        Y = rng.standard_normal((n, V))
        pw = PrewhitenConfig(method="none")
        res_pw = estimate_single_trial(Y, X, method="lss", prewhiten=pw)
        res_no = estimate_single_trial(Y, X, method="lss")
        assert_allclose(res_pw.betas, res_no.betas, atol=1e-12)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"method": "bad"}, "method must be one of"),
            ({"pooling": "bad"}, "pooling must be one of"),
            ({"exact_first": "bad"}, "exact_first must be one of"),
            ({"p": -1}, "p must be a non-negative integer"),
            ({"q": -1}, "q must be a non-negative integer"),
            ({"p_max": 0}, "p_max must be a positive integer"),
        ],
    )
    def test_prewhiten_config_validates_options(self, kwargs, message):
        from fmrimod.single._prewhiten import PrewhitenConfig

        with pytest.raises(ValueError, match=message):
            PrewhitenConfig(**kwargs)

    def test_prewhiten_config_is_frozen_and_normalizes_ar_order(self):
        from dataclasses import FrozenInstanceError

        from fmrimod.single._prewhiten import PrewhitenConfig

        cfg = PrewhitenConfig(p=2.0)  # type: ignore[arg-type]
        assert cfg.p == 2
        with pytest.raises(FrozenInstanceError):
            cfg.method = "none"  # type: ignore[misc]


class TestSingleTrialMethod:
    def test_all_methods(self):
        assert set(m.value for m in SingleTrialMethod) == {
            "lss", "lsa", "oasis", "sbhm", "mixed", "lss_voxel_hrf"
        }
