"""Tests for the estimate_single_trial dispatcher and cross-method equivalence."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fmrimod.single import estimate_single_trial
from fmrimod.single._types import SingleTrialResult, SingleTrialMethod


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


class TestSingleTrialMethod:
    def test_all_methods(self):
        assert set(m.value for m in SingleTrialMethod) == {
            "lss", "lsa", "oasis", "sbhm", "mixed"
        }
