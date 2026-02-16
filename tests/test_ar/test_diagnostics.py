"""Tests for diagnostics: acorr_diagnostics, sandwich SEs."""

import numpy as np
import pytest

from fmrimod.ar.diagnostics import acorr_diagnostics, sandwich_from_whitened_resid


class TestAcorrDiagnostics:
    def test_basic(self):
        rng = np.random.RandomState(42)
        resid = rng.randn(200, 10)
        result = acorr_diagnostics(resid, max_lag=10)
        assert "lags" in result
        assert "acf" in result
        assert "ci" in result
        assert len(result["lags"]) == 10
        assert len(result["acf"]) == 10
        assert result["ci"] > 0

    def test_white_noise_within_ci(self):
        rng = np.random.RandomState(42)
        resid = rng.randn(1000, 10)
        result = acorr_diagnostics(resid, max_lag=5)
        # Most ACF values should be within CI for white noise
        assert np.all(np.abs(result["acf"]) < 3 * result["ci"])

    def test_detects_ar1(self):
        rng = np.random.RandomState(42)
        n = 500
        resid = rng.randn(n, 5)
        for t in range(1, n):
            resid[t] += 0.5 * resid[t - 1]
        result = acorr_diagnostics(resid, max_lag=5)
        assert abs(result["acf"][0]) > result["ci"]

    def test_aggregate_none(self):
        rng = np.random.RandomState(42)
        resid = rng.randn(200, 5)
        result = acorr_diagnostics(resid, max_lag=5, aggregate="none")
        assert result["acf"].shape == (5, 5)

    def test_aggregate_median(self):
        rng = np.random.RandomState(42)
        resid = rng.randn(200, 5)
        result = acorr_diagnostics(resid, max_lag=5, aggregate="median")
        assert len(result["acf"]) == 5


class TestSandwichSEs:
    def test_iid(self):
        rng = np.random.RandomState(42)
        n, p, v = 100, 3, 5
        Xw = rng.randn(n, p)
        Yw = rng.randn(n, v)
        result = sandwich_from_whitened_resid(Xw, Yw, type="iid")
        assert result["se"].shape == (p, v)
        assert result["type"] == "iid"
        assert np.all(result["se"] > 0)
        assert np.all(result["sigma2"] > 0)

    def test_hc0(self):
        rng = np.random.RandomState(42)
        n, p, v = 100, 3, 5
        Xw = rng.randn(n, p)
        Yw = rng.randn(n, v)
        result = sandwich_from_whitened_resid(Xw, Yw, type="hc0")
        assert result["se"].shape == (p, v)
        assert result["type"] == "hc0"
        assert np.all(result["se"] > 0)

    def test_iid_matches_manual(self):
        rng = np.random.RandomState(42)
        n, p = 200, 2
        X = rng.randn(n, p)
        beta_true = np.array([[1.0], [2.0]])
        Y = X @ beta_true + rng.randn(n, 1)
        result = sandwich_from_whitened_resid(X, Y, type="iid")
        # Manual computation
        XtX_inv = np.linalg.inv(X.T @ X)
        beta_hat = XtX_inv @ (X.T @ Y)
        e = Y - X @ beta_hat
        sigma2 = np.sum(e ** 2) / (n - p)
        se_manual = np.sqrt(np.outer(np.diag(XtX_inv), sigma2))
        np.testing.assert_allclose(result["se"], se_manual, atol=1e-10)

    def test_with_beta(self):
        rng = np.random.RandomState(42)
        Xw = rng.randn(100, 2)
        Yw = rng.randn(100, 3)
        beta = np.linalg.lstsq(Xw, Yw, rcond=None)[0]
        result = sandwich_from_whitened_resid(Xw, Yw, beta=beta, type="iid")
        assert result["se"].shape == (2, 3)
