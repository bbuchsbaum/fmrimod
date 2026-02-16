"""Tests for Hannan-Rissanen ARMA estimation."""

import numpy as np
import pytest

from fmrimod.ar.hr_arma import hr_arma, _arma_innovations, _lag_matrix


class TestLagMatrix:
    def test_shape(self):
        x = np.arange(10, dtype=float)
        M = _lag_matrix(x, 3)
        assert M.shape == (10, 3)

    def test_values(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        M = _lag_matrix(x, 2)
        assert M[0, 0] == 0.0  # padded
        assert M[1, 0] == 1.0  # x[0]
        assert M[2, 0] == 2.0  # x[1]
        assert M[2, 1] == 1.0  # x[0]

    def test_zero_lags(self):
        x = np.arange(5, dtype=float)
        M = _lag_matrix(x, 0)
        assert M.shape == (5, 0)


class TestArmaInnovations:
    def test_pure_ar(self):
        rng = np.random.RandomState(42)
        n = 200
        e = rng.randn(n)
        y = np.zeros(n)
        y[0] = e[0]
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + e[t]
        innovations = _arma_innovations(y, np.array([0.5]), np.array([]))
        np.testing.assert_allclose(innovations[5:], e[5:], atol=0.5)

    def test_identity_filter(self):
        rng = np.random.RandomState(42)
        y = rng.randn(100)
        innovations = _arma_innovations(y, np.array([]), np.array([]))
        np.testing.assert_allclose(innovations, y)


class TestHRArma:
    def test_recover_ar1(self):
        rng = np.random.RandomState(42)
        n = 500
        e = rng.randn(n)
        y = np.zeros(n)
        y[0] = e[0]
        for t in range(1, n):
            y[t] = 0.6 * y[t - 1] + e[t]
        result = hr_arma(y, p=1, q=0)
        assert abs(result["phi"][0] - 0.6) < 0.15
        assert result["order"] == (1, 0)
        assert result["method"] == "hr"

    def test_recover_arma11(self):
        rng = np.random.RandomState(42)
        n = 1000
        e = rng.randn(n)
        y = np.zeros(n)
        y[0] = e[0]
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + e[t] + 0.3 * e[t - 1]
        result = hr_arma(y, p=1, q=1)
        assert abs(result["phi"][0] - 0.5) < 0.2
        assert abs(result["theta"][0] - 0.3) < 0.3

    def test_pure_ma1(self):
        rng = np.random.RandomState(42)
        n = 1000
        e = rng.randn(n)
        y = e.copy()
        for t in range(1, n):
            y[t] += 0.4 * e[t - 1]
        result = hr_arma(y, p=0, q=1)
        assert len(result["phi"]) == 0
        assert abs(result["theta"][0] - 0.4) < 0.3

    def test_short_series_raises(self):
        with pytest.raises(ValueError, match="too short"):
            hr_arma(np.array([1.0, 2.0, 3.0]), p=1, q=1)

    def test_refinement_iterations(self):
        rng = np.random.RandomState(42)
        n = 500
        e = rng.randn(n)
        y = np.zeros(n)
        y[0] = e[0]
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + e[t]
        r0 = hr_arma(y, p=1, q=0, n_iter=0)
        r2 = hr_arma(y, p=1, q=0, n_iter=2)
        assert r0["iterations"] == 0
        assert r2["iterations"] == 2

    def test_sigma2_positive(self):
        rng = np.random.RandomState(42)
        y = rng.randn(200)
        result = hr_arma(y, p=1, q=0)
        assert result["sigma2"] > 0
