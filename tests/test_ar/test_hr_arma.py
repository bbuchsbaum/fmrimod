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


class TestPooledOrderGrid:
    """``_hr_arma_pooled`` candidate-grid BIC (fmriAR a75a5de, hr_pooled.R)."""

    @staticmethod
    def _sim(phi, theta, rng, n=240, v=30):
        from scipy.signal import lfilter

        e = rng.standard_normal((n + 200, v))
        return lfilter(np.r_[1.0, theta], np.r_[1.0, -np.asarray(phi)], e, axis=0)[200:]

    def test_selects_white_and_ar1_orders(self):
        from fmrimod.ar.hr_arma import _hr_arma_pooled

        rng = np.random.default_rng(9126)
        for phi, truth in (([], (0, 0)), ([0.5], (1, 0))):
            for _ in range(3):
                R = self._sim(phi, [], rng)
                fit = _hr_arma_pooled([(R, np.array([0]))], range(7), range(3))
                assert fit.order == truth
                assert (fit.phi.size, fit.theta.size) == truth

    def test_selection_is_the_bic_argmin_of_the_common_row_regression(self):
        """Brute-force check: each candidate refit by least squares on the
        same rows, BIC on N = rows x effective voxels."""
        from fmrimod.ar.acvf import acvf_from_pooled, pooled_acvf_segments
        from fmrimod.ar.hr_arma import (
            _arma_unit,
            _effective_voxels,
            _hr_arma_pooled,
        )
        from fmrimod.ar.numhelpers import levinson_durbin

        R = self._sim([0.5], [0.4], np.random.default_rng(3))
        fit = _hr_arma_pooled([(R, np.array([0]))], range(4), range(3))
        unit = _arma_unit(R, np.array([0]))
        Y = unit.mat
        n, V = Y.shape
        p_big = min(max(8, 3 + 2 + 5, int(np.ceil(10 * np.log10(n)))), 40)
        g, _ = acvf_from_pooled(pooled_acvf_segments(Y, np.ones(n), p_big), order=p_big)
        phi_big, _ = levinson_durbin(g, p_big)
        E = np.column_stack([_arma_innovations(Y[:, j], phi_big, np.array([]))
                             for j in range(V)])
        start = max(3, p_big + 2)
        rows = np.arange(start, n)
        n_bic = rows.size * _effective_voxels([unit])
        best = None
        for qq in range(3):
            for pp in range(4):
                cols = [Y[rows - a].ravel(order="F") for a in range(1, pp + 1)]
                cols += [E[rows - a].ravel(order="F") for a in range(1, qq + 1)]
                y = Y[rows].ravel(order="F")
                rss = float(y @ y)
                if cols:
                    Z = np.column_stack(cols)
                    resid = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
                    rss = float(resid @ resid)
                bic = n_bic * np.log(rss / (rows.size * V)) + (pp + qq) * np.log(n_bic)
                if best is None or bic < best[0]:
                    best = (bic, pp, qq)
        assert fit.order == (best[1], best[2])
