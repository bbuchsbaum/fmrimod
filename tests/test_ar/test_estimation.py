"""Tests for AR estimation: fit_noise, BIC selection, run-aware."""

import numpy as np
import pytest

from fmrimod.ar.estimation import estimate_ar_bic, fit_noise
from fmrimod.ar.plan import WhiteningPlan


def _sim_ar1(n, phi, V=10, seed=42):
    """Simulate AR(1) data."""
    rng = np.random.RandomState(seed)
    e = rng.randn(n, V)
    for t in range(1, n):
        e[t] += phi * e[t - 1]
    return e


def _sim_ar2(n, phi, V=10, seed=42):
    """Simulate AR(2) data."""
    rng = np.random.RandomState(seed)
    e = rng.randn(n, V)
    for t in range(2, n):
        e[t] += phi[0] * e[t - 1] + phi[1] * e[t - 2]
    return e


class TestEstimateARBIC:
    def test_recovers_ar1(self):
        rng = np.random.RandomState(42)
        y = rng.randn(500)
        for t in range(1, 500):
            y[t] += 0.6 * y[t - 1]
        result = estimate_ar_bic(y, p_max=4)
        assert result["order"][0] >= 1
        assert abs(result["phi"][0] - 0.6) < 0.15

    def test_selects_order_zero_for_white_noise(self):
        rng = np.random.RandomState(42)
        y = rng.randn(500)
        result = estimate_ar_bic(y, p_max=4)
        assert result["order"][0] == 0

    def test_recovers_ar2(self):
        rng = np.random.RandomState(42)
        y = rng.randn(500)
        phi_true = [0.5, -0.3]
        for t in range(2, 500):
            y[t] += phi_true[0] * y[t - 1] + phi_true[1] * y[t - 2]
        result = estimate_ar_bic(y, p_max=6)
        assert result["order"][0] >= 2
        np.testing.assert_allclose(result["phi"][:2], phi_true, atol=0.15)

    def test_short_series(self):
        y = np.array([1.0, 2.0])
        result = estimate_ar_bic(y, p_max=4)
        assert result["order"][0] == 0


class TestFitNoise:
    def test_basic_ar1(self):
        resid = _sim_ar1(200, 0.6)
        plan = fit_noise(resid=resid, method="ar", p=1)
        assert isinstance(plan, WhiteningPlan)
        assert plan.order == (1, 0)
        assert plan.method == "ar"
        assert plan.pooling == "global"
        assert len(plan.phi) == 1
        assert abs(plan.phi[0][0] - 0.6) < 0.15

    def test_auto_order_selection(self):
        resid = _sim_ar1(300, 0.6, seed=123)
        plan = fit_noise(resid=resid, method="ar", p="auto", p_max=4)
        assert plan.order[0] >= 1

    def test_from_Y_X(self):
        rng = np.random.RandomState(42)
        n, k, V = 200, 3, 10
        X = rng.randn(n, k)
        beta_true = rng.randn(k, V)
        noise = _sim_ar1(n, 0.5, V=V, seed=99)
        Y = X @ beta_true + noise
        plan = fit_noise(Y=Y, X=X, method="ar", p=1)
        assert abs(plan.phi[0][0] - 0.5) < 0.2

    def test_run_pooling(self):
        resid = _sim_ar1(200, 0.5)
        runs = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
        plan = fit_noise(resid=resid, runs=runs, method="ar", p=1, pooling="run")
        assert plan.pooling == "run"
        assert len(plan.phi) == 2

    def test_global_pooling_averages(self):
        resid = _sim_ar1(200, 0.5)
        runs = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
        plan = fit_noise(resid=resid, runs=runs, method="ar", p=1, pooling="global")
        assert plan.pooling == "global"
        assert len(plan.phi) == 1

    def test_censor_boolean(self):
        resid = _sim_ar1(200, 0.5)
        censor = np.zeros(200, dtype=bool)
        censor[10] = True
        censor[50] = True
        plan = fit_noise(resid=resid, censor=censor, method="ar", p=1)
        assert plan.censor is not None

    def test_censor_indices(self):
        resid = _sim_ar1(200, 0.5)
        censor = np.array([10, 50, 150])
        plan = fit_noise(resid=resid, censor=censor, method="ar", p=1)
        assert plan.censor is not None

    def test_exact_first(self):
        resid = _sim_ar1(200, 0.5)
        plan = fit_noise(resid=resid, method="ar", p=1, exact_first="ar1")
        assert plan.exact_first is True
        plan2 = fit_noise(resid=resid, method="ar", p=1, exact_first="none")
        assert plan2.exact_first is False

    def test_too_short_raises(self):
        resid = np.random.randn(5, 3)
        with pytest.raises(ValueError, match="too short"):
            fit_noise(resid=resid, method="ar", p=1)

    def test_invalid_method_raises(self):
        resid = np.random.randn(100, 5)
        with pytest.raises(ValueError, match="method"):
            fit_noise(resid=resid, method="invalid")

    def test_arma_method(self):
        resid = _sim_ar1(200, 0.5)
        plan = fit_noise(resid=resid, method="arma", p=1, q=1)
        assert plan.method == "arma"
        assert plan.order[1] >= 0  # q might be 0 or 1

    def test_parcel_pooling(self):
        rng = np.random.RandomState(42)
        n, V = 200, 20
        resid = _sim_ar1(n, 0.5, V=V, seed=42)
        parcels = np.array([1]*5 + [2]*5 + [3]*5 + [4]*5)
        plan = fit_noise(resid=resid, method="ar", p=1, pooling="parcel", parcels=parcels)
        assert plan.pooling == "parcel"
        assert plan.phi_by_parcel is not None
        assert len(plan.phi_by_parcel) == 4

    def test_1d_resid(self):
        rng = np.random.RandomState(42)
        y = rng.randn(200)
        for t in range(1, 200):
            y[t] += 0.5 * y[t - 1]
        plan = fit_noise(resid=y, method="ar", p=1)
        assert plan.order[0] == 1
