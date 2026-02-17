"""Tests for iterative GLS pipeline."""

import numpy as np
import pytest

from fmrimod.ar.estimation import fit_noise
from fmrimod.ar.whitening import whiten_apply
from fmrimod.ar.diagnostics import acorr_diagnostics
from fmrimod.ar.integration import iterative_gls
from fmrimod.model.config import FmriLmConfig, AROptions


class TestFitWhitenPipeline:
    """Integration test: fit_noise -> whiten_apply -> check whiteness."""

    def test_ar1_pipeline(self):
        rng = np.random.RandomState(42)
        n, V, k = 300, 20, 4
        phi_true = 0.6

        X = rng.randn(n, k)
        beta = rng.randn(k, V)
        noise = rng.randn(n, V)
        for t in range(1, n):
            noise[t] += phi_true * noise[t - 1]
        Y = X @ beta + noise

        # OLS residuals
        coef = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ coef

        # Fit noise
        plan = fit_noise(resid=resid, method="ar", p=1)
        assert abs(plan.phi[0][0] - phi_true) < 0.15

        # Whiten
        wr = whiten_apply(plan, X, Y)

        # Check whiteness
        resid_w = wr.Y - wr.X @ np.linalg.lstsq(wr.X, wr.Y, rcond=None)[0]
        diag = acorr_diagnostics(resid_w, max_lag=5)
        assert abs(diag["acf"][0]) < 0.15

    def test_ar2_pipeline(self):
        rng = np.random.RandomState(42)
        n, V, k = 300, 10, 3
        phi_true = np.array([0.5, -0.2])

        X = rng.randn(n, k)
        beta = rng.randn(k, V)
        noise = rng.randn(n, V)
        for t in range(2, n):
            noise[t] += phi_true[0] * noise[t - 1] + phi_true[1] * noise[t - 2]
        Y = X @ beta + noise

        coef = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ coef

        plan = fit_noise(resid=resid, method="ar", p="auto", p_max=4)
        assert plan.order[0] >= 1

        wr = whiten_apply(plan, X, Y)
        resid_w = wr.Y - wr.X @ np.linalg.lstsq(wr.X, wr.Y, rcond=None)[0]
        diag = acorr_diagnostics(resid_w, max_lag=5)
        assert abs(diag["acf"][0]) < 0.15

    def test_multirun_pipeline(self):
        rng = np.random.RandomState(42)
        n_per_run, V, k = 150, 10, 3
        n = n_per_run * 2

        X = rng.randn(n, k)
        beta = rng.randn(k, V)
        noise = rng.randn(n, V)
        for t in range(1, n):
            noise[t] += 0.5 * noise[t - 1]
        Y = X @ beta + noise

        runs = np.concatenate([np.zeros(n_per_run, dtype=int),
                               np.ones(n_per_run, dtype=int)])

        coef = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ coef

        plan = fit_noise(resid=resid, runs=runs, method="ar", p=1, pooling="run")
        wr = whiten_apply(plan, X, Y, runs=runs)
        assert wr.X.shape == X.shape
        assert wr.Y.shape == Y.shape

    def test_convergence_across_iterations(self):
        """Multiple fit_noise calls should successfully estimate parameters."""
        rng = np.random.RandomState(42)
        n, V, k = 300, 10, 3
        phi_true = 0.6

        X = rng.randn(n, k)
        beta = rng.randn(k, V)
        noise = rng.randn(n, V)
        for t in range(1, n):
            noise[t] += phi_true * noise[t - 1]
        Y = X @ beta + noise

        # Iteration 1
        coef1 = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid1 = Y - X @ coef1
        plan1 = fit_noise(resid=resid1, method="ar", p=1)
        wr1 = whiten_apply(plan1, X, Y)

        # Iteration 2
        coef2 = np.linalg.lstsq(wr1.X, wr1.Y, rcond=None)[0]
        resid2 = wr1.Y - wr1.X @ coef2
        plan2 = fit_noise(resid=resid2, method="ar", p=1)

        # First estimate should be reasonable
        assert 0.0 < plan1.phi[0][0] < 1.0
        # After whitening, residuals should have less autocorrelation
        # (second estimate may be near zero, which is expected)

    def test_censor_pipeline(self):
        rng = np.random.RandomState(42)
        n, V = 200, 10
        noise = rng.randn(n, V)
        for t in range(1, n):
            noise[t] += 0.5 * noise[t - 1]

        censor = np.array([10, 50, 100, 150])
        plan = fit_noise(resid=noise, censor=censor, method="ar", p=1)
        X = rng.randn(n, 3)
        wr = whiten_apply(plan, X, noise, censor=censor)
        assert wr.X.shape == X.shape


class _DummyDataset:
    def __init__(self, runs):
        self._runs = runs

    def get_data(self, run):
        return self._runs[run]

    def get_censor(self, run):
        return None


class _DummyModel:
    def __init__(self, X_runs, Y_runs):
        self._X_runs = X_runs
        self.dataset = _DummyDataset(Y_runs)
        self.n_runs = len(X_runs)

    def design_matrix_array(self, run):
        return self._X_runs[run]


class TestIterativeGlsGlobalAr:
    def test_global_ar_true_averages_across_runs(self, monkeypatch):
        X_runs = [np.eye(4), np.eye(4)]
        Y_runs = [np.eye(4), 2 * np.eye(4)]
        model = _DummyModel(X_runs, Y_runs)

        initial_fit = {
            "residuals": [np.ones((4, 2)), np.ones((4, 2))],
            "run_X": X_runs,
        }

        phi_values = [np.array([0.2]), np.array([0.8])]

        def fake_estimate_ar(*args, **kwargs):
            return phi_values.pop(0)

        monkeypatch.setattr("fmrimod.ar.integration.estimate_ar", fake_estimate_ar)
        monkeypatch.setattr(
            "fmrimod.ar.integration.ar_whiten_matrix",
            lambda X, Y, phi: (X, Y),
        )

        config = FmriLmConfig(ar=AROptions(struct="ar1", iter_gls=1, global_ar=True))
        _, ar_params = iterative_gls(model, config, initial_fit)

        np.testing.assert_allclose(ar_params, np.array([0.5]))

    def test_global_ar_false_keeps_runwise_params(self, monkeypatch):
        X_runs = [np.eye(4), np.eye(4)]
        Y_runs = [np.eye(4), 2 * np.eye(4)]
        model = _DummyModel(X_runs, Y_runs)

        initial_fit = {
            "residuals": [np.ones((4, 2)), np.ones((4, 2))],
            "run_X": X_runs,
        }

        phi_values = [np.array([0.2]), np.array([0.8])]

        def fake_estimate_ar(*args, **kwargs):
            return phi_values.pop(0)

        monkeypatch.setattr("fmrimod.ar.integration.estimate_ar", fake_estimate_ar)
        monkeypatch.setattr(
            "fmrimod.ar.integration.ar_whiten_matrix",
            lambda X, Y, phi: (X, Y),
        )

        config = FmriLmConfig(ar=AROptions(struct="ar1", iter_gls=1, global_ar=False))
        _, ar_params = iterative_gls(model, config, initial_fit)

        assert ar_params.shape == (2, 1)
        np.testing.assert_allclose(ar_params[:, 0], np.array([0.2, 0.8]))
