"""Behavioral tests for robust IRLS refitting."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.glm.solver import fast_lm_matrix, fast_preproject
from fmrimod.model.config import FmriLmConfig, RobustOptions
from fmrimod.robust.irls import robust_refit


class _DummyDataset:
    def __init__(self, runs: list[np.ndarray]) -> None:
        self._runs = [np.asarray(run, dtype=np.float64) for run in runs]

    def get_data(self, run: int) -> np.ndarray:
        return self._runs[run]


class _DummyModel:
    def __init__(self, runs: list[np.ndarray]) -> None:
        self.dataset = _DummyDataset(runs)
        self.n_runs = len(runs)


def _make_initial_fit(X: np.ndarray, Y: np.ndarray) -> dict:
    proj = fast_preproject(X)
    ols = fast_lm_matrix(X, Y, proj, return_fitted=True)
    assert ols.fitted is not None
    residuals = Y - ols.fitted
    return {
        "betas": ols.betas,
        "sigma": np.sqrt(ols.sigma2),
        "dfres": ols.dfres,
        "XtXinv": proj.XtXinv,
        "projections": [proj],
        "run_results": [ols],
        "residuals": [residuals],
        "run_X": [X],
    }


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(777)


class TestRobustRefit:
    def test_huber_downweights_outliers_and_improves_slope(
        self, rng: np.random.Generator
    ) -> None:
        n = 160
        x = rng.uniform(-1.0, 1.0, size=n)
        X = np.column_stack([np.ones(n), x])
        true_beta = np.array([0.5, 2.5], dtype=np.float64)
        Y = (X @ true_beta)[:, np.newaxis] + 0.08 * rng.standard_normal((n, 1))
        Y[:4, 0] += np.array([35.0, -30.0, 40.0, -33.0])

        initial = _make_initial_fit(X, Y)
        config = FmriLmConfig(robust=RobustOptions(type="huber", max_iter=3))
        model = _DummyModel([Y])

        fit_result, weights = robust_refit(model, config, initial)

        assert weights.shape == (n, 1)
        assert np.all(np.isfinite(weights))
        assert np.all((weights >= 0.0) & (weights <= 1.0))
        assert weights[:4, 0].mean() < 0.2
        assert weights[4:, 0].mean() > 0.8

        ols_slope_error = abs(initial["betas"][1, 0] - true_beta[1])
        robust_slope_error = abs(fit_result["betas"][1, 0] - true_beta[1])
        assert robust_slope_error < ols_slope_error

    def test_bisquare_assigns_near_zero_weight_to_extreme_outliers(
        self, rng: np.random.Generator
    ) -> None:
        n = 120
        x = rng.normal(size=n)
        X = np.column_stack([np.ones(n), x])
        Y = (X @ np.array([1.0, -1.5]))[:, np.newaxis] + 0.05 * rng.normal(size=(n, 1))
        Y[:3, 0] += 50.0

        initial = _make_initial_fit(X, Y)
        config = FmriLmConfig(robust=RobustOptions(type="bisquare", max_iter=3))
        model = _DummyModel([Y])

        _fit_result, weights = robust_refit(model, config, initial)
        assert np.max(weights[:3, 0]) < 1e-6

    def test_clean_data_stays_close_to_ols(self, rng: np.random.Generator) -> None:
        n = 150
        x = rng.uniform(-1.0, 1.0, size=n)
        X = np.column_stack([np.ones(n), x])
        Y = (X @ np.array([1.0, 2.0]))[:, np.newaxis] + 0.06 * rng.standard_normal((n, 1))

        initial = _make_initial_fit(X, Y)
        config = FmriLmConfig(robust=RobustOptions(type="huber", max_iter=3))
        model = _DummyModel([Y])

        fit_result, weights = robust_refit(model, config, initial)
        np.testing.assert_allclose(fit_result["betas"], initial["betas"], atol=0.08)
        assert weights.mean() > 0.9

    @pytest.mark.parametrize("scope", ["run", "global", "voxel"])
    def test_scale_scope_modes_return_finite_results(
        self, rng: np.random.Generator, scope: str
    ) -> None:
        n = 90
        x = rng.standard_normal(size=n)
        X = np.column_stack([np.ones(n), x])
        Y = (X @ np.array([0.2, 1.1]))[:, np.newaxis] + 0.1 * rng.standard_normal((n, 1))
        Y[:2, 0] += 25.0

        initial = _make_initial_fit(X, Y)
        config = FmriLmConfig(
            robust=RobustOptions(type="huber", max_iter=1, scale_scope=scope)
        )
        model = _DummyModel([Y])

        fit_result, weights = robust_refit(model, config, initial)
        assert np.all(np.isfinite(fit_result["betas"]))
        assert np.all(np.isfinite(weights))
        assert weights.shape == (n, 1)
