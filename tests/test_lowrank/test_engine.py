"""Tests for the sketch-based GLM engine."""

import numpy as np
import pytest

from fmrimod.glm.solver import fast_preproject, fast_lm_matrix
from fmrimod.lowrank.engine import LowRankConfig, fit_sketched


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestFitSketched:
    def test_beta_recovery_gaussian(self, rng):
        """Sketched OLS should approximately recover true betas."""
        n, p, V = 500, 4, 30
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
        true_B = np.array([10.0, 3.0, -1.5, 0.5])
        Y = X @ true_B[:, np.newaxis] * np.ones((1, V)) + rng.standard_normal((n, V)) * 0.5

        config = LowRankConfig(sketch_kind="gaussian", sketch_ratio=0.6, seed=42)
        result = fit_sketched(X, Y, config)

        assert result.betas.shape == (p, V)
        for j in range(p):
            np.testing.assert_allclose(
                result.betas[j].mean(), true_B[j], atol=0.5,
                err_msg=f"Beta {j} recovery failed",
            )

    def test_beta_recovery_countsketch(self, rng):
        n, p, V = 500, 3, 10
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        true_B = np.array([5.0, -2.0, 1.0])
        Y = X @ true_B[:, np.newaxis] * np.ones((1, V)) + rng.standard_normal((n, V)) * 0.5

        config = LowRankConfig(sketch_kind="countsketch", sketch_ratio=0.7, seed=42)
        result = fit_sketched(X, Y, config)

        for j in range(p):
            np.testing.assert_allclose(
                result.betas[j].mean(), true_B[j], atol=0.5,
            )

    def test_no_sketch_when_ratio_1(self, rng):
        """sketch_ratio=1.0 should give exact OLS."""
        n, p, V = 100, 3, 5
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        true_B = np.array([[1.0], [2.0], [-1.0]]) * np.ones((1, V))
        Y = X @ true_B

        config = LowRankConfig(sketch_ratio=1.0, seed=42)
        result = fit_sketched(X, Y, config)
        np.testing.assert_allclose(result.betas, true_B, atol=1e-8)

    def test_ratio_one_matches_solver_outputs(self, rng):
        """Differential check: ratio=1 should equal the baseline OLS solver."""
        n, p, V = 137, 5, 17
        X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
        Y = rng.standard_normal((n, V))

        config = LowRankConfig(sketch_ratio=1.0, seed=17)
        got = fit_sketched(X, Y, config)

        proj = fast_preproject(X)
        ref = fast_lm_matrix(X, Y, proj)

        np.testing.assert_allclose(got.betas, ref.betas, atol=1e-12)
        np.testing.assert_allclose(got.rss, ref.rss, atol=1e-12)
        np.testing.assert_allclose(got.sigma2, ref.sigma2, atol=1e-12)
        assert got.dfres == ref.dfres
        assert got.rank == ref.rank

    def test_ridge_penalty(self, rng):
        n, p, V = 100, 3, 5
        X = rng.standard_normal((n, p))
        Y = rng.standard_normal((n, V))

        config = LowRankConfig(ridge=1.0, sketch_ratio=1.0, seed=42)
        result = fit_sketched(X, Y, config)
        # Ridge should shrink betas towards zero
        config_no_ridge = LowRankConfig(ridge=0.0, sketch_ratio=1.0, seed=42)
        result_no_ridge = fit_sketched(X, Y, config_no_ridge)
        assert np.linalg.norm(result.betas) <= np.linalg.norm(result_no_ridge.betas) + 0.1

    def test_with_landmarks(self, rng):
        """Landmark extension should still recover betas approximately."""
        n, p, V = 200, 3, 50
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        true_B = np.array([5.0, -1.0, 2.0])
        Y = X @ true_B[:, np.newaxis] * np.ones((1, V)) + rng.standard_normal((n, V)) * 0.5

        coords = rng.standard_normal((V, 3))
        config = LowRankConfig(
            use_landmarks=True,
            n_landmarks=20,
            landmark_k=5,
            sketch_ratio=1.0,
            seed=42,
        )
        result = fit_sketched(X, Y, config, coords=coords)
        assert result.betas.shape == (p, V)

        for j in range(p):
            np.testing.assert_allclose(
                result.betas[j].mean(), true_B[j], atol=0.5,
            )

    def test_sketch_seed_controls_reproducibility(self, rng):
        n, p, V = 180, 4, 25
        X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
        true_B = np.array([2.0, -1.0, 0.5, 3.0])
        Y = X @ true_B[:, np.newaxis] + 0.3 * rng.standard_normal((n, V))

        cfg_a = LowRankConfig(sketch_kind="gaussian", sketch_ratio=0.45, seed=123)
        cfg_b = LowRankConfig(sketch_kind="gaussian", sketch_ratio=0.45, seed=123)
        cfg_c = LowRankConfig(sketch_kind="gaussian", sketch_ratio=0.45, seed=124)

        fit_a = fit_sketched(X, Y, cfg_a)
        fit_b = fit_sketched(X, Y, cfg_b)
        fit_c = fit_sketched(X, Y, cfg_c)

        np.testing.assert_allclose(fit_a.betas, fit_b.betas, atol=0.0, rtol=0.0)
        assert np.mean(np.abs(fit_a.betas - fit_c.betas)) > 1e-4
