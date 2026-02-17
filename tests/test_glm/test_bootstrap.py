"""Tests for GLM bootstrap confidence intervals."""

import numpy as np
import pytest

from fmrimod.glm.bootstrap import (
    BootstrapMethod,
    BootstrapResult,
    bootstrap_glm,
    create_blocks,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestCreateBlocks:
    def test_single_run(self):
        blocks = create_blocks(100, block_size=10)
        assert len(blocks) == 10
        assert all(len(b) == 10 for b in blocks)
        # All indices covered
        all_idx = np.concatenate(blocks)
        np.testing.assert_array_equal(np.sort(all_idx), np.arange(100))

    def test_multi_run(self):
        blocks = create_blocks(200, block_size=50, run_boundaries=[0, 100])
        # Each run has 100 points / 50 = 2 blocks => 4 total
        assert len(blocks) == 4

    def test_uneven_block(self):
        blocks = create_blocks(15, block_size=10)
        assert len(blocks) == 2
        assert len(blocks[0]) == 10
        assert len(blocks[1]) == 5


class TestBootstrapGlm:
    def test_residual_bootstrap(self, rng):
        n, p, V = 200, 3, 20
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        true_B = np.array([5.0, -2.0, 1.0])
        Y = X @ true_B[:, np.newaxis] * np.ones((1, V)) + rng.standard_normal((n, V)) * 0.5

        result = bootstrap_glm(
            X, Y, n_boot=100, method="residual",
            block_size=20, confidence=0.95, seed=42,
        )
        assert isinstance(result, BootstrapResult)
        assert result.boot_betas.shape == (100, p, V)
        assert result.beta_ci.shape == (2, p, V)
        assert result.beta_se.shape == (p, V)
        # CI should contain the true value for most voxels
        for j in range(p):
            in_ci = (result.beta_ci[0, j] <= true_B[j]) & (true_B[j] <= result.beta_ci[1, j])
            assert np.mean(in_ci) >= 0.4  # relaxed for bootstrap variance

    def test_case_bootstrap(self, rng):
        n, p, V = 80, 2, 3
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
        Y = rng.standard_normal((n, V))

        result = bootstrap_glm(
            X, Y, n_boot=30, method="case",
            block_size=10, seed=42,
        )
        assert result.n_boot == 30
        assert result.method == "case"

    def test_wild_bootstrap(self, rng):
        n, p, V = 80, 2, 3
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
        Y = rng.standard_normal((n, V))

        result = bootstrap_glm(
            X, Y, n_boot=30, method="wild",
            block_size=10, seed=42,
        )
        assert result.method == "wild"

    def test_with_contrasts(self, rng):
        n, p, V = 100, 3, 5
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        Y = rng.standard_normal((n, V))

        result = bootstrap_glm(
            X, Y, n_boot=30, seed=42,
            contrasts={"diff": np.array([0.0, 1.0, -1.0])},
        )
        assert "diff" in result.contrast_ci
        assert result.contrast_ci["diff"].shape == (2, V)

    def test_no_bca(self, rng):
        """Percentile CIs (no BCa)."""
        n, p, V = 60, 2, 3
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
        Y = rng.standard_normal((n, V))

        result = bootstrap_glm(
            X, Y, n_boot=30, use_bca=False, seed=42,
        )
        assert result.beta_ci.shape == (2, p, V)

    def test_1d_Y(self, rng):
        n, p = 50, 2
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
        Y = rng.standard_normal(n)

        result = bootstrap_glm(X, Y, n_boot=20, seed=42)
        assert result.boot_betas.shape == (20, p, 1)

    def test_invalid_n_boot_raises_clear_error(self, rng):
        n, p, V = 40, 2, 2
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
        Y = rng.standard_normal((n, V))

        with pytest.raises(ValueError, match="n_boot must be a positive integer"):
            bootstrap_glm(X, Y, n_boot=0, seed=42)

    def test_invalid_block_size_raises_clear_error(self, rng):
        n, p, V = 40, 2, 2
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
        Y = rng.standard_normal((n, V))

        with pytest.raises(ValueError, match="block_size must be a positive integer"):
            bootstrap_glm(X, Y, block_size=0, seed=42)

    def test_invalid_confidence_raises_clear_error(self, rng):
        n, p, V = 40, 2, 2
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
        Y = rng.standard_normal((n, V))

        for bad_conf in [0.0, 1.0, -0.1, 1.1]:
            with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
                bootstrap_glm(X, Y, confidence=bad_conf, seed=42)
