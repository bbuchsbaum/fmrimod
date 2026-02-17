"""Tests for GLM bootstrap confidence intervals."""

import numpy as np
import pytest

from fmrimod.glm.bootstrap import (
    BootstrapMethod,
    BootstrapResult,
    _resample_case,
    _resample_residual,
    _sample_block_indices,
    bootstrap_glm,
    create_blocks,
)
from fmrimod.glm.solver import fast_preproject, fast_lm_matrix


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

    def test_run_boundaries_must_start_at_zero(self):
        with pytest.raises(ValueError, match="run_boundaries must start at 0"):
            create_blocks(20, block_size=5, run_boundaries=[5, 10])

    def test_run_boundaries_must_be_strictly_increasing(self):
        with pytest.raises(ValueError, match="run_boundaries must be strictly increasing"):
            create_blocks(20, block_size=5, run_boundaries=[0, 10, 8])

    def test_run_boundaries_must_be_within_range(self):
        with pytest.raises(ValueError, match="run_boundaries entries must be in \\[0, n\\)"):
            create_blocks(20, block_size=5, run_boundaries=[0, 25])


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

    def test_default_bootstrap_ci_matches_percentile_parity(self, rng):
        """Parity: default CIs should be percentile, not BCa."""
        n, p, V = 40, 2, 2
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
        Y = rng.standard_normal((n, V))

        res_default = bootstrap_glm(X, Y, n_boot=25, seed=0)
        res_percentile = bootstrap_glm(X, Y, n_boot=25, use_bca=False, seed=0)

        np.testing.assert_allclose(res_default.beta_ci, res_percentile.beta_ci, atol=0.0, rtol=0.0)

    def test_sample_block_indices_always_returns_n(self, rng):
        blocks = create_blocks(5, block_size=4)  # lengths [4, 1]
        for _ in range(50):
            idx = _sample_block_indices(blocks, 5, rng)
            assert idx.shape == (5,)
            assert np.all((0 <= idx) & (idx < 5))

    def test_resample_residual_handles_uneven_blocks(self, rng):
        n, p, V = 5, 2, 1
        X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
        Y = rng.standard_normal((n, V))
        proj = fast_preproject(X)
        fit = fast_lm_matrix(X, Y, proj, return_fitted=True)
        fitted = fit.fitted
        residuals = Y - fitted
        blocks = create_blocks(n, block_size=4)  # uneven blocks [4, 1]

        for _ in range(20):
            X_star, Y_star = _resample_residual(X, fitted, residuals, blocks, rng)
            assert X_star.shape[0] == n
            assert Y_star.shape[0] == n

    def test_resample_case_handles_uneven_blocks(self, rng):
        n, p, V = 5, 2, 1
        X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
        Y = rng.standard_normal((n, V))
        blocks = create_blocks(n, block_size=4)  # uneven blocks [4, 1]

        for _ in range(20):
            X_star, Y_star = _resample_case(X, Y, blocks, rng)
            assert X_star.shape[0] == n
            assert Y_star.shape[0] == n

    def test_residual_bootstrap_handles_uneven_blocks_without_shape_errors(self, rng):
        """Residual block resampling should always reconstruct exactly n rows."""
        n, p, V = 5, 2, 2
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
        Y = rng.standard_normal((n, V))

        result = bootstrap_glm(
            X, Y, n_boot=5, method="residual", block_size=4, use_bca=False, seed=0
        )
        assert result.boot_betas.shape == (5, p, V)

    def test_case_bootstrap_handles_uneven_blocks_without_shape_errors(self, rng):
        """Case block resampling should also return exactly n rows."""
        n, p, V = 5, 2, 2
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
        Y = rng.standard_normal((n, V))

        result = bootstrap_glm(
            X, Y, n_boot=5, method="case", block_size=4, use_bca=False, seed=0
        )
        assert result.boot_betas.shape == (5, p, V)

    def test_default_bootstrap_ci_matches_percentile_mode(self, rng):
        """Parity: default CIs should match explicit percentile mode (use_bca=False)."""
        n, p, V = 40, 2, 2
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
        Y = rng.standard_normal((n, V))

        default = bootstrap_glm(X, Y, n_boot=20, seed=123)
        explicit = bootstrap_glm(X, Y, n_boot=20, use_bca=False, seed=123)
        np.testing.assert_allclose(default.beta_ci, explicit.beta_ci, atol=1e-12)

    def test_run_indices_matches_run_boundaries(self, rng):
        """fmrireg-style run_indices should match run_boundaries behavior."""
        n, p, V = 12, 2, 2
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
        Y = rng.standard_normal((n, V))

        by_boundaries = bootstrap_glm(
            X,
            Y,
            n_boot=20,
            block_size=3,
            run_boundaries=[0, 6],
            use_bca=False,
            seed=7,
        )
        by_run_indices = bootstrap_glm(
            X,
            Y,
            n_boot=20,
            block_size=3,
            run_indices=[np.arange(0, 6), np.arange(6, 12)],
            use_bca=False,
            seed=7,
        )
        np.testing.assert_allclose(by_run_indices.boot_betas, by_boundaries.boot_betas, atol=1e-12)
        np.testing.assert_allclose(by_run_indices.beta_ci, by_boundaries.beta_ci, atol=1e-12)

    def test_run_indices_compatibility_matches_run_boundaries(self, rng):
        """fmrireg-style run_indices should produce boundary-equivalent blocks."""
        n, p, V = 10, 2, 2
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
        Y = rng.standard_normal((n, V))

        run_indices_0 = [np.arange(0, 5), np.arange(5, 10)]
        run_indices_1 = [np.arange(1, 6), np.arange(6, 11)]

        by_boundaries = bootstrap_glm(
            X,
            Y,
            n_boot=8,
            block_size=3,
            run_boundaries=[0, 5],
            use_bca=False,
            seed=11,
        )
        by_run_indices_0 = bootstrap_glm(
            X,
            Y,
            n_boot=8,
            block_size=3,
            run_indices=run_indices_0,
            use_bca=False,
            seed=11,
        )
        by_run_indices_1 = bootstrap_glm(
            X,
            Y,
            n_boot=8,
            block_size=3,
            run_indices=run_indices_1,
            use_bca=False,
            seed=11,
        )

        np.testing.assert_allclose(by_run_indices_0.boot_betas, by_boundaries.boot_betas, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(by_run_indices_1.boot_betas, by_boundaries.boot_betas, atol=0.0, rtol=0.0)

    def test_run_indices_and_run_boundaries_mutually_exclusive(self, rng):
        n, p, V = 10, 2, 1
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
        Y = rng.standard_normal((n, V))
        run_indices = [np.arange(0, 5), np.arange(5, 10)]

        with pytest.raises(ValueError, match="only one of run_boundaries or run_indices"):
            bootstrap_glm(
                X,
                Y,
                n_boot=4,
                block_size=3,
                run_boundaries=[0, 5],
                run_indices=run_indices,
                seed=0,
            )

    def test_default_block_size_matches_fmrireg_single_run_rule(self, rng):
        """Parity: default block_size should be round(max(10, n/20))."""
        n, p, V = 400, 2, 2
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
        Y = rng.standard_normal((n, V))

        res_default = bootstrap_glm(X, Y, n_boot=6, use_bca=False, seed=3)
        res_explicit = bootstrap_glm(X, Y, n_boot=6, block_size=20, use_bca=False, seed=3)

        np.testing.assert_allclose(res_default.boot_betas, res_explicit.boot_betas, atol=0.0, rtol=0.0)

    def test_default_block_size_matches_fmrireg_run_indices_rule(self, rng):
        """Parity: with run_indices default block_size should use min(run_len)/4."""
        n, p, V = 32, 2, 2
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 1))])
        Y = rng.standard_normal((n, V))
        run_indices = [np.arange(0, 20), np.arange(20, 32)]  # min run length = 12 -> round(3)

        res_default = bootstrap_glm(
            X, Y, n_boot=6, run_indices=run_indices, use_bca=False, seed=7
        )
        res_explicit = bootstrap_glm(
            X, Y, n_boot=6, block_size=3, run_indices=run_indices, use_bca=False, seed=7
        )

        np.testing.assert_allclose(res_default.boot_betas, res_explicit.boot_betas, atol=0.0, rtol=0.0)
