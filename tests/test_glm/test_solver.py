"""Tests for the core GLM solver."""

import numpy as np
import pytest
from scipy import stats as sp_stats

from fmrimod.glm.solver import fast_preproject, fast_lm_matrix, Projection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_design(rng):
    """Simple full-rank design: intercept + 2 regressors, 100 obs."""
    n, p = 100, 3
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    return X


@pytest.fixture
def rank_deficient_design(rng):
    """Rank-deficient design: col3 = col1 + col2."""
    n = 100
    X = rng.standard_normal((n, 2))
    X = np.column_stack([X, X[:, 0] + X[:, 1]])
    return X


# ---------------------------------------------------------------------------
# fast_preproject tests
# ---------------------------------------------------------------------------

class TestFastPreproject:
    def test_full_rank(self, simple_design):
        proj = fast_preproject(simple_design)
        assert proj.is_full_rank
        assert proj.rank == 3
        assert proj.dfres == 97.0
        assert proj.Pinv.shape == (3, 100)
        assert proj.XtXinv.shape == (3, 3)

    def test_rank_deficient(self, rank_deficient_design):
        proj = fast_preproject(rank_deficient_design)
        assert not proj.is_full_rank
        assert proj.rank == 2
        assert proj.dfres == 98.0

    def test_nan_raises(self):
        X = np.array([[1.0, np.nan], [1.0, 2.0]])
        with pytest.raises(ValueError, match="NA/Inf"):
            fast_preproject(X)

    def test_inf_raises(self):
        X = np.array([[1.0, np.inf], [1.0, 2.0]])
        with pytest.raises(ValueError, match="NA/Inf"):
            fast_preproject(X)

    def test_empty_matrix_raises_clear_error(self):
        with pytest.raises(
            ValueError,
            match="Design matrix must have at least one row and one column",
        ):
            fast_preproject(np.empty((0, 0)))

    def test_XtXinv_is_inverse(self, simple_design):
        proj = fast_preproject(simple_design)
        X = simple_design
        XtX = X.T @ X
        identity_approx = proj.XtXinv @ XtX
        np.testing.assert_allclose(identity_approx, np.eye(3), atol=1e-10)

    def test_Pinv_recovers_betas(self, rng, simple_design):
        """Pinv @ Y should recover true betas."""
        true_betas = np.array([5.0, -2.0, 1.5])
        Y = simple_design @ true_betas + rng.standard_normal(100) * 0.01
        proj = fast_preproject(simple_design)
        betas_hat = proj.Pinv @ Y
        np.testing.assert_allclose(betas_hat, true_betas, atol=0.05)


# ---------------------------------------------------------------------------
# fast_lm_matrix tests
# ---------------------------------------------------------------------------

class TestFastLmMatrix:
    def test_beta_recovery(self, rng, simple_design):
        """Recover known betas from noiseless data."""
        true_betas = np.array([[5.0], [-2.0], [1.5]])
        Y = simple_design @ true_betas
        proj = fast_preproject(simple_design)
        result = fast_lm_matrix(simple_design, Y, proj)
        np.testing.assert_allclose(result.betas, true_betas, atol=1e-10)
        np.testing.assert_allclose(result.rss, [0.0], atol=1e-10)

    def test_multi_voxel(self, rng, simple_design):
        """Multiple voxels simultaneously."""
        V = 20
        true_B = rng.standard_normal((3, V))
        Y = simple_design @ true_B + rng.standard_normal((100, V)) * 0.001
        proj = fast_preproject(simple_design)
        result = fast_lm_matrix(simple_design, Y, proj)
        np.testing.assert_allclose(result.betas, true_B, atol=0.01)
        assert result.sigma2.shape == (V,)

    def test_sigma2_correct(self, rng, simple_design):
        """Verify sigma^2 matches known noise."""
        noise_sd = 2.0
        true_B = np.array([[1.0], [0.0], [0.0]])
        Y = simple_design @ true_B + rng.standard_normal((100, 1)) * noise_sd
        proj = fast_preproject(simple_design)
        result = fast_lm_matrix(simple_design, Y, proj)
        # sigma^2 should be close to noise_sd^2
        np.testing.assert_allclose(result.sigma2[0], noise_sd ** 2, rtol=0.3)

    def test_return_fitted(self, rng, simple_design):
        Y = rng.standard_normal((100, 5))
        proj = fast_preproject(simple_design)
        result = fast_lm_matrix(simple_design, Y, proj, return_fitted=True)
        assert result.fitted is not None
        assert result.fitted.shape == (100, 5)
        # fitted + residual = Y
        residuals = Y - result.fitted
        rss_check = np.sum(residuals ** 2, axis=0)
        np.testing.assert_allclose(result.rss, rss_check, atol=1e-8)

    def test_1d_Y(self, rng, simple_design):
        """1-D Y vector should work."""
        Y = rng.standard_normal(100)
        proj = fast_preproject(simple_design)
        result = fast_lm_matrix(simple_design, Y, proj)
        assert result.betas.shape == (3, 1)

    def test_matches_numpy_lstsq(self, rng, simple_design):
        """Results should match np.linalg.lstsq."""
        V = 10
        Y = rng.standard_normal((100, V))
        proj = fast_preproject(simple_design)
        result = fast_lm_matrix(simple_design, Y, proj)

        betas_ref, _, _, _ = np.linalg.lstsq(simple_design, Y, rcond=None)
        np.testing.assert_allclose(result.betas, betas_ref, atol=1e-10)

    def test_dimension_mismatch_rows_raises_clear_error(self, rng, simple_design):
        """Y with different row count should fail with a stable ValueError."""
        Y_bad = rng.standard_normal((simple_design.shape[0] - 1, 2))
        proj = fast_preproject(simple_design)
        with pytest.raises(ValueError, match="X and Y dimensions do not match"):
            fast_lm_matrix(simple_design, Y_bad, proj)

    def test_projection_shape_mismatch_raises_clear_error(self, rng):
        """Projection from a different design should fail fast."""
        X = np.column_stack([np.ones(20), rng.standard_normal((20, 2))])
        X_other = np.column_stack([np.ones(20), rng.standard_normal((20, 1))])
        Y = rng.standard_normal((20, 3))

        proj_other = fast_preproject(X_other)
        with pytest.raises(ValueError, match="X and projection dimensions do not match"):
            fast_lm_matrix(X, Y, proj_other)
