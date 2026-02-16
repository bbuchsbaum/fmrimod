"""Tests for contrast computation (t and F statistics)."""

import numpy as np
import pytest
from scipy import stats as sp_stats

from fmrimod.glm.solver import fast_preproject, fast_lm_matrix
from fmrimod.glm.contrasts import contrast_t, contrast_f, contrast_f_vectorized


@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def fitted_model(rng):
    """Fit a simple model and return components for contrast tests."""
    n, p, V = 200, 4, 30
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    true_B = np.zeros((p, V))
    true_B[1, :] = 3.0  # strong effect on regressor 1
    true_B[2, :] = 0.0  # null effect on regressor 2

    Y = X @ true_B + rng.standard_normal((n, V)) * 1.0
    proj = fast_preproject(X)
    result = fast_lm_matrix(X, Y, proj)
    return {
        "betas": result.betas,
        "XtXinv": proj.XtXinv,
        "sigma": np.sqrt(result.sigma2),
        "dfres": result.dfres,
        "true_B": true_B,
    }


class TestContrastT:
    def test_significant_contrast(self, fitted_model):
        """Regressor 1 has true effect = 3, should be significant."""
        con = np.array([0.0, 1.0, 0.0, 0.0])
        res = contrast_t(
            con, fitted_model["betas"], fitted_model["XtXinv"],
            fitted_model["sigma"], fitted_model["dfres"],
        )
        assert res.stat_type == "t"
        # All voxels should have t > 2
        assert np.all(res.stat > 2.0)
        # All p-values should be < 0.05
        assert np.all(res.p_value < 0.05)

    def test_null_contrast(self, fitted_model):
        """Regressor 2 has true effect = 0, should not be significant (mostly)."""
        con = np.array([0.0, 0.0, 1.0, 0.0])
        res = contrast_t(
            con, fitted_model["betas"], fitted_model["XtXinv"],
            fitted_model["sigma"], fitted_model["dfres"],
        )
        # Most p-values should be > 0.05
        assert np.mean(res.p_value > 0.05) > 0.8

    def test_contrast_estimate(self, fitted_model):
        """Contrast estimate should be close to true beta."""
        con = np.array([0.0, 1.0, 0.0, 0.0])
        res = contrast_t(
            con, fitted_model["betas"], fitted_model["XtXinv"],
            fitted_model["sigma"], fitted_model["dfres"],
        )
        np.testing.assert_allclose(res.estimate, 3.0, atol=0.3)

    def test_se_positive(self, fitted_model):
        con = np.array([0.0, 1.0, 0.0, 0.0])
        res = contrast_t(
            con, fitted_model["betas"], fitted_model["XtXinv"],
            fitted_model["sigma"], fitted_model["dfres"],
        )
        assert res.se is not None
        assert np.all(res.se > 0)

    def test_wrong_length_raises(self, fitted_model):
        con = np.array([1.0, 0.0])  # too short
        with pytest.raises(ValueError, match="length"):
            contrast_t(
                con, fitted_model["betas"], fitted_model["XtXinv"],
                fitted_model["sigma"], fitted_model["dfres"],
            )

    def test_difference_contrast(self, fitted_model):
        """Test [0, 1, -1, 0] contrast."""
        con = np.array([0.0, 1.0, -1.0, 0.0])
        res = contrast_t(
            con, fitted_model["betas"], fitted_model["XtXinv"],
            fitted_model["sigma"], fitted_model["dfres"],
        )
        # true difference is 3 - 0 = 3
        np.testing.assert_allclose(res.estimate, 3.0, atol=0.3)


class TestContrastF:
    def test_significant_f(self, fitted_model):
        """Test regressor 1 with F-test (should be equivalent to t^2)."""
        con = np.array([[0.0, 1.0, 0.0, 0.0]])
        res = contrast_f(
            con, fitted_model["betas"], fitted_model["XtXinv"],
            fitted_model["sigma"], fitted_model["dfres"],
        )
        assert res.stat_type == "F"
        assert np.all(res.stat > 4.0)  # F = t^2 should be large

    def test_multi_row_f(self, fitted_model):
        """Joint test of regressors 1 and 2."""
        con = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        res = contrast_f(
            con, fitted_model["betas"], fitted_model["XtXinv"],
            fitted_model["sigma"], fitted_model["dfres"],
        )
        assert res.df == (2.0, fitted_model["dfres"])
        assert np.all(res.stat >= 0)

    def test_vectorized_matches_loop(self, fitted_model):
        """Vectorised F should match loop-based F."""
        con = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        res_loop = contrast_f(
            con, fitted_model["betas"], fitted_model["XtXinv"],
            fitted_model["sigma"], fitted_model["dfres"],
        )
        res_vec = contrast_f_vectorized(
            con, fitted_model["betas"], fitted_model["XtXinv"],
            fitted_model["sigma"], fitted_model["dfres"],
        )
        np.testing.assert_allclose(res_loop.stat, res_vec.stat, atol=1e-10)
        np.testing.assert_allclose(res_loop.p_value, res_vec.p_value, atol=1e-10)
