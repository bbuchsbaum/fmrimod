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

    def test_single_row_f_matches_t_squared(self, fitted_model):
        """Single-row F contrasts should equal squared t contrasts voxelwise."""
        con_vec = np.array([0.0, 1.0, 0.0, 0.0])
        con_mat = con_vec[np.newaxis, :]

        t_res = contrast_t(
            con_vec, fitted_model["betas"], fitted_model["XtXinv"],
            fitted_model["sigma"], fitted_model["dfres"],
        )
        f_res = contrast_f(
            con_mat, fitted_model["betas"], fitted_model["XtXinv"],
            fitted_model["sigma"], fitted_model["dfres"],
        )
        f_res_vec = contrast_f_vectorized(
            con_mat, fitted_model["betas"], fitted_model["XtXinv"],
            fitted_model["sigma"], fitted_model["dfres"],
        )

        np.testing.assert_allclose(f_res.stat, t_res.stat ** 2, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(f_res_vec.stat, t_res.stat ** 2, atol=1e-10, rtol=1e-10)

        p_from_t = 2.0 * sp_stats.t.sf(np.abs(t_res.stat), fitted_model["dfres"])
        np.testing.assert_allclose(f_res.p_value, p_from_t, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(f_res_vec.p_value, p_from_t, atol=1e-12, rtol=1e-12)

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

    def test_vectorized_matches_loop_for_near_singular_covariance(self):
        """Near-singular F-covariance should not diverge by implementation path."""
        C = np.array([
            [0.49169351638328906, 0.49993099756858916],
            [0.49993099756858916, 0.5083064836167119],
        ])
        con = np.linalg.cholesky(C)
        XtXinv = np.eye(2)
        cb = np.array([-0.7129566076275948, 0.7012070271068789])
        betas = np.linalg.pinv(con) @ cb[:, np.newaxis]
        sigma = np.array([1.0])
        dfres = 50.0

        with pytest.warns(RuntimeWarning, match="singular"):
            res_loop = contrast_f(con, betas, XtXinv, sigma, dfres)
        with pytest.warns(RuntimeWarning, match="singular"):
            res_vec = contrast_f_vectorized(con, betas, XtXinv, sigma, dfres)

        np.testing.assert_allclose(res_loop.stat, res_vec.stat, rtol=1e-8, atol=0.0)
        np.testing.assert_allclose(res_loop.p_value, res_vec.p_value, atol=0.0)

    def test_singular_contrast_matrix_emits_warning(self, fitted_model):
        """Parity: singular contrast covariance should emit a warning."""
        singular = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
            ]
        )

        with pytest.warns(RuntimeWarning, match="singular"):
            contrast_f(
                singular,
                fitted_model["betas"],
                fitted_model["XtXinv"],
                fitted_model["sigma"],
                fitted_model["dfres"],
            )

        with pytest.warns(RuntimeWarning, match="singular"):
            contrast_f_vectorized(
                singular,
                fitted_model["betas"],
                fitted_model["XtXinv"],
                fitted_model["sigma"],
                fitted_model["dfres"],
            )

    def test_near_singular_contrast_paths_stay_nonnegative_and_consistent(self):
        """Near-singular C(X'X)^-1C' should not diverge across implementations."""
        con = np.array(
            [
                [0.27883477737431556, 0.9451932065760001, -0.7398868049493273, 0.7135642142391335],
                [0.2788347773800641, 0.9451932065802877, -0.7398868049528264, 0.7135642142464169],
            ],
            dtype=np.float64,
        )
        XtXinv = np.array(
            [
                [2.1665714590504734, -1.5155012449012177, 0.5119548327016575, -1.3750317990254468],
                [-1.5155012449012177, 2.871574347603966, 0.0984410703500544, 1.1369647753467884],
                [0.5119548327016575, 0.0984410703500544, 1.8204386931028385, 0.4115173263091359],
                [-1.3750317990254468, 1.1369647753467884, 0.4115173263091359, 1.1964216834646226],
            ],
            dtype=np.float64,
        )
        betas = np.array(
            [
                [-0.2980298067196495, -0.2855725668594085, -0.5665872435909065, -0.15357909625050895, -1.7406346901674916],
                [0.8765996934014191, 0.9616865870883877, -0.4427738728469823, -1.3797546259079334, -0.6467092698202764],
                [0.9476143296978757, 0.6255207733817739, -0.30035356091877463, 0.8972747801405839, -1.0414807563325195],
                [-0.6126363310909915, 0.47460724668348964, -0.09591332970445857, -0.5892256991552035, -2.5121277305114025],
            ],
            dtype=np.float64,
        )
        sigma = np.array(
            [0.7712163346120159, 0.4237218133973122, 1.8429223285514202, 0.7003256861085275, 0.11436375875858676],
            dtype=np.float64,
        )

        with pytest.warns(RuntimeWarning, match="singular"):
            res_loop = contrast_f(con, betas, XtXinv, sigma, 50.0)
        with pytest.warns(RuntimeWarning, match="singular"):
            res_vec = contrast_f_vectorized(con, betas, XtXinv, sigma, 50.0)

        assert np.all(res_loop.stat >= 0)
        assert np.all(res_vec.stat >= 0)
        np.testing.assert_allclose(res_loop.stat, res_vec.stat, rtol=1e-8, atol=1e-8)

    def test_near_singular_full_rank_paths_remain_consistent(self):
        """Regression: full-rank but ill-conditioned contrasts should not diverge."""
        con = np.array(
            [
                [-2.1745990929414605, -0.2299241734353348, -0.4965968636170511, 0.8793377350589894],
                [-2.1745990329050393, -0.22992405560704776, -0.4965967255258879, 0.8793378240108157],
            ],
            dtype=np.float64,
        )
        XtXinv = np.array(
            [
                [4.836087839523261, 1.375071407562073, 0.613603206980325, 3.2761630210390784],
                [1.375071407562073, 2.13534657975023, -0.4661270313701171, 0.8054592937756313],
                [0.613603206980325, -0.4661270313701171, 1.3288360155610794, -0.14437520463960474],
                [3.2761630210390784, 0.8054592937756313, -0.14437520463960474, 3.384119138249119],
            ],
            dtype=np.float64,
        )
        betas = np.array(
            [
                [-1.2540775301851623, 0.91759001436943, 1.9685895863625478, 0.7646390177729496, -1.9073797742752288],
                [0.031042851965216136, 0.6048951922036501, 0.864272530755256, -0.5228076795885733, -0.7921099430022497],
                [-0.23814085094626347, 0.8210366361819899, 1.18655043580779, -0.0453276274279349, -0.8778759891864901],
                [-0.5767331237505733, 2.6153956499442335, -0.025604592133619923, 0.6134180872766275, 0.35101824002975734],
            ],
            dtype=np.float64,
        )
        sigma = np.array(
            [0.5858254482365985, 0.11081279321442283, 1.831228667534842, 1.4241653451289142, 0.48278532704179955],
            dtype=np.float64,
        )

        with pytest.warns(RuntimeWarning, match="singular"):
            res_loop = contrast_f(con, betas, XtXinv, sigma, 50.0)
        with pytest.warns(RuntimeWarning, match="singular"):
            res_vec = contrast_f_vectorized(con, betas, XtXinv, sigma, 50.0)

        assert np.all(res_loop.stat >= 0)
        assert np.all(res_vec.stat >= 0)
        np.testing.assert_allclose(res_loop.stat, res_vec.stat, rtol=1e-8, atol=1e-8)

    def test_empty_contrast_matrix_raises(self, fitted_model):
        """Empty-row F-contrast should fail with a clear validation error."""
        con = np.zeros((0, fitted_model["betas"].shape[0]))

        with pytest.raises(ValueError, match="at least one contrast row"):
            contrast_f(
                con, fitted_model["betas"], fitted_model["XtXinv"],
                fitted_model["sigma"], fitted_model["dfres"],
            )

        with pytest.raises(ValueError, match="at least one contrast row"):
            contrast_f_vectorized(
                con, fitted_model["betas"], fitted_model["XtXinv"],
                fitted_model["sigma"], fitted_model["dfres"],
            )

    def test_f_contrast_wrong_width_raises(self, fitted_model):
        """F-contrast column count must match number of model coefficients."""
        con = np.array([[1.0, 0.0]])  # too few columns

        with pytest.raises(ValueError, match="model has"):
            contrast_f(
                con, fitted_model["betas"], fitted_model["XtXinv"],
                fitted_model["sigma"], fitted_model["dfres"],
            )

        with pytest.raises(ValueError, match="model has"):
            contrast_f_vectorized(
                con, fitted_model["betas"], fitted_model["XtXinv"],
                fitted_model["sigma"], fitted_model["dfres"],
            )

    def test_singular_contrast_rows_do_not_change_f_scale(self, fitted_model):
        """Duplicate/proportional contrast rows should use effective rank."""
        single = np.array([[0.0, 1.0, 0.0, 0.0]])
        singular = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
            ]
        )

        res_single = contrast_f(
            single, fitted_model["betas"], fitted_model["XtXinv"],
            fitted_model["sigma"], fitted_model["dfres"],
        )
        with pytest.warns(RuntimeWarning, match="singular"):
            res_singular = contrast_f(
                singular, fitted_model["betas"], fitted_model["XtXinv"],
                fitted_model["sigma"], fitted_model["dfres"],
            )
        np.testing.assert_allclose(res_singular.stat, res_single.stat, rtol=1e-10)
        assert res_singular.df[0] == 1.0

        with pytest.warns(RuntimeWarning, match="singular"):
            res_singular_vec = contrast_f_vectorized(
                singular, fitted_model["betas"], fitted_model["XtXinv"],
                fitted_model["sigma"], fitted_model["dfres"],
            )
        np.testing.assert_allclose(res_singular_vec.stat, res_single.stat, rtol=1e-10)
        assert res_singular_vec.df[0] == 1.0

    def test_all_zero_f_contrast_returns_zero_stat(self, fitted_model):
        """All-zero F contrast should produce F=0 and p=1, not NaN/division errors."""
        con = np.zeros((1, fitted_model["betas"].shape[0]))
        res = contrast_f(
            con, fitted_model["betas"], fitted_model["XtXinv"],
            fitted_model["sigma"], fitted_model["dfres"],
        )
        np.testing.assert_array_equal(res.stat, np.zeros_like(res.stat))
        np.testing.assert_array_equal(res.p_value, np.ones_like(res.p_value))

        res_vec = contrast_f_vectorized(
            con, fitted_model["betas"], fitted_model["XtXinv"],
            fitted_model["sigma"], fitted_model["dfres"],
        )
        np.testing.assert_array_equal(res_vec.stat, np.zeros_like(res_vec.stat))
        np.testing.assert_array_equal(res_vec.p_value, np.ones_like(res_vec.p_value))

    def test_near_singular_rows_remain_nonnegative_and_match_vectorized(self, fitted_model):
        """Near-singular redundant rows should not produce negative/unstable F stats."""
        con = np.array(
            [
                [0.40543018, -0.48349144, 0.48992811, -0.32665846],
                [0.81086036, -0.96698289, 0.97985622, -0.65331692],
            ]
        )

        with pytest.warns(RuntimeWarning, match="singular"):
            res_loop = contrast_f(
                con, fitted_model["betas"], fitted_model["XtXinv"],
                fitted_model["sigma"], fitted_model["dfres"],
            )
        with pytest.warns(RuntimeWarning, match="singular"):
            res_vec = contrast_f_vectorized(
                con, fitted_model["betas"], fitted_model["XtXinv"],
                fitted_model["sigma"], fitted_model["dfres"],
            )

        assert np.all(np.isfinite(res_loop.stat))
        assert np.all(res_loop.stat >= 0.0)
        np.testing.assert_allclose(res_loop.stat, res_vec.stat, rtol=1e-8, atol=1e-8)
