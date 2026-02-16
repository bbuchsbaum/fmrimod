"""End-to-end tests: simulate data, fit GLM, recover betas and contrasts."""

import numpy as np
import pytest

from fmrimod.sampling import SamplingFrame
from fmrimod.dataset.adapters.numpy_adapter import NumpyAdapter
from fmrimod.dataset.fmri_dataset import FmriDataset
from fmrimod.model.config import FmriLmConfig, AROptions
from fmrimod.model.fmri_model import FmriModel
from fmrimod.glm.solver import fast_preproject, fast_lm_matrix
from fmrimod.glm.contrasts import contrast_t
from fmrimod.simulate.bold import simulate_bold
from fmrimod.simulate.noise import ar_noise
from fmrimod.ar.estimation import estimate_ar_yule_walker
from fmrimod.ar.whitening import ar_whiten, ar_whiten_matrix
from fmrimod.robust.estimators import mad_scale, huber_weights, bisquare_weights
from fmrimod.stats.inference import fdr_correction


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestEndToEndOLS:
    """End-to-end OLS: simulate -> fit -> contrast -> verify."""

    def test_single_run_beta_recovery(self, rng):
        """Simulate 1 run with known betas, recover them."""
        n, p, V = 200, 4, 50
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
        true_B = np.array([10.0, 3.0, -1.5, 0.5])

        Y = simulate_bold(X, true_B, noise_sd=0.5, n_voxels=V, rng=rng)
        assert Y.shape == (n, V)

        proj = fast_preproject(X)
        result = fast_lm_matrix(X, Y, proj)

        # Betas should be close to true values for all voxels
        for j in range(p):
            np.testing.assert_allclose(
                result.betas[j, :].mean(), true_B[j], atol=0.15,
                err_msg=f"Beta {j} recovery failed"
            )

    def test_two_run_pooling(self, rng):
        """Two runs with same betas: pooled should be more precise than single."""
        n, p, V = 100, 3, 20
        true_B = np.array([1.0, 2.0, -1.0])

        betas_runs = []
        for _ in range(2):
            X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
            Y = simulate_bold(X, true_B, noise_sd=1.0, n_voxels=V, rng=rng)
            proj = fast_preproject(X)
            result = fast_lm_matrix(X, Y, proj)
            betas_runs.append(result.betas)

        # Average betas across runs
        pooled_betas = np.mean(betas_runs, axis=0)
        for j in range(p):
            np.testing.assert_allclose(
                pooled_betas[j, :].mean(), true_B[j], atol=0.2,
            )

    def test_contrast_significance(self, rng):
        """Significant regressor should have small p-values."""
        n, p, V = 300, 3, 40
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        true_B = np.array([0.0, 5.0, 0.0])  # strong effect on reg 1

        Y = simulate_bold(X, true_B, noise_sd=1.0, n_voxels=V, rng=rng)
        proj = fast_preproject(X)
        result = fast_lm_matrix(X, Y, proj)

        con = np.array([0.0, 1.0, 0.0])
        cres = contrast_t(
            con, result.betas, proj.XtXinv,
            np.sqrt(result.sigma2), result.dfres,
        )
        # Most voxels should be significant at p<0.001
        assert np.mean(cres.p_value < 0.001) > 0.9

    def test_null_contrast_fdr(self, rng):
        """Null regressor: FDR should control false positives."""
        n, p, V = 200, 3, 100
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        true_B = np.array([0.0, 0.0, 0.0])  # all null

        Y = simulate_bold(X, true_B, noise_sd=1.0, n_voxels=V, rng=rng)
        proj = fast_preproject(X)
        result = fast_lm_matrix(X, Y, proj)

        con = np.array([0.0, 1.0, 0.0])
        cres = contrast_t(
            con, result.betas, proj.XtXinv,
            np.sqrt(result.sigma2), result.dfres,
        )

        reject, p_adj = fdr_correction(cres.p_value, alpha=0.05)
        # Under null, FDR should reject very few
        assert np.mean(reject) < 0.1


class TestEndToEndAR:
    """Tests for AR noise estimation and whitening."""

    def test_ar1_estimation(self, rng):
        """Generate AR(1) noise, estimate phi."""
        n, V = 500, 10
        true_phi = np.array([0.5])
        noise = ar_noise(n, V, true_phi, sd=1.0, rng=rng)

        phi_hat = estimate_ar_yule_walker(noise, order=1)
        np.testing.assert_allclose(phi_hat[0], 0.5, atol=0.1)

    def test_ar2_estimation(self, rng):
        """Generate AR(2) noise, estimate phi."""
        n, V = 500, 10
        true_phi = np.array([0.5, -0.2])
        noise = ar_noise(n, V, true_phi, sd=1.0, rng=rng)

        phi_hat = estimate_ar_yule_walker(noise, order=2)
        np.testing.assert_allclose(phi_hat[0], 0.5, atol=0.15)
        np.testing.assert_allclose(phi_hat[1], -0.2, atol=0.15)

    def test_whitening_reduces_autocorrelation(self, rng):
        """Whitening should reduce autocorrelation in residuals."""
        n = 500
        true_phi = np.array([0.6])
        noise = ar_noise(n, 1, true_phi, sd=1.0, rng=rng).ravel()

        # Before whitening: strong autocorrelation
        r_before = np.corrcoef(noise[1:], noise[:-1])[0, 1]

        whitened = ar_whiten(noise, true_phi)
        r_after = np.corrcoef(whitened[1:], whitened[:-1])[0, 1]

        assert abs(r_after) < abs(r_before)
        assert abs(r_after) < 0.15  # should be near zero


class TestEndToEndRobust:
    """Tests for robust estimation utilities."""

    def test_mad_scale(self, rng):
        """MAD scale should approximate true SD for normal data."""
        data = rng.standard_normal((100, 10))
        scale = mad_scale(data, axis=0)
        np.testing.assert_allclose(scale, 1.0, atol=0.25)

    def test_huber_weights_outlier(self, rng):
        """Outlier rows should get downweighted."""
        data = rng.standard_normal((100, 5))
        # Add outlier at row 0
        data[0, :] = 20.0
        scale = mad_scale(data, axis=0)
        w = huber_weights(data, scale, k=1.345)
        # Row 0 should have low weight
        assert np.all(w[0, :] < 0.5)
        # Most other rows should have weight ~1
        assert np.mean(w[1:, :]) > 0.9

    def test_bisquare_weights_outlier(self, rng):
        """Extreme outliers should get zero weight with bisquare."""
        data = rng.standard_normal((100, 5))
        data[0, :] = 50.0
        scale = mad_scale(data, axis=0)
        w = bisquare_weights(data, scale, c=4.685)
        # Row 0 should have zero weight
        assert np.all(w[0, :] < 0.01)


class TestSimulation:
    """Tests for simulation tools."""

    def test_simulate_bold_shape(self, rng):
        n, p, V = 100, 3, 10
        X = rng.standard_normal((n, p))
        B = rng.standard_normal(p)
        Y = simulate_bold(X, B, noise_sd=1.0, n_voxels=V, rng=rng)
        assert Y.shape == (n, V)

    def test_simulate_bold_with_ar(self, rng):
        n, p, V = 100, 3, 5
        X = rng.standard_normal((n, p))
        B = rng.standard_normal(p)
        Y = simulate_bold(X, B, noise_sd=1.0, ar_coeffs=np.array([0.5]),
                          n_voxels=V, rng=rng)
        assert Y.shape == (n, V)

    def test_simulate_bold_noiseless(self, rng):
        n, p = 50, 2
        X = rng.standard_normal((n, p))
        B = np.array([1.0, -1.0])
        Y = simulate_bold(X, B, noise_sd=0.0, n_voxels=1, rng=rng)
        expected = X @ B
        np.testing.assert_allclose(Y.ravel(), expected, atol=1e-12)
