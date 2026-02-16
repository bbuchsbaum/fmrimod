"""Tests for numeric helpers: PACF, stationarity, invertibility, ACVF."""

import numpy as np
import pytest

from fmrimod.ar.numhelpers import (
    ar_to_pacf,
    enforce_invertible_ma,
    enforce_stationary_ar,
    levinson_durbin,
    pacf_to_ar,
    run_avg_acvf,
    segmented_acvf,
)


class TestPACFRoundTrip:
    """pacf_to_ar and ar_to_pacf should be exact inverses."""

    @pytest.mark.parametrize("kappa", [
        np.array([0.6]),
        np.array([0.6, -0.3]),
        np.array([0.6, -0.3, 0.1]),
        np.array([0.9, -0.5, 0.3, -0.1]),
        np.array([0.0]),
        np.array([0.0, 0.0]),
    ])
    def test_round_trip(self, kappa):
        phi = pacf_to_ar(kappa)
        kappa_back = ar_to_pacf(phi)
        np.testing.assert_allclose(kappa, kappa_back, atol=1e-10)

    def test_empty(self):
        assert len(pacf_to_ar(np.array([]))) == 0
        assert len(ar_to_pacf(np.array([]))) == 0

    def test_ar1_identity(self):
        # For AR(1), PACF == phi
        phi = np.array([0.7])
        kappa = ar_to_pacf(phi)
        np.testing.assert_allclose(kappa, phi, atol=1e-12)

    def test_inverse_round_trip(self):
        """Start from AR coefficients, go to PACF, back to AR."""
        phi = np.array([0.5, -0.2])
        kappa = ar_to_pacf(phi)
        phi_back = pacf_to_ar(kappa)
        np.testing.assert_allclose(phi, phi_back, atol=1e-10)


class TestEnforceStationaryAR:
    def test_already_stationary(self):
        phi = np.array([0.5])
        result = enforce_stationary_ar(phi)
        np.testing.assert_allclose(result, phi, atol=1e-12)

    def test_clips_ar1(self):
        phi = np.array([1.5])
        result = enforce_stationary_ar(phi)
        assert abs(result[0]) <= 0.99

    def test_higher_order_stationary(self):
        """Enforce stationarity on AR(3) coefficients."""
        phi = np.array([1.5, -0.8, 0.3])
        result = enforce_stationary_ar(phi)
        # Check companion roots inside unit circle
        poly = np.concatenate([[1.0], -result])
        roots = np.roots(poly)
        assert np.all(np.abs(roots) < 1.0 + 1e-6)

    def test_empty(self):
        result = enforce_stationary_ar(np.array([]))
        assert len(result) == 0

    def test_bound_parameter(self):
        phi = np.array([0.95])
        result = enforce_stationary_ar(phi, bound=0.9)
        assert abs(result[0]) <= 0.9 + 1e-10


class TestEnforceInvertibleMA:
    def test_already_invertible(self):
        theta = np.array([0.3])
        result = enforce_invertible_ma(theta)
        np.testing.assert_allclose(result, theta, atol=1e-10)

    def test_reflects_root(self):
        theta = np.array([2.0])  # root at z = -0.5, inside unit circle
        result = enforce_invertible_ma(theta)
        # Check all roots outside unit circle
        poly = np.concatenate([[1.0], result])
        roots = np.roots(poly[::-1])
        assert np.all(np.abs(roots) > 1.0 - 1e-6)

    def test_empty(self):
        result = enforce_invertible_ma(np.array([]))
        assert len(result) == 0

    def test_ma2(self):
        theta = np.array([1.5, 0.8])
        result = enforce_invertible_ma(theta)
        poly = np.concatenate([[1.0], result])
        roots = np.roots(poly[::-1])
        assert np.all(np.abs(roots) > 1.0 - 1e-6)


class TestLevinsonDurbin:
    def test_ar1_recovery(self):
        """Recover AR(1) from known autocovariance."""
        phi_true = 0.6
        sigma2 = 1.0
        # gamma(0) = sigma2 / (1 - phi^2), gamma(1) = phi * gamma(0)
        gamma0 = sigma2 / (1 - phi_true ** 2)
        gamma = np.array([gamma0, phi_true * gamma0])
        phi_est, sig2 = levinson_durbin(gamma, 1)
        np.testing.assert_allclose(phi_est, [phi_true], atol=1e-10)

    def test_ar2_recovery(self):
        """Recover AR(2) from known autocovariance."""
        phi = np.array([0.5, -0.3])
        # Compute ACVF from AR(2): gamma(k) = phi1*gamma(k-1) + phi2*gamma(k-2)
        # gamma(0) = 1/(1 - phi1*r1 - phi2*r2), but we can use Yule-Walker
        # directly: gamma(1) = phi1*gamma(0) + phi2*gamma(-1) = phi1*gamma(0) + phi2*gamma(1)
        # => gamma(1)*(1 - phi2) = phi1*gamma(0) => gamma(1) = phi1/(1-phi2) * gamma(0)
        gamma = np.zeros(3)
        gamma[0] = 1.0
        gamma[1] = phi[0] / (1.0 - phi[1])  # from Yule-Walker: gamma(1)(1-phi2)=phi1*gamma(0)
        gamma[2] = phi[0] * gamma[1] + phi[1] * gamma[0]
        phi_est, _ = levinson_durbin(gamma, 2)
        np.testing.assert_allclose(phi_est, phi, atol=1e-10)

    def test_zero_variance(self):
        gamma = np.array([0.0, 0.0])
        phi, sig2 = levinson_durbin(gamma, 1)
        assert np.all(phi == 0)

    def test_order_zero(self):
        gamma = np.array([1.0])
        phi, sig2 = levinson_durbin(gamma, 0)
        assert len(phi) == 0


class TestSegmentedACVF:
    def test_single_segment(self):
        np.random.seed(42)
        y = np.random.randn(200)
        gamma = segmented_acvf(y, np.array([0]), max_lag=5)
        assert len(gamma) == 6
        assert gamma[0] > 0  # variance > 0

    def test_two_segments(self):
        np.random.seed(42)
        y = np.random.randn(200)
        seg_starts = np.array([0, 100])
        gamma = segmented_acvf(y, seg_starts, max_lag=3)
        assert len(gamma) == 4

    def test_centering(self):
        # With constant series, centered ACVF should be ~0
        y = np.ones(100) * 5.0
        gamma = segmented_acvf(y, np.array([0]), max_lag=3, center=True)
        np.testing.assert_allclose(gamma, 0, atol=1e-10)

    def test_unbiased(self):
        np.random.seed(42)
        y = np.random.randn(100)
        gamma_b = segmented_acvf(y, np.array([0]), max_lag=3, unbiased=False)
        gamma_u = segmented_acvf(y, np.array([0]), max_lag=3, unbiased=True)
        # Unbiased ACVF at lag 0 should be close to biased but slightly larger
        assert gamma_u[0] >= gamma_b[0] - 1e-10


class TestRunAvgACVF:
    def test_basic(self):
        np.random.seed(42)
        mat = np.random.randn(100, 5)
        gamma = run_avg_acvf(mat, max_lag=3)
        assert len(gamma) == 4
        assert gamma[0] > 0

    def test_1d_input(self):
        y = np.random.randn(100)
        gamma = run_avg_acvf(y, max_lag=3)
        assert len(gamma) == 4
