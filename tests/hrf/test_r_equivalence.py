"""Tests to ensure equivalence with R fmrihrf package test coverage."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from fmrimod import as_hrf, get_hrf, regressor, regressor_set
from fmrimod.hrf import hrf_library
from fmrimod.hrf.core import hrf_from_coefficients
from fmrimod.hrf.generators import gamma_generator
from fmrimod.hrf.penalty import penalty_matrix
from fmrimod.hrf.reconstruction import reconstruction_matrix
from fmrimod.sampling import SamplingFrame
from fmrimod.utils.misc import hrf_toeplitz, recycle_or_error, single_trial_regressor


class TestHRFToeplitz:
    """Test HRF Toeplitz matrix construction (test_fft_and_toeplitz.R)."""
    
    def test_toeplitz_construction(self):
        """hrf_toeplitz constructs correct Toeplitz matrix."""
        # Create a simple box HRF
        def box_hrf_fun(t):
            t = np.asarray(t)
            return np.where((t >= 0) & (t <= 1), 1.0, 0.0)
        
        box_hrf = as_hrf(box_hrf_fun, name="box", span=1)
        
        time = np.arange(3)  # 0, 1, 2
        n = 5
        
        # Dense matrix
        H_dense = hrf_toeplitz(box_hrf, time, n, sparse=False)
        
        # Check it's a proper Toeplitz matrix
        assert H_dense.shape == (n, n)
        
        # First column should be [1, 1, 0, 0, 0]
        expected_col = [1, 1, 0, 0, 0]
        assert np.allclose(H_dense[:, 0], expected_col)
        
        # First row should be [1, 0, 0, 0, 0]
        expected_row = [1, 0, 0, 0, 0]
        assert np.allclose(H_dense[0, :], expected_row)
        
        # Sparse matrix
        H_sparse = hrf_toeplitz(box_hrf, time, n, sparse=True)
        assert issparse(H_sparse)
        assert np.allclose(H_sparse.toarray(), H_dense)

    def test_toeplitz_matrix_hrf_output(self):
        """hrf_toeplitz flattens matrix-valued HRFs in column-major order."""
        matrix_hrf = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

        expected = hrf_toeplitz(
            lambda t: matrix_hrf,
            np.array([0, 1, 2]),
            6,
            sparse=False
        )
        assert expected.shape == (6, 6)
        assert np.allclose(
            expected[:, 0],
            [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
        )

    def test_toeplitz_length_validation(self):
        """hrf_toeplitz rejects lengths shorter than HRF output."""
        matrix_hrf = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        with pytest.raises(ValueError, match="invalid times argument"):
            hrf_toeplitz(lambda t: matrix_hrf, np.array([0, 1, 2]), 4)


class TestRegressor:
    """Test regressor functionality (test_regressor.R)."""
    
    def test_shift_method(self):
        """Regressor shift method works correctly."""
        reg = regressor(onsets=[10, 30, 50], hrf="spmg1")
        
        # Shift forward
        reg_shifted = reg.shift(2.0)
        assert np.array_equal(reg_shifted.onsets, [12, 32, 52])
        
        # Shift backward
        reg_back = reg.shift(-5.0)
        assert np.array_equal(reg_back.onsets, [5, 25, 45])
        
        # Check other attributes preserved
        assert reg_shifted.hrf == reg.hrf
        assert np.array_equal(reg_shifted.duration, reg.duration)
        assert np.array_equal(reg_shifted.amplitude, reg.amplitude)
        assert reg_shifted.span == reg.span
        assert reg_shifted.summate == reg.summate
    
    def test_zero_amplitude_filtering(self):
        """Zero amplitude events are filtered out."""
        reg = regressor(
            onsets=[10, 20, 30, 40],
            amplitude=[1, 0, 1, 0],
            hrf="spmg1"
        )
        
        # After filtering, should only have 2 events
        assert len(reg.onsets) == 2
        assert np.array_equal(reg.onsets, [10, 30])
        assert np.array_equal(reg.amplitude, [1, 1])
    
    def test_nan_handling(self):
        """NaN values are handled correctly."""
        reg = regressor(
            onsets=[10, 20, 30],
            amplitude=[1, np.nan, 1],
            hrf="spmg1"
        )
        
        # NaN amplitude should be filtered
        assert len(reg.onsets) == 2
        assert np.array_equal(reg.onsets, [10, 30])


class TestPenaltyMatrix:
    """Test penalty matrix construction (test_penalty_matrix.R)."""
    
    def test_bspline_penalty(self):
        """B-spline roughness penalties."""
        hrf = get_hrf("bspline")
        P = penalty_matrix(hrf, order=2)
        
        # Should be symmetric
        assert np.allclose(P, P.T)
        
        # Should be nbasis x nbasis
        assert P.shape == (hrf.nbasis, hrf.nbasis)
        
        # Should be positive semi-definite
        eigenvals = np.linalg.eigvals(P)
        assert np.all(eigenvals >= -1e-10)  # Allow small numerical errors
    
    def test_spmg3_penalty(self):
        """SPMG3 derivative shrinkage."""
        hrf = get_hrf("spmg3")
        P = penalty_matrix(hrf, shrink_deriv=4)
        
        assert P.shape == (3, 3)
        # Diagonal should be [0, shrink_deriv, shrink_deriv]
        assert P[0, 0] == 0
        assert P[1, 1] == 4
        assert P[2, 2] == 4
        # Off-diagonal should be 0
        assert P[0, 1] == 0
        assert P[0, 2] == 0
        assert P[1, 2] == 0


class TestHRFFromCoefficients:
    """Test HRF from coefficients functionality (test_hrf_from_coefficients.R)."""
    
    def test_creates_single_hrf(self):
        """hrf_from_coefficients creates single HRF from weighted combination."""
        hrf_base = get_hrf("spmg3")
        coefs = np.array([1.0, 0.5, -0.2])
        
        hrf_combined = hrf_from_coefficients(hrf_base, coefs)
        
        # Should have single basis
        assert hrf_combined.nbasis == 1
        
        # Evaluate and check
        t = np.linspace(0, 20, 100)
        result = hrf_combined(t)
        
        # Compare with manual calculation
        basis_eval = hrf_base(t)
        expected = basis_eval @ coefs
        
        assert np.allclose(result, expected)


class TestHRFLibrary:
    """Test HRF library generation (test_hrf_library.R)."""
    
    def test_parameter_grid_expansion(self):
        """HRF library expands parameter grid correctly."""
        param_grid = pd.DataFrame({
            'shape': [4, 6, 8],
            'rate': [1.0, 1.0, 1.0]
        })
        
        lib = hrf_library(gamma_generator, param_grid)
        
        # Should have 3 basis functions
        assert lib.nbasis == 3
        
        # Test with dict
        param_dict = {'shape': [4, 6, 8], 'rate': 1.0}
        lib2 = hrf_library(gamma_generator, param_dict)
        assert lib2.nbasis == 3
    
    def test_duplicate_parameter_detection(self):
        """Duplicate parameter names are detected."""
        param_grid = pd.DataFrame({'shape': [4, 6]})
        
        with pytest.raises(ValueError, match="Duplicate parameter"):
            hrf_library(gamma_generator, param_grid, shape=5)


class TestReconstructionMatrix:
    """Test reconstruction matrix functionality."""
    
    def test_reconstruction_matrix_shape(self):
        """Reconstruction matrix has correct shape."""
        hrf = get_hrf("spmg3")
        sf = SamplingFrame(blocklens=[50], tr=2.0)
        
        R = reconstruction_matrix(hrf, sf)
        
        # Shape should be (n_samples, nbasis)
        assert R.shape == (50, 3)
        
        # With time array
        t = np.linspace(0, 20, 100)
        R2 = reconstruction_matrix(hrf, t)
        assert R2.shape == (100, 3)
    
    def test_reconstruction_equivalence(self):
        """Reconstruction matches hrf_from_coefficients."""
        hrf = get_hrf("spmg3")
        coefs = np.array([1.0, 0.5, -0.2])
        
        # Method 1: reconstruction matrix
        t = np.linspace(0, 20, 100)
        R = reconstruction_matrix(hrf, t)
        recon1 = R @ coefs
        
        # Method 2: hrf_from_coefficients
        hrf_combined = hrf_from_coefficients(hrf, coefs)
        recon2 = hrf_combined(t)
        
        assert np.allclose(recon1, recon2)


class TestUtils:
    """Test utility functions (test_utils.R)."""
    
    def test_recycle_or_error(self):
        """recycle_or_error works correctly."""
        # Scalar recycling
        result = recycle_or_error(5.0, 3, "test")
        assert np.array_equal(result, [5.0, 5.0, 5.0])
        
        # Array of correct length
        result = recycle_or_error([1, 2, 3], 3, "test")
        assert np.array_equal(result, [1, 2, 3])
        
        # Array of wrong length
        with pytest.raises(ValueError, match="must have length 1 or 3"):
            recycle_or_error([1, 2], 3, "test")
    
    def test_single_trial_regressor(self):
        """single_trial_regressor creates regressor with single event."""
        reg = single_trial_regressor(
            onset=10.0,
            hrf="spmg1",
            duration=2.0,
            amplitude=1.5,
            span=30.0
        )
        
        assert len(reg.onsets) == 1
        assert reg.onsets[0] == 10.0
        assert reg.duration[0] == 2.0
        assert reg.amplitude[0] == 1.5
        assert reg.span == 30.0  # explicit span= is honored, not overridden
        
        # Should reject non-scalar inputs
        with pytest.raises(ValueError, match="onset must be a scalar"):
            single_trial_regressor(onset=[10, 20])


class TestRegressorSet:
    """Test regressor set functionality (test_regressor_set.R)."""
    
    def test_factor_based_creation(self):
        """RegressorSet can be created from factor."""
        # Create factor (using integers to represent levels)
        fac = np.array([0, 1, 0, 1, 2])  # 3 levels
        onsets = np.array([10, 20, 30, 40, 50])
        
        reg_set = regressor_set(
            onsets=onsets,
            fac=fac,
            hrf="spmg1"
        )
        
        # Should have 3 regressors
        assert len(reg_set.regressors) == 3
        
        # Check each regressor has correct events
        assert np.array_equal(reg_set.regressors[0].onsets, [10, 30])
        assert np.array_equal(reg_set.regressors[1].onsets, [20, 40])
        assert np.array_equal(reg_set.regressors[2].onsets, [50])
    
    def test_empty_levels(self):
        """RegressorSet handles empty levels correctly."""
        fac = np.array([0, 0, 2, 2])  # Level 1 is missing
        onsets = np.array([10, 20, 30, 40])
        
        reg_set = regressor_set(
            onsets=onsets,
            fac=fac,
            hrf="spmg1"
        )
        
        # Should have 2 regressors (only for levels that exist)
        assert len(reg_set.regressors) == 2
        
        # Check levels are correct
        assert reg_set.levels == ['0', '2']


class TestBoxcarHRFEquivalence:
    """Test boxcar_hrf matches R hrf_boxcar behaviour."""

    def test_basic_boxcar(self):
        """R: hrf_boxcar(width=5) -> amplitude 1 for 0<=t<5, else 0."""
        from fmrimod.hrf.functions import boxcar_hrf
        t = np.array([-1, 0, 2.5, 4.999, 5, 6])
        y = boxcar_hrf(t, width=5.0)
        expected = np.array([0, 1, 1, 1, 0, 0], dtype=np.float64)
        np.testing.assert_array_equal(y, expected)

    def test_normalized_boxcar(self):
        """R: hrf_boxcar(width=5, normalize=TRUE) -> amplitude = 1/5.

        ``normalize=True`` retired by bd-01KRGCZ6QJME1JD8FD5D4PGC04; the
        equivalent Python composition is ``amplitude=1/width`` explicitly.
        """
        from fmrimod.hrf.functions import boxcar_hrf
        t = np.array([0, 2.5])
        y = boxcar_hrf(t, width=5.0, amplitude=1.0 / 5.0)
        np.testing.assert_allclose(y, [0.2, 0.2])

    def test_boxcar_as_hrf_span(self):
        """R: as_hrf wraps boxcar with span=width."""
        from fmrimod.hrf.generators import boxcar_generator
        hrf = boxcar_generator(width=7)
        assert hrf.span == 7.0
        assert hrf.nbasis == 1


class TestWeightedHRFEquivalence:
    """Test weighted_hrf matches R hrf_weighted behaviour."""

    def test_width_weights_constant(self):
        """R: hrf_weighted(width=6, weights=c(0.2,0.5,0.8,0.3), method='constant')."""
        from fmrimod.hrf.functions import weighted_hrf
        # R generates times = seq(0, 6, length.out=4) = [0, 2, 4, 6]
        # approxfun(method='constant') gives step function
        t = np.array([0, 1, 2, 3, 4, 5, 6])
        y = weighted_hrf(t, weights=[0.2, 0.5, 0.8, 0.3], width=6.0,
                         method="constant")
        # At t=0 -> 0.2, t=1 -> 0.2 (still in first bin), t=2 -> 0.5, etc.
        assert y[0] == pytest.approx(0.2)
        assert y[2] == pytest.approx(0.5)
        assert y[4] == pytest.approx(0.8)

    def test_times_weights_linear(self):
        """R: hrf_weighted(times=c(0,1,3,5,6), weights=c(0.1,0.5,0.8,0.5,0.1), method='linear')."""
        from fmrimod.hrf.functions import weighted_hrf
        t = np.array([0, 3, 6])
        y = weighted_hrf(t, weights=[0.1, 0.5, 0.8, 0.5, 0.1],
                         times=[0, 1, 3, 5, 6], method="linear")
        np.testing.assert_allclose(y, [0.1, 0.8, 0.1], atol=1e-10)

    def test_weighted_span(self):
        """R: span = max(times)."""
        from fmrimod.hrf.generators import weighted_generator
        hrf = weighted_generator(weights=[1, 2, 1], width=8.0)
        assert hrf.span == 8.0


class TestTentGeneratorEquivalence:
    """Test tent_generator matches R hrf_tent_generator."""

    def test_tent_is_degree1_bspline(self):
        """R: hrf_tent_generator calls hrf_bspline(t, span, N, degree=1)."""
        from fmrimod.hrf.functions import bspline_hrf
        from fmrimod.hrf.generators import tent_generator
        t = np.linspace(0, 24, 200)
        tent = tent_generator(nbasis=5, span=24)
        tent_vals = tent(t)
        bspline_vals = bspline_hrf(t, n_basis=5, degree=1, span=24)
        np.testing.assert_allclose(tent_vals, bspline_vals, atol=1e-12)

    def test_tent_attributes(self):
        """R: class(obj) <- c('Tent_HRF', class(obj))."""
        from fmrimod.hrf.generators import tent_generator
        hrf = tent_generator(nbasis=6, span=20)
        assert hrf.nbasis == 6
        assert hrf.span == 20.0
        assert "tent" in hrf.name.lower()


class TestTrialVaryingEquivalence:
    """Test trial-varying regressor matches R list-of-HRFs behaviour."""

    def test_list_hrf_evaluation(self):
        """R: regressor(onsets=c(0,20), hrf=list(hrf_short, hrf_long))."""
        from fmrimod.hrf.generators import boxcar_generator
        h_short = boxcar_generator(width=2)
        h_long = boxcar_generator(width=8)
        reg = regressor(onsets=[0, 20], hrf=[h_short, h_long])

        grid = np.arange(0, 40, 0.5)
        result = reg.evaluate(grid, precision=0.1)
        assert result.shape == (len(grid),)

        # Event at t=0 with width=2: response only in [0, 2)
        # Event at t=20 with width=8: response in [20, 28)
        r_event1 = result[(grid >= 0) & (grid < 2)]
        r_event2 = result[(grid >= 20) & (grid < 28)]
        assert np.all(r_event1 > 0)
        assert np.all(r_event2 > 0)

        # Outside these windows, near the events, should be zero
        r_gap = result[(grid >= 3) & (grid < 19)]
        np.testing.assert_allclose(r_gap, 0, atol=1e-10)

    def test_length1_list_recycling(self):
        """R: length-1 list is recycled to all events."""
        from fmrimod.hrf.generators import boxcar_generator
        h = boxcar_generator(width=5)
        reg = regressor(onsets=[10, 20, 30], hrf=[h])
        assert len(reg.hrf) == 3
        for hrf_i in reg.hrf:
            assert hrf_i.span == 5.0
