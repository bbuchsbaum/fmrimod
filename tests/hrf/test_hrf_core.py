"""Tests for HRF core functionality."""

import numpy as np
import pytest

from fmrimod.hrf.core import HRF, FunctionHRF, as_hrf, bind_basis
from fmrimod.hrf.library import (
    SPM_CANONICAL, SPM_WITH_DERIVATIVE, SPM_WITH_DISPERSION,
    GAMMA_HRF, GAUSSIAN_HRF, BSPLINE_HRF, FIR_HRF
)


class TestHRFBase:
    """Test base HRF functionality."""
    
    def test_spm_canonical_structure(self):
        """Test SPM canonical HRF has correct structure."""
        assert isinstance(SPM_CANONICAL, HRF)
        assert SPM_CANONICAL.name == "SPMG1"
        assert SPM_CANONICAL.nbasis == 1
        assert SPM_CANONICAL.span == 24.0
        assert SPM_CANONICAL.param_names == ["p1", "p2", "a1"]
        assert SPM_CANONICAL.params == {"p1": 5.0, "p2": 15.0, "a1": 0.0833}
    
    def test_spm_canonical_evaluation(self, time_grid):
        """Test SPM canonical HRF evaluation."""
        result = SPM_CANONICAL(time_grid)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == time_grid.shape
        assert np.all(np.isfinite(result))
        
        # Check negative times return zero
        neg_times = np.array([-5, -2, -1])
        neg_result = SPM_CANONICAL(neg_times)
        assert np.allclose(neg_result, 0)
        
        # Check approximate peak timing (should be around 5-6 seconds)
        peak_idx = np.argmax(result)
        peak_time = time_grid[peak_idx]
        assert 4 <= peak_time <= 7
    
    def test_spm_with_derivative(self, time_grid):
        """Test SPM with temporal derivative."""
        assert SPM_WITH_DERIVATIVE.name == "SPMG2"
        assert SPM_WITH_DERIVATIVE.nbasis == 2
        
        result = SPM_WITH_DERIVATIVE(time_grid)
        assert result.shape == (len(time_grid), 2)
        
        # Test single time point returns proper shape
        single_result = SPM_WITH_DERIVATIVE(5.0)
        assert single_result.shape == (1, 2)
    
    def test_spm_with_dispersion(self, time_grid):
        """Test SPM with dispersion derivative."""
        assert SPM_WITH_DISPERSION.name == "SPMG3"
        assert SPM_WITH_DISPERSION.nbasis == 3
        
        result = SPM_WITH_DISPERSION(time_grid)
        assert result.shape == (len(time_grid), 3)
    
    def test_gamma_hrf(self, time_grid):
        """Test Gamma HRF."""
        assert GAMMA_HRF.name == "gamma"
        assert GAMMA_HRF.param_names == ["shape", "rate"]
        
        result = GAMMA_HRF(time_grid)
        assert result.shape == time_grid.shape
        assert np.all(result >= 0)  # Gamma should be non-negative
    
    def test_gaussian_hrf(self, time_grid):
        """Test Gaussian HRF."""
        assert GAUSSIAN_HRF.name == "gaussian"
        assert GAUSSIAN_HRF.param_names == ["mean", "sd"]
        
        result = GAUSSIAN_HRF(time_grid)
        assert result.shape == time_grid.shape
        assert np.all(result >= 0)  # Gaussian PDF is non-negative
    
    def test_bspline_hrf(self, time_grid):
        """Test B-spline HRF."""
        assert BSPLINE_HRF.name == "bspline"
        assert BSPLINE_HRF.nbasis == 5
        
        result = BSPLINE_HRF(time_grid)
        assert result.shape == (len(time_grid), 5)
    
    def test_evaluate_with_duration(self):
        """Test HRF evaluation with block duration."""
        t = np.arange(0, 20, 0.2)
        
        # Zero duration should match direct evaluation
        result_zero = SPM_CANONICAL.evaluate(t, duration=0)
        result_direct = SPM_CANONICAL(t)
        assert np.allclose(result_zero, result_direct)
        
        # Non-zero duration should produce larger response
        result_block = SPM_CANONICAL.evaluate(t, duration=2)
        assert np.max(result_block) > np.max(result_direct)
        
        # Test summation vs max
        result_sum = SPM_CANONICAL.evaluate(t, duration=2, summate=True)
        result_max = SPM_CANONICAL.evaluate(t, duration=2, summate=False)
        assert not np.allclose(result_sum, result_max)
        
        # Test normalization
        result_norm = SPM_CANONICAL.evaluate(t, duration=2, normalize=True)
        assert np.abs(np.max(np.abs(result_norm)) - 1.0) < 1e-7
    
    def test_evaluate_precision(self):
        """Test evaluation precision parameter."""
        t = np.arange(0, 20, 0.2)
        
        result_fine = SPM_CANONICAL.evaluate(t, duration=2, precision=0.1)
        result_coarse = SPM_CANONICAL.evaluate(t, duration=2, precision=0.5)
        
        # Results should be similar but not identical
        assert not np.array_equal(result_fine, result_coarse)
        assert np.corrcoef(result_fine, result_coarse)[0, 1] > 0.99


class TestAsHRF:
    """Test as_hrf function."""
    
    def test_simple_function(self):
        """Test converting simple function to HRF."""
        def my_func(t):
            return t ** 2
        
        hrf_obj = as_hrf(my_func, name="test_sq", nbasis=1, span=10,
                        params={"power": 2})
        
        assert isinstance(hrf_obj, HRF)
        assert hrf_obj.name == "test_sq"
        assert hrf_obj.nbasis == 1
        assert hrf_obj.span == 10
        assert hrf_obj.param_names == ["power"]
        assert hrf_obj.params == {"power": 2}
        
        # Test evaluation
        assert hrf_obj(5) == 25
        assert np.array_equal(hrf_obj(np.array([1, 2, 3])), np.array([1, 4, 9]))
    
    def test_defaults(self):
        """Test as_hrf with default parameters."""
        def my_func(t):
            return np.sin(t)
        
        hrf_obj = as_hrf(my_func)
        assert hrf_obj.name == "my_func"
        assert hrf_obj.nbasis == 1
        assert hrf_obj.span == 24.0
        assert hrf_obj.params == {}
        assert hrf_obj.param_names is None
    
    def test_multi_basis_function(self):
        """Test as_hrf with multi-basis function."""
        def my_multi_func(t):
            t = np.asarray(t)
            return np.column_stack([t, t**2])
        
        hrf_obj = as_hrf(my_multi_func, nbasis=2)
        assert hrf_obj.nbasis == 2
        
        result = hrf_obj(3)
        assert np.array_equal(result, np.array([[3, 9]]))


class TestBindBasis:
    """Test bind_basis function."""
    
    def test_combine_single_basis(self):
        """Test combining single-basis HRFs."""
        f1 = lambda t: t
        f2 = lambda t: t**2
        f3 = lambda t: np.ones_like(t)
        
        hrf1 = as_hrf(f1, name="linear", span=10)
        hrf2 = as_hrf(f2, name="quadratic", span=12)
        hrf3 = as_hrf(f3, name="constant", span=8)
        
        combined = bind_basis(hrf1, hrf2, hrf3)
        
        assert combined.name == "linear + quadratic + constant"
        assert combined.nbasis == 3
        assert combined.span == 12  # max(10, 12, 8)
        
        # Test evaluation
        t_vals = np.array([0, 1, 2, 5])
        result = combined(t_vals)
        expected = np.column_stack([f1(t_vals), f2(t_vals), f3(t_vals)])
        assert np.array_equal(result, expected)
    
    def test_combine_with_multi_basis(self):
        """Test combining with multi-basis HRF."""
        f1 = lambda t: t
        f_multi = lambda t: np.column_stack([np.sin(t), np.cos(t)])
        
        hrf1 = as_hrf(f1, name="linear", nbasis=1, span=10)
        hrf_multi = as_hrf(f_multi, name="trig", nbasis=2, span=15)
        
        combined = bind_basis(hrf1, hrf_multi)
        
        assert combined.nbasis == 3
        assert combined.span == 15
        assert combined.name == "linear + trig"
        
        t_vals = np.array([0, 1, 2])
        result = combined(t_vals)
        assert result.shape == (3, 3)
    
    def test_single_hrf(self):
        """Test bind_basis with single HRF."""
        hrf = as_hrf(lambda t: t**2, name="square")
        bound = bind_basis(hrf)
        
        assert bound.name == "square"
        assert bound.nbasis == 1
        assert np.array_equal(bound(5), hrf(5))
    
    def test_empty_input(self):
        """Test bind_basis with no arguments."""
        with pytest.raises(ValueError, match="At least one HRF"):
            bind_basis()


class TestHRFFromCoefficients:
    """Test HRF from_coefficients method."""
    
    def test_single_basis(self):
        """Test from_coefficients with single basis."""
        weighted = SPM_CANONICAL.from_coefficients([2.5])
        
        t = np.array([0, 5, 10])
        expected = 2.5 * SPM_CANONICAL(t)
        assert np.allclose(weighted(t), expected)
    
    def test_multi_basis(self):
        """Test from_coefficients with multi-basis HRF."""
        coeffs = np.array([1.0, -0.5, 0.3])
        weighted = SPM_WITH_DISPERSION.from_coefficients(coeffs)
        
        t = np.array([0, 5, 10])
        basis_vals = SPM_WITH_DISPERSION(t)
        expected = basis_vals @ coeffs
        assert np.allclose(weighted(t), expected)
    
    def test_wrong_number_coefficients(self):
        """Test error with wrong number of coefficients."""
        with pytest.raises(ValueError, match="Number of coefficients"):
            SPM_WITH_DERIVATIVE.from_coefficients([1.0])  # Needs 2 coeffs


class TestHRFValidation:
    """Test HRF input validation."""
    
    def test_evaluate_empty_grid(self):
        """Test evaluate with empty grid."""
        with pytest.raises(ValueError, match="grid must contain"):
            SPM_CANONICAL.evaluate(np.array([]))
    
    def test_evaluate_nan_grid(self):
        """Test evaluate with NaN in grid."""
        with pytest.raises(ValueError, match="cannot contain NaN"):
            SPM_CANONICAL.evaluate(np.array([0, 5, np.nan, 10]))
    
    def test_evaluate_invalid_precision(self):
        """Test evaluate with invalid precision."""
        t = np.array([0, 5, 10])
        
        with pytest.raises(ValueError, match="precision must be positive"):
            SPM_CANONICAL.evaluate(t, precision=0)
        
        with pytest.raises(ValueError, match="precision must be positive"):
            SPM_CANONICAL.evaluate(t, precision=-0.5)


class TestHRFStringRepresentation:
    """Test HRF string representations."""
    
    def test_str_representation(self):
        """Test __str__ method."""
        str_repr = str(SPM_CANONICAL)
        assert "HRF(name='SPMG1'" in str_repr
        assert "nbasis=1" in str_repr
        assert "span=24.0" in str_repr
        assert "p1=5.0" in str_repr
    
    def test_repr(self):
        """Test __repr__ method."""
        repr_str = repr(GAMMA_HRF)
        assert "HRF(name='gamma'" in repr_str
        assert "shape=6.0" in repr_str
        assert "rate=1.0" in repr_str
