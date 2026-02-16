"""Tests for as_hrf function - mirroring and extending R tests."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from fmrimod import as_hrf, get_hrf
from fmrimod.hrf_dispatch import ArrayHRF, FunctionHRF, DictHRF


class TestAsHRF:
    """Test as_hrf conversion function."""
    
    def test_as_hrf_from_string(self):
        """Test converting string name to HRF."""
        hrf = as_hrf('spm')
        assert hasattr(hrf, 'evaluate')
        assert hasattr(hrf, 'name')
        assert hasattr(hrf, 'nbasis')
        assert 'spm' in hrf.name.lower()
    
    def test_as_hrf_from_existing_hrf(self):
        """Test that existing HRF is returned as-is."""
        original = get_hrf('spm')
        result = as_hrf(original)
        assert result is original  # Same object
    
    def test_as_hrf_from_array(self):
        """Test converting array to HRF."""
        hrf_values = np.array([0, 0.5, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0])
        hrf = as_hrf(hrf_values)
        
        assert isinstance(hrf, ArrayHRF)
        assert hrf.name == 'array_hrf'
        assert hrf.nbasis == 1
        
        # Test evaluation
        t = np.arange(len(hrf_values))
        result = hrf.evaluate(t)
        assert_array_almost_equal(result, hrf_values)
    
    def test_as_hrf_from_list(self):
        """Test converting list to HRF."""
        hrf_values = [0, 0.5, 1.0, 0.5, 0]
        hrf = as_hrf(hrf_values)
        
        assert isinstance(hrf, ArrayHRF)
        assert hrf.nbasis == 1
    
    def test_as_hrf_from_function(self):
        """Test converting function to HRF."""
        def gamma_hrf(t):
            """Simple gamma function."""
            return t * np.exp(-t / 2)
        
        hrf = as_hrf(gamma_hrf)
        
        assert isinstance(hrf, FunctionHRF)
        assert hrf.name == 'gamma_hrf'
        assert hrf.nbasis == 1
        
        # Test evaluation
        t = np.linspace(0, 10, 50)
        result = hrf.evaluate(t)
        expected = gamma_hrf(t)
        assert_array_almost_equal(result, expected)
    
    def test_as_hrf_from_lambda(self):
        """Test converting lambda to HRF."""
        hrf = as_hrf(lambda t: np.sin(t))
        
        assert isinstance(hrf, FunctionHRF)
        assert hrf.name == '<lambda>'
        
        # Test evaluation
        t = np.array([0, np.pi/2, np.pi])
        result = hrf.evaluate(t)
        expected = np.array([0, 1, 0])
        assert_array_almost_equal(result, expected, decimal=5)
    
    def test_as_hrf_from_dict(self):
        """Test converting dict to HRF."""
        def my_evaluate(t):
            return np.exp(-t)
        
        hrf_dict = {
            'evaluate': my_evaluate,
            'name': 'exp_decay',
            'nbasis': 2,
            'custom_param': 42
        }
        
        hrf = as_hrf(hrf_dict)
        
        assert isinstance(hrf, DictHRF)
        assert hrf.name == 'exp_decay'
        assert hrf.nbasis == 2
        assert hrf.custom_param == 42
        
        # Test evaluation
        t = np.array([0, 1, 2])
        result = hrf.evaluate(t)
        expected = np.exp(-t)
        assert_array_almost_equal(result, expected)
    
    def test_as_hrf_dict_missing_evaluate(self):
        """Test that dict without 'evaluate' raises error."""
        bad_dict = {'name': 'bad', 'nbasis': 1}
        
        with pytest.raises(ValueError, match="evaluate"):
            as_hrf(bad_dict)
    
    def test_as_hrf_with_kwargs(self):
        """Test passing kwargs to HRF constructors."""
        # Array with custom sampling rate
        hrf_array = np.array([0, 1, 0])
        hrf = as_hrf(hrf_array, sampling_rate=2.0, name='custom')
        
        assert hrf.name == 'custom'
        assert hrf.sampling_rate == 2.0
        
        # Function with custom nbasis
        hrf = as_hrf(lambda t: t, nbasis=3)
        assert hrf.nbasis == 3
    
    def test_as_hrf_invalid_type(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError, match="Cannot convert"):
            as_hrf(123)  # int
        
        with pytest.raises(TypeError, match="Cannot convert"):
            as_hrf(set([1, 2, 3]))  # set
    
    def test_array_hrf_interpolation(self):
        """Test ArrayHRF interpolation behavior."""
        # Create simple triangular HRF
        hrf_values = np.array([0, 0.5, 1.0, 0.5, 0])
        hrf = ArrayHRF(hrf_values, sampling_rate=1.0)
        
        # Test at original points
        t = np.array([0, 1, 2, 3, 4])
        result = hrf.evaluate(t)
        assert_array_almost_equal(result, hrf_values)
        
        # Test interpolation
        t_interp = np.array([0.5, 1.5, 2.5, 3.5])
        result_interp = hrf.evaluate(t_interp)
        expected = np.array([0.25, 0.75, 0.75, 0.25])
        assert_array_almost_equal(result_interp, expected)
        
        # Test extrapolation (should be 0)
        t_extrap = np.array([-1, 5, 10])
        result_extrap = hrf.evaluate(t_extrap)
        assert_array_almost_equal(result_extrap, np.zeros(3))
    
    def test_array_hrf_different_sampling_rates(self):
        """Test ArrayHRF with different sampling rates."""
        hrf_values = np.array([0, 1, 0])
        
        # SR = 0.5 (one sample every 2 seconds)
        hrf1 = ArrayHRF(hrf_values, sampling_rate=0.5)
        t = np.array([0, 2, 4])  # Actual time points
        result1 = hrf1.evaluate(t)
        assert_array_almost_equal(result1, hrf_values)
        
        # SR = 2.0 (two samples per second)
        hrf2 = ArrayHRF(hrf_values, sampling_rate=2.0)
        t = np.array([0, 0.5, 1.0])  # Actual time points
        result2 = hrf2.evaluate(t)
        assert_array_almost_equal(result2, hrf_values)
    
    def test_function_hrf_preserves_name(self):
        """Test that FunctionHRF preserves function name."""
        def double_gamma_hrf(t):
            """Double gamma HRF."""
            return t * np.exp(-t)
        
        hrf = as_hrf(double_gamma_hrf)
        assert hrf.name == 'double_gamma_hrf'
    
    def test_dict_hrf_minimal(self):
        """Test DictHRF with minimal specification."""
        hrf_dict = {'evaluate': lambda t: t**2}
        hrf = as_hrf(hrf_dict)
        
        assert hrf.name == 'dict_hrf'  # Default name
        assert hrf.nbasis == 1  # Default nbasis
        
        # Test evaluation
        t = np.array([1, 2, 3])
        result = hrf.evaluate(t)
        assert_array_almost_equal(result, t**2)
    
    def test_hrf_protocol_compliance(self):
        """Test that all converted HRFs comply with HRFProtocol."""
        test_cases = [
            'spm',
            np.array([0, 1, 0]),
            lambda t: t,
            {'evaluate': lambda t: t, 'name': 'test', 'nbasis': 2}
        ]
        
        for case in test_cases:
            hrf = as_hrf(case)
            
            # Check required attributes
            assert hasattr(hrf, 'name')
            assert hasattr(hrf, 'nbasis')
            assert hasattr(hrf, 'evaluate')
            
            # Check that evaluate is callable
            assert callable(hrf.evaluate)
            
            # Check that properties work
            assert isinstance(hrf.name, str)
            assert isinstance(hrf.nbasis, int)
            
            # Check that evaluate returns array
            t = np.array([0, 1, 2])
            result = hrf.evaluate(t)
            assert isinstance(result, np.ndarray)