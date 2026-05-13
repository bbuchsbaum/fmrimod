"""Tests for HRF generators."""

import numpy as np
import pytest

from fmrimod.hrf.core import HRF
from fmrimod.hrf.functions import bspline_hrf, gamma_hrf
from fmrimod.hrf.generators import (
    bspline_generator,
    daguerre_generator,
    fir_generator,
    fourier_generator,
    gen_hrf,
    gen_hrf_set,
    make_hrf,
)
from fmrimod.hrf.library import GAMMA_HRF, SPM_CANONICAL


class TestGenHRF:
    """Test gen_hrf function."""
    
    def test_gen_hrf_from_hrf_object(self):
        """Test gen_hrf with HRF object input."""
        result = gen_hrf(SPM_CANONICAL)
        
        # Should return the same HRF
        assert result is SPM_CANONICAL
    
    def test_gen_hrf_with_lag(self):
        """Test gen_hrf with lag parameter."""
        lag = 2.0
        result = gen_hrf(SPM_CANONICAL, lag=lag)
        
        assert "_lag(2.0)" in result.name
        assert result.span == SPM_CANONICAL.span + lag
        
        # Check lagged evaluation
        t = np.arange(0, 20, 0.5)
        assert np.allclose(result(t), SPM_CANONICAL(t - lag))
    
    def test_gen_hrf_with_width(self):
        """Test gen_hrf with width parameter."""
        width = 3.0
        result = gen_hrf(SPM_CANONICAL, width=width)
        
        assert "_block(w=3.0)" in result.name
        assert result.span == SPM_CANONICAL.span + width
    
    def test_gen_hrf_with_lag_and_width(self):
        """Test gen_hrf with both lag and width."""
        lag = 1.5
        width = 2.5
        result = gen_hrf(SPM_CANONICAL, lag=lag, width=width)
        
        # Both decorators should be applied
        assert "_lag(" in result.name
        assert "_block(" in result.name
        assert result.span > SPM_CANONICAL.span + lag + width - 0.1  # Some tolerance
    
    def test_gen_hrf_with_normalize(self):
        """Test gen_hrf with normalization."""
        result = gen_hrf(SPM_CANONICAL, normalize=True)
        
        t = np.arange(0, 20, 0.1)
        values = result(t)
        assert abs(np.max(np.abs(values)) - 1.0) < 0.001  # Allow 0.1% tolerance
    
    def test_gen_hrf_from_function(self):
        """Test gen_hrf with function input."""
        result = gen_hrf(gamma_hrf)
        
        assert isinstance(result, HRF)
        assert result.nbasis == 1
        
        # Should evaluate like the original function
        t = np.arange(0, 20, 0.5)
        assert np.allclose(result(t), gamma_hrf(t))
    
    def test_gen_hrf_from_string(self):
        """Test gen_hrf with string (registry) input."""
        result = gen_hrf("gamma")
        
        assert isinstance(result, HRF)
        assert result.name == "gamma"
        
        # Should match the pre-defined HRF
        t = np.arange(0, 20, 0.5)
        assert np.allclose(result(t), GAMMA_HRF(t))
    
    def test_gen_hrf_with_kwargs(self):
        """Test gen_hrf passing kwargs to generators."""
        # This will work once registry is updated
        # For now, test with direct function
        result = gen_hrf(bspline_hrf, n_basis=7, degree=3)

        assert isinstance(result, HRF)
        assert result.nbasis == 7  # Should detect from sampling

        t = np.arange(0, 20, 0.5)
        expected = bspline_hrf(t, n_basis=7, degree=3)
        assert np.allclose(result(t), expected)


class TestGenHRFSet:
    """Test gen_hrf_set function."""
    
    def test_gen_hrf_set_basic(self):
        """Test basic HRF set creation."""
        hrf1 = SPM_CANONICAL
        hrf2 = GAMMA_HRF
        
        result = gen_hrf_set(hrf1, hrf2)
        
        assert isinstance(result, HRF)
        assert result.nbasis == 2
        assert "SPMG1 + gamma" in result.name
    
    def test_gen_hrf_set_with_functions(self):
        """Test with function inputs."""
        def hrf1(t):
            return np.exp(-t/3) * (t > 0)
        
        def hrf2(t):
            return np.exp(-t/5) * (t > 0)
        
        result = gen_hrf_set(hrf1, hrf2)
        
        assert result.nbasis == 2
        
        t = np.arange(0, 20, 0.5)
        values = result(t)
        assert np.allclose(values[:, 0], hrf1(t))
        assert np.allclose(values[:, 1], hrf2(t))
    
    def test_gen_hrf_set_custom_name(self):
        """Test custom naming."""
        result = gen_hrf_set(SPM_CANONICAL, GAMMA_HRF, name="my_set")
        
        assert result.name == "my_set"
    
    def test_gen_hrf_set_single_input(self):
        """Test with single HRF."""
        result = gen_hrf_set(SPM_CANONICAL)
        
        assert result.nbasis == 1
        t = np.arange(0, 20, 0.5)
        assert np.allclose(result(t), SPM_CANONICAL(t))


class TestBasisGenerators:
    """Test basis function generators."""
    
    def test_bspline_generator(self):
        """Test B-spline generator."""
        N = 7
        degree = 3
        span = 25.0

        hrf = bspline_generator(n_basis=N, degree=degree, span=span)

        assert isinstance(hrf, HRF)
        assert hrf.nbasis == N
        assert hrf.span == span
        assert hrf.params["n_basis"] == N
        assert hrf.params["degree"] == degree

        # Test evaluation
        t = np.arange(0, span, 0.5)
        result = hrf(t)
        assert result.shape == (len(t), N)

        # Should match direct function call
        expected = bspline_hrf(t, n_basis=N, degree=degree, span=span)
        assert np.allclose(result, expected)
    
    def test_fir_generator(self):
        """Test FIR generator."""
        N = 12
        span = 24.0

        hrf = fir_generator(n_basis=N, span=span)

        assert isinstance(hrf, HRF)
        assert hrf.nbasis == N
        assert hrf.span == span
        assert hrf.params["n_basis"] == N

        # Test evaluation
        t = np.arange(0, span, 0.5)
        result = hrf(t)
        assert result.shape == (len(t), N)

        # Check basis properties
        # Each time point should be in exactly one bin
        for i in range(len(t)):
            assert np.sum(result[i, :]) <= 1.0
    
    def test_fourier_generator(self):
        """Test Fourier generator."""
        N = 7
        span = 24.0

        hrf = fourier_generator(n_basis=N, span=span)

        assert isinstance(hrf, HRF)
        assert hrf.nbasis == N
        assert hrf.span == span

        # Test evaluation
        t = np.arange(0, span, 0.5)
        result = hrf(t)
        assert result.shape == (len(t), N)

        # First basis should be sine (alternating sin/cos pattern)
        # Verify it's not constant
        assert not np.allclose(result[:, 0], result[0, 0])

        # Verify values outside [0, span] are zeroed
        t_outside = np.array([-1, span + 1])
        result_outside = hrf(t_outside)
        assert np.allclose(result_outside, 0)

    def test_fourier_generator_even_n(self):
        """Test Fourier generator works with even N (R-compatible)."""
        N = 6
        span = 24.0

        hrf = fourier_generator(n_basis=N, span=span)

        assert isinstance(hrf, HRF)
        assert hrf.nbasis == N

        # Test evaluation
        t = np.arange(0, span, 0.5)
        result = hrf(t)
        assert result.shape == (len(t), N)
    
    def test_daguerre_generator(self):
        """Test Daguerre generator."""
        N = 7
        span = 20.0

        hrf = daguerre_generator(n_basis=N, span=span)

        assert isinstance(hrf, HRF)
        assert hrf.nbasis == N
        assert hrf.span == span
        assert "daguerre" in hrf.name

        # Test evaluation
        t = np.arange(0, span, 0.5)
        result = hrf(t)
        assert result.shape == (len(t), N)
    
    def test_generators_with_custom_names(self):
        """Test generators with custom names."""
        hrf1 = bspline_generator(n_basis=5, name="my_bspline")
        hrf2 = fir_generator(n_basis=10, name="my_fir")
        hrf3 = fourier_generator(n_basis=5, name="my_fourier")
        hrf4 = daguerre_generator(n_basis=7, name="my_daguerre")

        assert hrf1.name == "my_bspline"
        assert hrf2.name == "my_fir"
        assert hrf3.name == "my_fourier"
        assert hrf4.name == "my_daguerre"


class TestMakeHRF:
    """Test make_hrf function."""
    
    def test_make_hrf_simple_string(self):
        """Test make_hrf with simple string."""
        hrf = make_hrf("spmg1")
        
        assert isinstance(hrf, HRF)
        
        # Should match pre-defined HRF
        t = np.arange(0, 20, 0.5)
        assert np.allclose(hrf(t), SPM_CANONICAL(t))
    
    def test_make_hrf_with_params_string_retired(self):
        """Parameterized string DSL is retired in favor of typed constructors."""
        with pytest.raises(ValueError, match="string-DSL HRF specs.*retired"):
            make_hrf("bspline(n_basis=7, degree=3)")

        hrf = bspline_generator(n_basis=7, degree=3)
        assert isinstance(hrf, HRF)
        assert hrf.nbasis == 7
    
    def test_make_hrf_dict_spec(self):
        """Test make_hrf with dictionary specification."""
        spec = {
            "type": "gamma",
            "shape": 5.0,
            "rate": 0.9
        }
        
        hrf = make_hrf(spec)
        
        assert isinstance(hrf, HRF)
        
        # Should use specified parameters
        t = np.arange(0, 20, 0.5)
        expected = gamma_hrf(t, shape=5.0, rate=0.9)
        assert np.allclose(hrf(t), expected)
    
    def test_make_hrf_with_lag(self):
        """Test make_hrf with lag."""
        hrf = make_hrf("spmg1", lag=2.0)
        
        assert "_lag(2.0)" in hrf.name
        
        t = np.arange(0, 20, 0.5)
        expected = SPM_CANONICAL(t - 2.0)
        assert np.allclose(hrf(t), expected)
    
    def test_make_hrf_with_normalize(self):
        """Test make_hrf with normalization."""
        hrf = make_hrf("gamma", normalize=True)
        
        t = np.arange(0, 20, 0.1)
        values = hrf(t)
        assert abs(np.max(np.abs(values)) - 1.0) < 0.001  # Allow 0.1% tolerance

    def test_make_hrf_with_hrf_object(self):
        """Test make_hrf accepts existing HRF objects directly."""
        hrf = make_hrf(SPM_CANONICAL, lag=1.0)

        assert isinstance(hrf, HRF)
        assert "_lag(1.0)" in hrf.name
        assert hrf.span == SPM_CANONICAL.span + 1.0
