"""Tests for HRF decorators."""

import numpy as np
import pytest

from fmrimod.hrf.core import HRF, BoundBasisHRF, FunctionHRF, as_hrf
from fmrimod.hrf.decorators import (
    block_hrf,
    gen_hrf_blocked,
    gen_hrf_lagged,
    lag_hrf,
)
from fmrimod.hrf.library import SPM_CANONICAL, SPM_WITH_DERIVATIVE
from fmrimod.hrf.normalization import normalize as _normalize


def _normalize_per_basis(hrf):
    """Test-only adapter: matches the retired decorator's per-basis semantic."""
    return _normalize(hrf, "unit_peak_per_basis")


def _trapezoid_compat(y, x):
    """Integrate with NumPy API compatibility across versions."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


class TestLagHRF:
    """Test lag_hrf decorator."""
    
    def test_lag_hrf_basic(self, time_grid):
        """Test basic lag functionality."""
        lag_amount = 2.0
        lagged = lag_hrf(SPM_CANONICAL, lag_amount)
        
        # Check attributes
        assert isinstance(lagged, HRF)
        assert lagged.nbasis == SPM_CANONICAL.nbasis
        assert lagged.span == SPM_CANONICAL.span + lag_amount
        assert "_lag(2.0)" in lagged.name
        assert lagged.lag == lag_amount
        
        # Check function evaluation
        t = time_grid
        result_lagged = lagged(t)
        result_direct = SPM_CANONICAL(t - lag_amount)
        assert np.allclose(result_lagged, result_direct)
    
    def test_lag_hrf_peak_shift(self):
        """Test that peak is shifted by lag amount."""
        t = np.arange(0, 20, 0.1)
        lag_amount = 3.0
        
        original = SPM_CANONICAL(t)
        lagged = lag_hrf(SPM_CANONICAL, lag_amount)
        result_lagged = lagged(t)
        
        # Find peaks
        peak_original = t[np.argmax(original)]
        peak_lagged = t[np.argmax(result_lagged)]
        
        # Peak should be shifted by lag amount (within discretization tolerance)
        assert abs((peak_lagged - peak_original) - lag_amount) < 0.2
    
    def test_lag_hrf_zero(self):
        """Test lag with zero amount."""
        t = np.arange(0, 20, 0.5)
        lagged_zero = lag_hrf(SPM_CANONICAL, 0)
        
        assert np.allclose(lagged_zero(t), SPM_CANONICAL(t))
        assert lagged_zero.span == SPM_CANONICAL.span
    
    def test_lag_hrf_multi_basis(self):
        """Test lag with multi-basis HRF."""
        t = np.arange(0, 20, 0.5)
        lag_amount = 2.5
        
        lagged_multi = lag_hrf(SPM_WITH_DERIVATIVE, lag_amount)
        
        assert lagged_multi.nbasis == SPM_WITH_DERIVATIVE.nbasis
        assert np.allclose(
            lagged_multi(t),
            SPM_WITH_DERIVATIVE(t - lag_amount)
        )
    
    def test_lag_hrf_validation(self):
        """Test input validation."""
        with pytest.raises(ValueError, match="finite"):
            lag_hrf(SPM_CANONICAL, np.inf)
        
        with pytest.raises(ValueError, match="finite"):
            lag_hrf(SPM_CANONICAL, np.nan)


class TestBlockHRF:
    """Test block_hrf decorator."""

    def test_trapezoid_compat_helper(self):
        """Test integration helper returns expected area."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = x.copy()
        area = _trapezoid_compat(y, x)
        assert area == pytest.approx(4.5)
    
    def test_block_hrf_basic(self):
        """Test basic block functionality."""
        width = 3.0
        blocked = block_hrf(SPM_CANONICAL, width)
        
        # Check attributes
        assert isinstance(blocked, HRF)
        assert blocked.nbasis == SPM_CANONICAL.nbasis
        assert blocked.span == SPM_CANONICAL.span + width
        assert "_block(w=3.0)" in blocked.name
        assert blocked.width == width
    
    def test_block_hrf_increases_response(self):
        """Test that blocking increases response magnitude."""
        t = np.arange(0, 30, 0.2)
        width = 5.0
        
        original = SPM_CANONICAL(t)
        blocked = block_hrf(SPM_CANONICAL, width)
        result_blocked = blocked(t)
        
        # Blocked response should have larger integral
        assert _trapezoid_compat(result_blocked, t) > _trapezoid_compat(original, t)
    
    def test_block_hrf_negligible_width(self):
        """Test that negligible width returns original."""
        t = np.arange(0, 20, 0.5)
        blocked_tiny = block_hrf(SPM_CANONICAL, width=0.01, precision=0.1)
        
        # Should be essentially unchanged
        assert np.allclose(blocked_tiny(t), SPM_CANONICAL(t), rtol=1e-4)
    
    def test_block_hrf_with_decay(self):
        """Test block with exponential decay."""
        t = np.arange(0, 30, 0.2)
        width = 5.0
        half_life = 2.0
        
        blocked_no_decay = block_hrf(SPM_CANONICAL, width, half_life=np.inf)
        blocked_decay = block_hrf(SPM_CANONICAL, width, half_life=half_life)
        
        result_no_decay = blocked_no_decay(t)
        result_decay = blocked_decay(t)
        
        # After stimulus offset, decay version should be smaller on average
        post_stim = t > (width + 5)  # Well after stimulus
        
        # Filter out values that are very close to zero to avoid numerical issues
        threshold = 1e-6
        significant_values = np.abs(result_no_decay[post_stim]) > threshold
        
        if np.any(significant_values):
            # Compare only where values are significant
            decay_vals = result_decay[post_stim][significant_values]
            no_decay_vals = result_no_decay[post_stim][significant_values]
            
            # Most absolute values should be smaller with decay (closer to zero)
            smaller_count = np.sum(np.abs(decay_vals) < np.abs(no_decay_vals))
            total_count = len(decay_vals)
            assert smaller_count / total_count > 0.8  # At least 80% should be smaller in magnitude
    
    def test_block_hrf_normalize_retired(self):
        """block_hrf(normalize=True) retired (bd-01KRGCZ6QJME1JD8FD5D4PGC04).

        The replacement composes the typed wrappers explicitly:
        ``normalize(block_hrf(...), 'unit_peak_per_basis')``.
        """
        with pytest.raises(ValueError, match="retired"):
            block_hrf(SPM_CANONICAL, 5.0, normalize=True)

        t = np.arange(0, 30, 0.2)
        blocked_norm = _normalize_per_basis(block_hrf(SPM_CANONICAL, 5.0))
        result = blocked_norm(t)
        # 1e-3 tolerance reflects the gap between the reference grid used
        # for peak detection (~0.02 s) and the evaluation grid here (0.2 s).
        assert abs(np.max(np.abs(result)) - 1.0) < 1e-3
    
    def test_block_hrf_validation(self):
        """Test input validation."""
        with pytest.raises(ValueError, match="finite"):
            block_hrf(SPM_CANONICAL, width=np.inf)
        
        with pytest.raises(ValueError, match="finite and positive"):
            block_hrf(SPM_CANONICAL, width=1.0, precision=0)
        
        with pytest.raises(ValueError, match="positive"):
            block_hrf(SPM_CANONICAL, width=1.0, half_life=-1)


class TestNormalizePerBasis:
    """Test the replacement for the retired normalize_hrf decorator."""
    
    def test_normalize_per_basis_single_basis(self):
        """Test normalization of single-basis HRF."""
        # Create unnormalized HRF
        def unnorm_func(t):
            return 5.0 * SPM_CANONICAL(t)  # Scaled by 5
        
        unnorm_hrf = as_hrf(unnorm_func, name="unnorm")
        norm_hrf = _normalize_per_basis(unnorm_hrf)
        
        # Check attributes
        assert "[norm=" in norm_hrf.name
        assert norm_hrf.norm_mode is not None
        
        # Check peak is approximately 1
        t = np.arange(0, 20, 0.1)
        result = norm_hrf(t)
        assert abs(np.max(np.abs(result)) - 1.0) < 0.001  # Allow 0.1% error
    
    def test_normalize_per_basis_multi_basis(self):
        """Test normalization of multi-basis HRF."""
        # Create scaled version
        def scaled_func(t):
            result = SPM_WITH_DERIVATIVE(t)
            result[:, 0] *= 3.0
            result[:, 1] *= 5.0
            return result
        
        scaled_hrf = as_hrf(scaled_func, name="scaled", nbasis=2)
        norm_hrf = _normalize_per_basis(scaled_hrf)
        
        t = np.arange(0, 20, 0.1)
        result = norm_hrf(t)
        
        # Each basis should be normalized
        assert abs(np.max(np.abs(result[:, 0])) - 1.0) < 0.001
        assert abs(np.max(np.abs(result[:, 1])) - 1.0) < 0.001
    
    def test_normalize_per_basis_already_normalized(self):
        """Test normalizing already normalized HRF."""
        # SPM canonical should already be close to normalized
        norm_hrf = _normalize_per_basis(SPM_CANONICAL)
        
        t = np.arange(0, 20, 0.1)
        original = SPM_CANONICAL(t)
        normalized = norm_hrf(t)
        
        # Should be scaled by inverse of original peak
        original_peak = np.max(np.abs(original))
        expected = original / original_peak
        assert np.allclose(normalized, expected, rtol=1e-4)


class TestGenHRFLagged:
    """Test gen_hrf_lagged function."""
    
    def test_gen_hrf_lagged_basic(self):
        """Test basic lagged set generation."""
        lags = [0, 1, 2, 3]
        lagged_set = gen_hrf_lagged(SPM_CANONICAL, lags)
        
        assert lagged_set.nbasis == len(lags)
        
        # Check each basis is correctly lagged
        t = np.arange(0, 20, 0.5)
        result = lagged_set(t)
        
        for i, lag in enumerate(lags):
            expected = SPM_CANONICAL(t - lag)
            assert np.allclose(result[:, i], expected)
    
    def test_gen_hrf_lagged_custom_name(self):
        """Test custom naming."""
        lags = [0, 2]
        lagged_set = gen_hrf_lagged(SPM_CANONICAL, lags, name="my_lagged_set")
        
        assert lagged_set.name == "my_lagged_set"
        assert isinstance(lagged_set, BoundBasisHRF)
        assert not isinstance(lagged_set, FunctionHRF)
    
    def test_gen_hrf_lagged_from_function(self):
        """Test with function input."""
        def my_hrf(t):
            return np.exp(-t/5) * (t > 0)
        
        lags = [0, 1, 2]
        lagged_set = gen_hrf_lagged(my_hrf, lags)
        
        assert lagged_set.nbasis == len(lags)


class TestGenHRFBlocked:
    """Test gen_hrf_blocked function."""
    
    def test_gen_hrf_blocked_basic(self):
        """Test basic blocked set generation."""
        widths = [0, 2, 4, 6]
        blocked_set = gen_hrf_blocked(SPM_CANONICAL, widths)
        
        assert blocked_set.nbasis == len(widths)
        
        # Check response increases with width
        t = np.arange(0, 30, 0.5)
        result = blocked_set(t)
        
        # Integral should increase with width
        integrals = [_trapezoid_compat(result[:, i], t) for i in range(len(widths))]
        assert all(integrals[i] <= integrals[i+1] for i in range(len(integrals)-1))
    
    def test_gen_hrf_blocked_normalize(self):
        """Test blocked set with normalization."""
        widths = [0, 3, 6]
        blocked_set = gen_hrf_blocked(SPM_CANONICAL, widths, normalize=True)
        
        t = np.arange(0, 30, 0.2)
        result = blocked_set(t)
        
        # Each basis should be normalized
        for i in range(len(widths)):
            assert abs(np.max(np.abs(result[:, i])) - 1.0) < 0.001
    
    def test_gen_hrf_blocked_custom_params(self):
        """Test with custom parameters."""
        widths = [2, 4]
        blocked_set = gen_hrf_blocked(
            SPM_CANONICAL,
            widths,
            precision=0.05,
            half_life=3.0,
            summate=False,
            name="custom_blocked"
        )
        
        assert blocked_set.name == "custom_blocked"
        assert blocked_set.nbasis == len(widths)
        assert isinstance(blocked_set, BoundBasisHRF)
        assert not isinstance(blocked_set, FunctionHRF)
