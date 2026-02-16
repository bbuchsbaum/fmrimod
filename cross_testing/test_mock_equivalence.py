"""Mock cross-testing to demonstrate the infrastructure without R dependencies."""

import numpy as np
import pytest
from fmrimod import get_hrf, regressor, regressor_set, SamplingFrame


class MockREquivalenceTester:
    """Mock version of REquivalenceTester for demonstration."""
    
    def __init__(self):
        """Initialize mock tester."""
        self.mock_results = {}
        
    def compare_arrays(self, r_result, py_result, rtol=1e-10, atol=1e-12):
        """Mock array comparison - just checks Python results."""
        # In real cross-testing, this would compare R and Python
        # For now, just verify Python results are reasonable
        assert isinstance(py_result, np.ndarray), "Result should be numpy array"
        assert not np.any(np.isnan(py_result)), "Result contains NaN values"
        assert not np.any(np.isinf(py_result)), "Result contains infinite values"
        
    def get_mock_r_result(self, key):
        """Get mock R result for comparison."""
        # In real testing, these would come from R
        # Here we generate reasonable mock values
        if key == 'spm_hrf':
            t = np.linspace(0, 30, 100)
            # Mock SPM HRF shape
            return 0.1 * np.exp(-t/6) * np.sin(t/2)
        elif key == 'regressor_result':
            return np.random.randn(100)
        return np.zeros(100)


class TestMockHRFEquivalence:
    """Test HRF functionality with mock R results."""
    
    def test_spm_canonical(self):
        """Test SPM canonical HRF."""
        tester = MockREquivalenceTester()
        
        # Python version
        t = np.linspace(0, 30, 100)
        from fmrimod import spm_canonical
        py_result = spm_canonical(t)
        
        # Mock R result
        r_result = tester.get_mock_r_result('spm_hrf')
        
        # Verify Python result is valid
        tester.compare_arrays(r_result, py_result)
        
        # Check basic properties
        assert py_result.shape == (100,)
        assert np.max(py_result) > 0  # Should have positive peak
        assert py_result[0] == 0  # Should start at 0
        
    def test_gamma_hrf_shapes(self):
        """Test gamma HRF with different shapes."""
        tester = MockREquivalenceTester()
        
        shapes = [4, 6, 8]
        t = np.linspace(0, 30, 100)
        
        from fmrimod import gamma_hrf
        
        for shape in shapes:
            py_result = gamma_hrf(t, shape=shape, rate=1.0)
            
            # Verify result
            tester.compare_arrays(None, py_result)
            
            # Check peak time increases with shape
            peak_idx = np.argmax(py_result)
            peak_time = t[peak_idx]
            
            # Higher shape parameter should lead to later peak
            expected_peak = shape - 1  # Approximate peak time for gamma
            assert abs(peak_time - expected_peak) < 2.0
    
    def test_regressor_basic(self):
        """Test basic regressor functionality."""
        tester = MockREquivalenceTester()
        
        # Create regressor
        onsets = [10, 30, 50]
        reg = regressor(onsets=onsets, hrf="spmg1", duration=2.0)
        
        # Evaluate
        sf = SamplingFrame(blocklens=100, tr=2.0)
        py_result = reg.evaluate(sf.samples)
        
        # Verify
        tester.compare_arrays(None, py_result)
        
        # Check properties
        assert py_result.shape == (100,)
        assert np.any(py_result > 0)  # Should have some activation
        
        # Check that we have responses around the event times
        for onset in onsets:
            # Find closest sample time
            idx = np.argmin(np.abs(sf.samples - onset))
            # Check there's activity in a window around the event
            window = slice(max(0, idx-5), min(len(py_result), idx+15))
            assert np.any(py_result[window] > 0.01)


class TestMockRegressorSet:
    """Test regressor set functionality with mock comparisons."""
    
    def test_design_matrix(self):
        """Test design matrix construction."""
        tester = MockREquivalenceTester()
        
        # Create events
        onsets = [5, 15, 25, 35]
        conditions = ['A', 'B', 'A', 'B']
        
        # Create regressor set
        rset = regressor_set(
            onsets=onsets,
            fac=conditions,
            hrf="spmg1"
        )
        
        # Evaluate
        sf = SamplingFrame(blocklens=50, tr=2.0)
        design = rset.evaluate(sf.samples)
        
        # Verify
        assert design.shape == (50, 2)  # 2 conditions
        assert not np.any(np.isnan(design))
        
        # Check columns are different
        assert not np.allclose(design[:, 0], design[:, 1])
        
        # Check both columns have some activation
        assert np.any(design[:, 0] > 0.01)
        assert np.any(design[:, 1] > 0.01)


class TestMockPerformance:
    """Test performance aspects without actual R comparison."""
    
    def test_large_scale_python_only(self):
        """Test Python performance with large datasets."""
        import time
        
        # Large dataset
        n_events = 1000
        n_timepoints = 5000
        
        np.random.seed(42)
        onsets = np.sort(np.random.uniform(0, 2500, n_events))
        times = np.linspace(0, 2500, n_timepoints)
        
        # Time Python implementation
        start = time.time()
        reg = regressor(onsets=onsets, hrf="spmg1")
        result = reg.evaluate(times)
        py_time = time.time() - start
        
        print(f"\nPython performance (large scale):")
        print(f"  Events: {n_events}, Timepoints: {n_timepoints}")
        print(f"  Time: {py_time:.3f}s")
        print(f"  Speed: {n_timepoints/py_time:.0f} samples/sec")
        
        # Verify result
        assert result.shape == (n_timepoints,)
        assert not np.any(np.isnan(result))
        assert np.max(result) > 0


def test_infrastructure_works():
    """Test that the cross-testing infrastructure is set up correctly."""
    # Check imports work
    from cross_testing.utils import RPY2_AVAILABLE
    
    if not RPY2_AVAILABLE:
        print("\nNote: rpy2 not available - using mock tests")
        print("For full cross-testing, install rpy2 and R fmrihrf package")
    
    # Check test discovery works
    import cross_testing
    assert hasattr(cross_testing, '__file__')
    
    # Verify structure
    from pathlib import Path
    cross_test_dir = Path(cross_testing.__file__).parent
    
    expected_files = [
        'utils.py',
        'conftest.py', 
        'test_hrf_equivalence.py',
        'test_regressor_equivalence.py',
        'test_complex_scenarios.py',
        'test_performance.py'
    ]
    
    for fname in expected_files:
        assert (cross_test_dir / fname).exists(), f"Missing {fname}"
    
    print("\n✅ Cross-testing infrastructure is properly set up!")
    print(f"   Location: {cross_test_dir}")
    print(f"   Test files: {len(expected_files)}")


if __name__ == "__main__":
    # Run the infrastructure test
    test_infrastructure_works()
    
    # Run mock tests
    print("\nRunning mock cross-tests...")
    pytest.main([__file__, '-v'])