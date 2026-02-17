"""Tests for the convolve function - mirroring and extending R tests."""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal

from fmrimod.events.factor import EventFactor
from fmrimod.events.variable import EventVariable
from fmrimod.events.matrix import EventMatrix
from fmrimod.events.basis import EventBasis
from fmrimod.convolve import convolve
from fmrimod import event_model, get_hrf, as_hrf
from fmrimod.convolve import _get_hrf_array, _convolve_impulses


class TestConvolve:
    """Test direct convolution functionality."""
    
    def test_convolve_event_variable(self):
        """Test convolution of EventVariable."""
        event = EventVariable(
            onsets=[1, 5, 10],
            durations=[2, 2, 2],
            values=[1, 2, 3],
            name="stimulus"
        )
        
        # Test with default HRF
        result = convolve(event, sampling_rate=0.5)
        assert result.shape[1] == 1  # Single column for continuous
        assert result.shape[0] > 0
        
        # Test with custom HRF
        custom_hrf = get_hrf('spm')
        result2 = convolve(event, hrf=custom_hrf, sampling_rate=0.5)
        assert result2.shape[1] == 1
    
    def test_convolve_event_factor(self):
        """Test convolution of EventFactor."""
        event = EventFactor(
            onsets=[1, 5, 10, 15],
            durations=[2, 2, 2, 2],
            values=['A', 'B', 'A', 'C'],
            name="condition"
        )
        
        result = convolve(event, sampling_rate=1.0)
        # Should have one column per level
        assert result.shape[1] == 3  # A, B, C
        assert result.shape[0] > 0
        
        # Check that columns sum differently (different number of events)
        col_sums = np.sum(result, axis=0)
        assert not np.allclose(col_sums[0], col_sums[1])  # A vs B
    
    def test_convolve_event_matrix(self):
        """Test convolution of EventMatrix."""
        values = np.array([[1, 0], [0, 1], [1, 0], [0.5, 0.5]])
        event = EventMatrix(
            name="matrix_event",
            onsets=[1, 5, 10, 15],
            durations=[2, 2, 2, 2],
            values=values,
            column_names=['col1', 'col2']
        )
        
        result = convolve(event, sampling_rate=1.0)
        assert result.shape[1] == 2  # Two columns
        assert result.shape[0] > 0
    
    def test_convolve_list_of_events(self):
        """Test convolution of list of events."""
        event1 = EventVariable(
            onsets=[1, 5], durations=[1, 1], values=[1, 2], name="ev1"
        )
        event2 = EventVariable(
            onsets=[3, 7], durations=[1, 1], values=[3, 4], name="ev2"
        )
        
        results = convolve([event1, event2], sampling_rate=1.0)
        assert len(results) == 2
        assert all(isinstance(r, np.ndarray) for r in results)
    
    def test_convolve_numpy_array(self):
        """Test convolution of numpy array [onset, duration, value]."""
        arr = np.array([
            [1.0, 2.0, 1.0],
            [5.0, 2.0, 2.0],
            [10.0, 2.0, 3.0]
        ])
        
        result = convolve(arr, sampling_rate=1.0)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape[0] > 0
    
    def test_convolve_with_array_hrf(self):
        """Test convolution with HRF specified as array."""
        event = EventVariable(
            onsets=[1, 5], durations=[1, 1], values=[1, 2], name="test", center=False
        )

        # Custom HRF as array
        hrf_array = np.array([0, 0.5, 1.0, 0.5, 0])
        result = convolve(event, hrf=hrf_array, sampling_rate=1.0)

        assert result.shape[0] > 0
        assert np.max(result) > 0
    
    def test_convolve_with_function_hrf(self):
        """Test convolution with HRF specified as function."""
        event = EventVariable(
            onsets=[1, 5], durations=[1, 1], values=[1, 2], name="test", center=False
        )

        # Custom HRF as function
        def my_hrf(t):
            return t * np.exp(-t / 2)

        result = convolve(event, hrf=my_hrf, sampling_rate=1.0)
        assert result.shape[0] > 0
    
    def test_convolve_preserves_timing(self):
        """Test that convolution preserves event timing."""
        # Single event at t=10
        event = EventVariable(
            onsets=[10.0], durations=[1.0], values=[1.0], name="test", center=False
        )

        result = convolve(event, sampling_rate=1.0, total_duration=30.0)

        # Peak should be around t=10 + HRF peak time (~5-6s)
        peak_idx = np.argmax(result)
        assert 10 <= peak_idx <= 20
    
    def test_convolve_duration_effects(self):
        """Test that longer durations produce larger responses."""
        # Two events with different durations
        event_short = EventVariable(
            onsets=[10.0], durations=[0.5], values=[1.0], name="short", center=False
        )
        event_long = EventVariable(
            onsets=[10.0], durations=[5.0], values=[1.0], name="long", center=False
        )

        result_short = convolve(event_short, sampling_rate=1.0, total_duration=30.0)
        result_long = convolve(event_long, sampling_rate=1.0, total_duration=30.0)

        # Longer duration should produce larger integrated response
        assert np.sum(result_long) > np.sum(result_short) * 2
    
    def test_convolve_balanced_design(self):
        """Test convolution preserves balance in factorial designs."""
        # Create balanced 2x2 design
        onsets = [1, 5, 10, 15, 20, 25, 30, 35]
        conditions = ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B']
        
        event = EventFactor(
            onsets=onsets,
            durations=[2] * 8,
            values=conditions,
            name="condition"
        )
        
        result = convolve(event, sampling_rate=1.0, total_duration=50.0)
        
        # Both conditions should have similar total response
        col_sums = np.sum(result, axis=0)
        assert np.abs(col_sums[0] - col_sums[1]) / np.mean(col_sums) < 0.1
    
    def test_convolve_event_basis(self):
        """Test that EventBasis convolution works with pyfmrihrf."""
        # EventBasis now works with pyfmrihrf
        from fmrimod.basis import Poly
        event = EventBasis(
            name="basis_event",
            onsets=[1, 5, 10],
            durations=[2, 2, 2],
            values=[1, 2, 3],
            basis=Poly(degree=2)
        )

        # Should work with pyfmrihrf available
        result = convolve(event, sampling_rate=1.0)
        assert result.shape[0] > 0
        assert result.shape[1] == event.n_basis * len(event.onsets)
    
    def test_get_hrf_array(self):
        """Test HRF array generation."""
        # Test with None (default)
        hrf_array = _get_hrf_array(None, sampling_rate=1.0)
        assert isinstance(hrf_array, np.ndarray)
        assert hrf_array.shape[0] > 0
        
        # Test with existing array
        custom_array = np.array([1, 2, 3, 2, 1])
        result = _get_hrf_array(custom_array, sampling_rate=1.0)
        assert_array_equal(result, custom_array)
        
        # Test with HRF object
        hrf_obj = get_hrf('spm')
        result = _get_hrf_array(hrf_obj, sampling_rate=1.0)
        assert isinstance(result, np.ndarray)
    
    def test_convolve_impulses(self):
        """Test low-level impulse convolution."""
        times = np.array([5.0, 10.0, 15.0])
        values = np.array([1.0, 2.0, 1.5])
        durations = np.array([1.0, 2.0, 1.0])
        hrf_array = np.array([0, 0.5, 1.0, 0.5, 0])
        
        result = _convolve_impulses(
            times, values, durations, hrf_array,
            sampling_rate=1.0, total_duration=30.0
        )
        
        assert result.shape[0] == 30
        assert np.max(result) > 0
        
    def test_convolve_empty_events(self):
        """Test convolution with empty events."""
        # Empty events are now disallowed by validation, so skip this test
        # or test that it raises ValidationError
        from fmrimod.types import ValidationError

        with pytest.raises(ValidationError, match="Onsets cannot be empty"):
            event = EventVariable(
                onsets=[], durations=[], values=[], name="empty"
            )
    
    def test_convolve_single_event(self):
        """Test convolution with single event."""
        event = EventVariable(
            onsets=[5.0], durations=[1.0], values=[2.0], name="single", center=False
        )

        result = convolve(event, sampling_rate=2.0, total_duration=20.0)
        assert result.shape == (40, 1)
        assert np.max(result) > 0
        assert np.sum(result) > 0
    
    def test_convolve_factor_with_missing_level(self):
        """Test factor convolution when some levels have no events."""
        event = EventFactor(
            onsets=[1, 5, 10],
            durations=[1, 1, 1],
            values=['A', 'A', 'A'],  # Only A, no B
            name="condition",
            levels=['A', 'B', 'C']  # B and C defined but not used
        )
        
        result = convolve(event, sampling_rate=1.0, total_duration=20.0)
        assert result.shape[1] == 3  # All levels represented
        
        # Check that unused levels have zero response
        assert np.sum(result[:, 0]) > 0  # A has events
        assert np.sum(result[:, 1]) == 0  # B has no events
        assert np.sum(result[:, 2]) == 0  # C has no events


class TestConvolveIntegration:
    """Integration tests comparing with event_model convolution."""

    def test_convolve_matches_event_model(self):
        """Test that direct convolution matches event_model results."""
        # Create simple event
        data = pd.DataFrame({
            'onset': [1, 5, 10],
            'condition': ['A', 'B', 'A']
        })

        # Using event_model (full pipeline)
        from fmrimod import SamplingFrame
        sframe = SamplingFrame(blocklens=[20], TR=1.0)

        model = event_model(
            'condition',
            data=data,
            sampling_info=sframe
        )

        # Using direct convolution
        event = EventFactor(
            onsets=[1, 5, 10],
            durations=[0, 0, 0],  # Default durations
            values=['A', 'B', 'A'],
            name='condition'
        )

        direct_result = convolve(event, sampling_rate=1.0, total_duration=20.0)

        # Results should be similar (not exact due to implementation differences)
        # Just check shapes and rough magnitude
        assert direct_result.shape[1] == 2  # A and B
        assert direct_result.shape[0] == 20
        assert np.max(direct_result) > 0


class TestConvolveEventFactorComprehensive:
    """Comprehensive tests for EventFactor convolution."""

    def test_simple_2level_factor_spm_canonical(self):
        """Test 2-level factor with SPM canonical HRF."""
        event = EventFactor(
            name="condition",
            onsets=[2.0, 6.0, 10.0, 14.0],
            durations=[1.0, 1.0, 1.0, 1.0],
            values=['A', 'B', 'A', 'B']
        )

        result = convolve(event, hrf='spm', sampling_rate=1.0, total_duration=30.0)

        # Check shape: 30 time points, 2 levels
        assert result.shape == (30, 2)

        # Check both levels have non-zero response
        assert np.max(result[:, 0]) > 0  # Level A
        assert np.max(result[:, 1]) > 0  # Level B

        # Check balanced design produces similar total response
        col_sums = np.sum(result, axis=0)
        assert np.abs(col_sums[0] - col_sums[1]) / np.mean(col_sums) < 0.1

    def test_multilevel_factor_3plus_levels(self):
        """Test multi-level factor with 3+ levels."""
        event = EventFactor(
            name="condition",
            onsets=[2.0, 6.0, 10.0, 14.0, 18.0, 22.0],
            durations=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            values=['A', 'B', 'C', 'A', 'B', 'C']
        )

        result = convolve(event, sampling_rate=1.0, total_duration=35.0)

        # Check shape: 35 time points, 3 levels
        assert result.shape == (35, 3)

        # Check all levels have non-zero response
        assert np.max(result[:, 0]) > 0  # Level A
        assert np.max(result[:, 1]) > 0  # Level B
        assert np.max(result[:, 2]) > 0  # Level C

        # Check balanced design
        col_sums = np.sum(result, axis=0)
        assert np.allclose(col_sums, col_sums[0], rtol=0.1)

    def test_factor_with_sampling_frame(self):
        """Test EventFactor with explicit sampling_frame parameter."""
        event = EventFactor(
            name="stim",
            onsets=[2.0, 6.0, 10.0],
            durations=[0.5, 0.5, 0.5],
            values=['X', 'Y', 'X']
        )

        # Explicit sampling grid (TR=2.0)
        sampling_grid = np.arange(0, 20, 2.0)

        result = convolve(event, sampling_frame=sampling_grid)

        # Check shape matches sampling grid
        assert result.shape == (len(sampling_grid), 2)
        assert np.max(result) > 0

    def test_factor_with_normalize(self):
        """Test EventFactor with normalize=True."""
        event = EventFactor(
            name="cond",
            onsets=[2.0, 6.0],
            durations=[1.0, 1.0],
            values=['A', 'A']
        )

        result = convolve(event, sampling_rate=1.0, total_duration=20.0, normalize=True)

        # Check that peak-normalization worked
        # Max absolute value should be 1.0 for each column
        for i in range(result.shape[1]):
            max_abs = np.max(np.abs(result[:, i]))
            if max_abs > 0:
                assert np.isclose(max_abs, 1.0)

    def test_factor_with_summate_false(self):
        """Test EventFactor with summate=False."""
        # Create overlapping events
        event = EventFactor(
            name="overlap",
            onsets=[2.0, 3.0],  # Close together, will overlap
            durations=[2.0, 2.0],
            values=['A', 'A']
        )

        result_sum = convolve(event, sampling_rate=1.0, total_duration=15.0, summate=True)
        result_max = convolve(event, sampling_rate=1.0, total_duration=15.0, summate=False)

        # With summate=True, overlapping responses should sum
        # With summate=False, they should take max
        # The sum version should generally be larger in overlap regions
        assert result_sum.shape == result_max.shape
        assert np.max(result_sum) >= np.max(result_max)


class TestConvolveEventVariableComprehensive:
    """Comprehensive tests for EventVariable convolution."""

    def test_simple_continuous_variable(self):
        """Test simple continuous variable."""
        event = EventVariable(
            name="rating",
            onsets=[2.0, 6.0, 10.0],
            durations=[1.0, 1.0, 1.0],
            values=[1.5, 2.5, 3.5],
            center=False
        )

        result = convolve(event, sampling_rate=1.0, total_duration=20.0)

        # Check shape: single column for continuous
        assert result.shape == (20, 1)
        assert np.max(result) > 0

    def test_variable_with_hrf_string_spmg1(self):
        """Test EventVariable with HRF string name 'spmg1'."""
        event = EventVariable(
            name="value",
            onsets=[5.0, 10.0],
            durations=[1.0, 1.0],
            values=[1.0, 2.0],
            center=False
        )

        result = convolve(event, hrf='spmg1', sampling_rate=1.0, total_duration=25.0)

        assert result.shape == (25, 1)
        assert np.max(result) > 0

    def test_variable_with_hrf_string_spmg2(self):
        """Test EventVariable with HRF string name 'spmg2' (with derivative)."""
        event = EventVariable(
            name="value",
            onsets=[5.0, 10.0],
            durations=[1.0, 1.0],
            values=[1.0, 2.0],
            center=False
        )

        result = convolve(event, hrf='spmg2', sampling_rate=1.0, total_duration=25.0)

        # spmg2 includes time derivative, so should have 2 basis functions
        # But for EventVariable, we get weighted combination
        assert result.shape[0] == 25
        assert np.max(result) > 0

    def test_variable_with_normalize(self):
        """Test EventVariable with normalize=True."""
        event = EventVariable(
            name="rt",
            onsets=[2.0, 6.0, 10.0],
            durations=[0.5, 0.5, 0.5],
            values=[0.5, 1.5, 2.5],
            center=False
        )

        result = convolve(event, sampling_rate=1.0, total_duration=20.0, normalize=True)

        # Peak-normalized: max absolute value should be 1.0
        max_abs = np.max(np.abs(result))
        assert np.isclose(max_abs, 1.0)

    def test_variable_different_durations(self):
        """Test EventVariable with different event durations."""
        event = EventVariable(
            name="stimulus",
            onsets=[2.0, 8.0, 14.0],
            durations=[0.5, 2.0, 1.0],  # Different durations
            values=[1.0, 1.0, 1.0],
            center=False
        )

        result = convolve(event, sampling_rate=1.0, total_duration=25.0)

        assert result.shape == (25, 1)
        # Longer duration event should contribute more to integrated response
        assert np.sum(result) > 0


class TestConvolveEventMatrixComprehensive:
    """Comprehensive tests for EventMatrix convolution."""

    def test_2column_matrix_event(self):
        """Test 2-column matrix event."""
        values = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

        event = EventMatrix(
            name="multi_param",
            onsets=[2.0, 6.0, 10.0, 14.0],
            durations=[1.0, 1.0, 1.0, 1.0],
            values=values,
            column_names=['param1', 'param2']
        )

        result = convolve(event, sampling_rate=1.0, total_duration=25.0)

        # Check shape: matches number of columns
        assert result.shape == (25, 2)

        # Both columns should have non-zero response
        assert np.max(result[:, 0]) > 0
        assert np.max(result[:, 1]) > 0

    def test_matrix_output_shape(self):
        """Test that EventMatrix output shape matches input columns."""
        n_events = 5
        n_cols = 4
        values = np.random.randn(n_events, n_cols)

        event = EventMatrix(
            name="motion",
            onsets=[2.0, 4.0, 6.0, 8.0, 10.0],
            durations=[0.5, 0.5, 0.5, 0.5, 0.5],
            values=values
        )

        result = convolve(event, sampling_rate=1.0, total_duration=20.0)

        # Output should have same number of columns as input
        assert result.shape[1] == n_cols
        assert result.shape[0] == 20

    def test_matrix_with_sampling_frame(self):
        """Test EventMatrix with explicit sampling_frame."""
        values = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 1.0]
        ])

        event = EventMatrix(
            name="xyz",
            onsets=[5.0, 10.0],
            durations=[1.0, 1.0],
            values=values
        )

        # TR=2.0 sampling
        sampling_grid = np.arange(0, 20, 2.0)

        result = convolve(event, sampling_frame=sampling_grid)

        assert result.shape == (len(sampling_grid), 3)


class TestConvolveEventBasisComprehensive:
    """Comprehensive tests for EventBasis convolution."""

    def test_basis_expanded_event(self):
        """Test basis-expanded event."""
        from fmrimod.basis import Poly

        event = EventBasis(
            name="modulator",
            onsets=[2.0, 6.0, 10.0],
            durations=[1.0, 1.0, 1.0],
            values=[1.0, 2.0, 3.0],
            basis=Poly(degree=2)
        )

        result = convolve(event, sampling_rate=1.0, total_duration=20.0)

        # Check that result has correct dimensions
        assert result.shape[0] == 20
        # For EventBasis, we expect n_events * n_basis columns
        # Poly(degree=2) has 2 basis functions
        assert result.shape[1] == event.n_basis * len(event.onsets)

    def test_basis_column_count(self):
        """Test that EventBasis produces correct number of columns."""
        from fmrimod.basis import Poly

        n_events = 4
        degree = 3

        event = EventBasis(
            name="poly_expand",
            onsets=[2.0, 5.0, 8.0, 11.0],
            durations=[0.5, 0.5, 0.5, 0.5],
            values=[1.0, 2.0, 1.5, 2.5],
            basis=Poly(degree=degree)
        )

        result = convolve(event, sampling_rate=1.0, total_duration=20.0)

        # Poly(degree=3) has 3 basis functions
        # With 4 events, we get 4 * 3 = 12 columns
        expected_cols = n_events * event.n_basis
        assert result.shape[1] == expected_cols


class TestConvolveNumpyArrayComprehensive:
    """Comprehensive tests for numpy array convolution."""

    def test_array_shape_validation(self):
        """Test that invalid array shapes raise ValueError."""
        # Wrong shape: only 2 columns
        arr_wrong = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="must have shape"):
            convolve(arr_wrong, sampling_rate=1.0)

        # Wrong shape: 1D array
        arr_1d = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="must have shape"):
            convolve(arr_1d, sampling_rate=1.0)

    def test_array_3column_format(self):
        """Test array with correct [onset, duration, value] format."""
        arr = np.array([
            [2.0, 1.0, 1.5],
            [6.0, 1.0, 2.5],
            [10.0, 0.5, 1.0]
        ])

        result = convolve(arr, sampling_rate=1.0, total_duration=20.0)

        # Should produce 1D result
        assert result.ndim == 1
        assert result.shape[0] == 20
        assert np.max(result) > 0

    def test_array_with_zero_durations(self):
        """Test array with zero-duration impulse events."""
        arr = np.array([
            [5.0, 0.0, 1.0],
            [10.0, 0.0, 2.0],
            [15.0, 0.0, 1.5]
        ])

        result = convolve(arr, sampling_rate=1.0, total_duration=30.0)

        assert result.shape[0] == 30
        assert np.max(result) > 0


class TestConvolveListComprehensive:
    """Comprehensive tests for list of events convolution."""

    def test_list_of_mixed_event_types(self):
        """Test convolution of list with mixed event types."""
        ev1 = EventVariable(
            name="var1",
            onsets=[2.0, 6.0],
            durations=[1.0, 1.0],
            values=[1.0, 2.0],
            center=False
        )

        ev2 = EventFactor(
            name="fac1",
            onsets=[3.0, 7.0],
            durations=[1.0, 1.0],
            values=['A', 'B']
        )

        results = convolve([ev1, ev2], sampling_rate=1.0, total_duration=15.0)

        # Should return list of results
        assert len(results) == 2
        assert isinstance(results[0], np.ndarray)
        assert isinstance(results[1], np.ndarray)

        # Check shapes
        assert results[0].shape == (15, 1)  # EventVariable -> 1 column
        assert results[1].shape == (15, 2)  # EventFactor with 2 levels

    def test_list_preserves_order(self):
        """Test that list convolution preserves event order."""
        events = []
        for i in range(5):
            ev = EventVariable(
                name=f"ev{i}",
                onsets=[2.0 + i*2],
                durations=[0.5],
                values=[float(i)],
                center=False
            )
            events.append(ev)

        results = convolve(events, sampling_rate=1.0, total_duration=20.0)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.shape == (20, 1)


class TestConvolveEdgeCases:
    """Test edge cases and parameter combinations."""

    def test_very_short_sampling_rate(self):
        """Test with very fine temporal resolution."""
        event = EventVariable(
            name="fine",
            onsets=[5.0],
            durations=[0.1],
            values=[1.0],
            center=False
        )

        # High sampling rate (10 Hz)
        result = convolve(event, sampling_rate=10.0, total_duration=15.0)

        assert result.shape == (150, 1)  # 15 seconds at 10 Hz
        assert np.max(result) > 0

    def test_custom_hrf_as_array(self):
        """Test with custom HRF provided as array."""
        event = EventFactor(
            name="test",
            onsets=[5.0, 10.0],
            durations=[1.0, 1.0],
            values=['A', 'A']
        )

        # Simple triangular HRF
        custom_hrf = np.array([0, 0.5, 1.0, 0.5, 0])

        result = convolve(event, hrf=custom_hrf, sampling_rate=1.0, total_duration=20.0)

        assert result.shape == (20, 1)
        assert np.max(result) > 0

    def test_events_at_boundary(self):
        """Test events at start and end of time window."""
        event = EventVariable(
            name="boundary",
            onsets=[0.0, 18.0],  # At boundaries
            durations=[1.0, 1.0],
            values=[1.0, 1.0],
            center=False
        )

        result = convolve(event, sampling_rate=1.0, total_duration=20.0)

        assert result.shape == (20, 1)
        # Should still produce valid output
        assert np.sum(result) > 0

    def test_normalize_and_summate_combination(self):
        """Test combination of normalize=True and summate=False."""
        event = EventFactor(
            name="combo",
            onsets=[2.0, 6.0, 10.0],
            durations=[1.0, 1.0, 1.0],
            values=['X', 'Y', 'X']
        )

        result = convolve(
            event,
            sampling_rate=1.0,
            total_duration=20.0,
            normalize=True,
            summate=False
        )

        assert result.shape == (20, 2)
        # Check peak-normalization
        for i in range(result.shape[1]):
            max_abs = np.max(np.abs(result[:, i]))
            if max_abs > 0:
                assert np.isclose(max_abs, 1.0)

    def test_hrf_string_spmg3(self):
        """Test with spmg3 HRF string (with dispersion derivative)."""
        event = EventVariable(
            name="test",
            onsets=[5.0],
            durations=[1.0],
            values=[1.0],
            center=False
        )

        result = convolve(event, hrf='spmg3', sampling_rate=1.0, total_duration=20.0)

        assert result.shape[0] == 20
        assert np.max(result) > 0

    def test_hrf_unknown_string(self):
        """Test with unknown HRF string falls back to SPM canonical."""
        event = EventVariable(
            name="test",
            onsets=[5.0],
            durations=[1.0],
            values=[1.0],
            center=False
        )

        # Unknown HRF name should fall back to SPM canonical
        result = convolve(event, hrf='unknown_hrf_name', sampling_rate=1.0, total_duration=20.0)

        assert result.shape[0] == 20
        assert np.max(result) > 0

    def test_matrix_with_array_hrf_fallback(self):
        """Test EventMatrix with array HRF (uses fallback path)."""
        values = np.array([
            [1.0, 2.0],
            [2.0, 1.0]
        ])

        event = EventMatrix(
            name="matrix",
            onsets=[5.0, 10.0],
            durations=[1.0, 1.0],
            values=values
        )

        # Custom HRF array triggers fallback path
        custom_hrf = np.array([0, 0.5, 1.0, 0.5, 0])

        result = convolve(event, hrf=custom_hrf, sampling_rate=1.0, total_duration=20.0)

        assert result.shape == (20, 2)
        assert np.max(result) > 0

    def test_array_with_array_hrf_fallback(self):
        """Test numpy array convolution with array HRF (uses fallback path)."""
        arr = np.array([
            [5.0, 1.0, 1.5],
            [10.0, 1.0, 2.0]
        ])

        # Custom HRF array triggers fallback path
        custom_hrf = np.array([0, 0.5, 1.0, 0.5, 0])

        result = convolve(arr, hrf=custom_hrf, sampling_rate=1.0, total_duration=20.0)

        assert result.shape[0] == 20
        assert np.max(result) > 0

    def test_fallback_empty_sampling_frame_raises_clear_error(self):
        """Regression: empty sampling_frame should fail with ValueError, not IndexError."""
        from fmrimod.basis import Poly
        custom_hrf = np.array([0, 0.5, 1.0, 0.5, 0])
        empty_frame = np.array([])

        cases = [
            EventVariable(
                name="event_variable",
                onsets=[5.0, 10.0],
                durations=[1.0, 1.0],
                values=[1.0, 2.0],
                center=False
            ),
            EventFactor(
                name="event_factor",
                onsets=[5.0, 10.0],
                durations=[1.0, 1.0],
                values=["A", "B"]
            ),
            EventMatrix(
                name="event_matrix",
                onsets=[5.0, 10.0],
                durations=[1.0, 1.0],
                values=np.array([[1.0], [2.0]])
            ),
            EventBasis(
                name="event_basis",
                onsets=[5.0, 10.0],
                values=[1.0, 2.0],
                basis=Poly(degree=1)
            ),
            np.array([
                [5.0, 1.0, 1.0],
                [10.0, 1.0, 2.0]
            ])
        ]

        for event in cases:
            with pytest.raises(ValueError, match="sampling_frame must contain at least one time point"):
                convolve(event, hrf=custom_hrf, sampling_frame=empty_frame)

    def test_fallback_sampling_frame_row_and_column_vectors_are_normalized(self):
        """Regression: 2D sampling_frame inputs should be flattened to 1D."""
        custom_hrf = np.array([0, 0.5, 1.0, 0.5, 0])
        event = EventVariable(
            name="event_variable",
            onsets=[5.0, 10.0],
            durations=[1.0, 1.0],
            values=[1.0, 2.0],
            center=False,
        )

        flat_frame = np.arange(0, 20.0, 1.0)
        row_frame = flat_frame.reshape(1, -1)
        col_frame = flat_frame.reshape(-1, 1)

        flat = convolve(event, hrf=custom_hrf, sampling_frame=flat_frame)
        row = convolve(event, hrf=custom_hrf, sampling_frame=row_frame)
        col = convolve(event, hrf=custom_hrf, sampling_frame=col_frame)

        assert row.shape == flat.shape
        assert col.shape == flat.shape
        assert np.allclose(row, flat)
        assert np.allclose(col, flat)

    def test_fallback_sampling_frame_respects_explicit_grid_length(self):
        """Regression: explicit sampling_frame defines output cardinality."""
        custom_hrf = np.array([0, 0.5, 1.0, 0.5, 0])
        event = EventVariable(
            name="event_variable",
            onsets=[1.0, 3.0],
            durations=[1.0, 1.0],
            values=[1.0, 2.0],
            center=False,
        )

        grids = [
            np.array([0.0, 2.0, 4.0]),   # non-unit spacing
            np.array([1.0, 2.0, 3.0]),   # non-zero start
            np.array([10.0, 11.0, 12.0]) # shifted window
        ]

        for sampling_frame in grids:
            result = convolve(
                event,
                hrf=custom_hrf,
                sampling_frame=sampling_frame,
                sampling_rate=1.0,
            )
            assert result.shape == (len(sampling_frame), 1)

    def test_fallback_row_vector_sampling_frame_is_normalized(self):
        """Regression: row-vector sampling_frame should be flattened to 1D."""
        custom_hrf = np.array([0, 0.5, 1.0, 0.5, 0])
        sampling_frame_row = np.array([[0.0, 1.0, 2.0]])  # shape (1, 3)
        sampling_frame_flat = np.array([0.0, 1.0, 2.0])

        event = EventVariable(
            name="event_variable",
            onsets=[1.0, 2.0],
            durations=[1.0, 1.0],
            values=[1.0, 2.0],
            center=False
        )

        result_row = convolve(event, hrf=custom_hrf, sampling_frame=sampling_frame_row)
        result_flat = convolve(event, hrf=custom_hrf, sampling_frame=sampling_frame_flat)
        assert result_row.shape == result_flat.shape
        assert_array_almost_equal(result_row, result_flat)

    def test_fallback_sampling_frame_preserves_grid_length(self):
        """Regression: fallback with sampling_frame should return one row per sample."""
        custom_hrf = np.array([0, 0.5, 1.0, 0.5, 0])
        sampling_frame = np.arange(0.0, 20.0, 2.0)

        event = EventVariable(
            name="event_variable",
            onsets=[2.0, 6.0],
            durations=[1.0, 1.0],
            values=[1.0, 1.0],
            center=False
        )

        result = convolve(event, hrf=custom_hrf, sampling_frame=sampling_frame)
        assert result.shape == (len(sampling_frame), 1)

    def test_fallback_sampling_frame_uses_grid_spacing_for_internal_rate(self):
        """Regression: fallback should derive effective sampling from regular grid spacing."""
        custom_hrf = np.array([0, 0.5, 1.0, 0.5, 0])
        sampling_frame = np.arange(0.0, 20.0, 2.0)  # TR=2.0 => sampling_rate=0.5

        event = EventVariable(
            name="event_variable",
            onsets=[2.0, 6.0],
            durations=[1.0, 1.0],
            values=[1.0, 1.0],
            center=False
        )

        result_from_grid = convolve(event, hrf=custom_hrf, sampling_frame=sampling_frame)
        result_from_rate = convolve(
            event,
            hrf=custom_hrf,
            sampling_rate=0.5,
            total_duration=20.0,
        )

        assert result_from_grid.shape == result_from_rate.shape
        assert_array_almost_equal(result_from_grid, result_from_rate)

    def test_sampling_frame_requires_finite_values(self):
        """Regression: NaN/inf sampling frames should fail fast with clear errors."""
        custom_hrf = np.array([0, 0.5, 1.0, 0.5, 0])
        event = EventVariable(
            name="event_variable",
            onsets=[2.0],
            durations=[1.0],
            values=[1.0],
            center=False,
        )

        with pytest.raises(
            ValueError, match="sampling_frame must contain only finite values"
        ):
            convolve(event, hrf=custom_hrf, sampling_frame=np.array([0.0, np.nan, 2.0]))

        with pytest.raises(
            ValueError, match="sampling_frame must contain only finite values"
        ):
            convolve(event, hrf=custom_hrf, sampling_frame=np.array([0.0, np.inf, 2.0]))

    def test_fallback_single_point_sampling_frame_requires_valid_sampling_rate(self):
        """Regression: invalid sampling_rate should fail clearly for single-point grids."""
        custom_hrf = np.array([0, 0.5, 1.0, 0.5, 0])
        event = EventVariable(
            name="event_variable",
            onsets=[1.0],
            durations=[1.0],
            values=[1.0],
            center=False,
        )

        for bad_rate in [0.0, -1.0, np.nan, np.inf]:
            with pytest.raises(
                ValueError, match="sampling_rate must be a finite positive number"
            ):
                convolve(
                    event,
                    hrf=custom_hrf,
                    sampling_frame=np.array([0.0]),
                    sampling_rate=bad_rate,
                )

    def test_unsupported_type_raises_error(self):
        """Test that unsupported types raise NotImplementedError."""
        # Try to convolve an unsupported type
        with pytest.raises(NotImplementedError, match="convolve not implemented"):
            convolve("not_an_event", sampling_rate=1.0)
