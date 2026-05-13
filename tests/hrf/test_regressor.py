"""Tests for Regressor functionality."""

import pytest
import numpy as np
import typing
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.sparse import issparse

from fmrimod.regressor import Regressor, RegressorSet, neural_input, regressor_design
from fmrimod.regressor.core import regressor, regressor_set
from fmrimod.hrf import HRF, get_hrf


class TestRegressor:
    """Test Regressor class functionality."""
    
    def test_basic_creation(self):
        """Test basic regressor creation."""
        onsets = [10, 30, 50]
        reg = regressor(onsets)
        
        assert isinstance(reg, Regressor)
        assert_array_equal(reg.onsets, onsets)
        assert len(reg.duration) == 3
        assert len(reg.amplitude) == 3
        assert reg.span == 24.0  # Default SPMG1 span
        assert reg.summate is True
    
    def test_parameter_recycling(self):
        """Test recycling of scalar parameters."""
        onsets = [10, 30, 50]
        reg = regressor(onsets, duration=2.0, amplitude=1.5)
        
        assert_array_equal(reg.duration, [2.0, 2.0, 2.0])
        assert_array_equal(reg.amplitude, [1.5, 1.5, 1.5])
    
    def test_array_parameters(self):
        """Test array parameters."""
        onsets = [10, 30, 50]
        durations = [1, 2, 3]
        amplitudes = [1.0, 1.5, 0.8]
        
        reg = regressor(onsets, duration=durations, amplitude=amplitudes)
        
        assert_array_equal(reg.duration, durations)
        assert_array_equal(reg.amplitude, amplitudes)
    
    def test_parameter_length_mismatch(self):
        """Test error on parameter length mismatch."""
        onsets = [10, 30, 50]
        durations = [1, 2]  # Wrong length
        
        with pytest.raises(ValueError, match="Length of duration"):
            regressor(onsets, duration=durations)
    
    def test_zero_amplitude_filtering(self):
        """Test filtering of zero amplitude events."""
        onsets = [10, 30, 50, 70]
        amplitudes = [1.0, 0.0, 1.5, 0.0]  # Two zeros
        
        reg = regressor(onsets, amplitude=amplitudes)
        
        # Should only keep non-zero amplitudes
        assert len(reg.onsets) == 2
        assert_array_equal(reg.onsets, [10, 50])
        assert_array_equal(reg.amplitude, [1.0, 1.5])
    
    def test_nan_amplitude_filtering(self):
        """Test filtering of NaN amplitude events."""
        onsets = [10, 30, 50]
        amplitudes = [1.0, np.nan, 1.5]
        
        reg = regressor(onsets, amplitude=amplitudes)
        
        # Should only keep non-NaN amplitudes
        assert len(reg.onsets) == 2
        assert_array_equal(reg.onsets, [10, 50])
        assert_array_equal(reg.amplitude, [1.0, 1.5])
    
    def test_all_filtered(self):
        """Test when all events are filtered out."""
        onsets = [10, 30, 50]
        amplitudes = [0.0, 0.0, 0.0]  # All zeros
        
        reg = regressor(onsets, amplitude=amplitudes)
        
        assert len(reg.onsets) == 0
        assert reg.filtered_all is True
    
    def test_empty_onsets(self):
        """Test empty onset array."""
        reg = regressor([])
        
        assert len(reg.onsets) == 0
        assert reg.filtered_all is True
    
    def test_single_nan_onset(self):
        """Test single NaN onset (R compatibility)."""
        reg = regressor([np.nan])
        
        assert len(reg.onsets) == 0
        assert reg.filtered_all is True
    
    def test_invalid_onsets(self):
        """Test error on invalid onsets."""
        with pytest.raises(ValueError, match="onsets must contain finite"):
            regressor([10, np.inf, 30])
        
        with pytest.raises(ValueError, match="onsets must be non-negative"):
            regressor([10, -5, 30])
    
    def test_invalid_duration(self):
        """Test error on invalid durations."""
        with pytest.raises(ValueError, match="duration must contain finite"):
            regressor([10, 30], duration=[1, np.nan])
        
        with pytest.raises(ValueError, match="duration cannot be negative"):
            regressor([10, 30], duration=[1, -2])
    
    def test_invalid_span(self):
        """Test error on invalid span."""
        with pytest.raises(ValueError, match="span must be a positive"):
            regressor([10, 30], span=0)
        
        with pytest.raises(ValueError, match="span must be a positive"):
            regressor([10, 30], span=np.inf)
    
    def test_hrf_string(self):
        """Test HRF creation from string."""
        reg = regressor([10, 30], hrf="spmg1")
        
        assert isinstance(reg.hrf, HRF)
        assert reg.hrf.name == "SPMG1"  # HRF names are uppercase
    
    def test_hrf_object(self):
        """Test HRF object input."""
        hrf = get_hrf("gamma")
        reg = regressor([10, 30], hrf=hrf)

        assert reg.hrf is hrf

    def test_direct_regressor_rejects_unresolved_hrf(self):
        """Constructing Regressor directly with a string/callable must raise.

        The string/callable surface lives only on the regressor() factory, so
        the dataclass is honest about what it actually stores.
        """
        with pytest.raises(TypeError, match="must be an HRF"):
            Regressor(
                onsets=np.array([10.0]),
                duration=np.array([0.0]),
                amplitude=np.array([1.0]),
                hrf="spmg1",
            )
        with pytest.raises(TypeError, match="list element"):
            Regressor(
                onsets=np.array([10.0]),
                duration=np.array([0.0]),
                amplitude=np.array([1.0]),
                hrf=["spmg1"],
            )

    def test_neural_input_generation(self):
        """Test neural input generation."""
        onsets = [10, 30, 50]
        durations = [2, 2, 2]
        reg = regressor(onsets, duration=durations)
        
        time, neural = reg.neural_input(start=0, end=60, resolution=1.0)
        
        assert len(time) == len(neural)
        assert time[0] == 0
        assert time[-1] == 60
        
        # Check that neural input is non-zero during events
        assert neural[10] > 0  # First event
        assert neural[11] > 0  # During first event
        assert neural[12] == 0  # After first event
    
    def test_neural_input_auto_end(self):
        """Test neural input with automatic end time."""
        onsets = [10, 30, 50]
        reg = regressor(onsets, duration=2)
        
        time, neural = reg.neural_input(start=0, resolution=1.0)
        
        # Should extend beyond last event
        assert time[-1] >= 52 + 10  # Last onset + duration + buffer
    
    def test_evaluate_basic(self):
        """Test basic regressor evaluation."""
        onsets = [10, 30, 50]
        reg = regressor(onsets, hrf="spmg1")
        
        grid = np.arange(0, 80, 0.5)
        result = reg.evaluate(grid)
        
        assert len(result) == len(grid)
        assert result.ndim == 1  # Single basis function
        
        # Should have responses after each onset
        assert np.any(result[20:25] > 0)  # After first onset
        assert np.any(result[60:65] > 0)  # After second onset
    
    def test_evaluate_multi_basis(self):
        """Test evaluation with multi-basis HRF."""
        onsets = [10, 30, 50]
        hrf = get_hrf("spmg3")  # 3-basis HRF
        reg = regressor(onsets, hrf=hrf)
        
        grid = np.arange(0, 80, 0.5)
        result = reg.evaluate(grid)
        
        assert result.shape == (len(grid), 3)
    
    def test_evaluate_sparse(self):
        """Test sparse evaluation."""
        onsets = [10, 30, 50]
        reg = regressor(onsets)
        
        grid = np.arange(0, 80, 0.5)
        result = reg.evaluate(grid, sparse=True)
        
        assert issparse(result)
        assert result.shape == (len(grid), 1)
    
    def test_evaluate_empty_regressor(self):
        """Test evaluation of empty regressor."""
        reg = regressor([])
        
        grid = np.arange(0, 80, 0.5)
        result = reg.evaluate(grid)
        
        assert_array_equal(result, 0)
    
    def test_evaluate_filtered_all(self):
        """Test evaluation when all events filtered."""
        reg = regressor([10, 30], amplitude=[0, 0])
        
        grid = np.arange(0, 80, 0.5)
        result = reg.evaluate(grid)

        assert_array_equal(result, 0)

    def test_evaluate_type_hints_resolve(self):
        """Regression: evaluate annotations should resolve without NameError."""
        import importlib

        mod = importlib.import_module("fmrimod.regressor.core")
        typing.get_type_hints(mod.Regressor.evaluate)
        typing.get_type_hints(mod.RegressorSet.evaluate)
    
    def test_evaluate_methods(self):
        """Test different evaluation methods give similar results."""
        onsets = [10, 30, 50]
        reg = regressor(onsets, duration=2)
        
        grid = np.arange(0, 80, 1.0)
        
        result_conv = reg.evaluate(grid, method="conv")
        result_fft = reg.evaluate(grid, method="fft")
        
        # Results should be very similar
        assert_array_almost_equal(result_conv, result_fft, decimal=5)
    
    def test_repr(self):
        """Test string representation."""
        reg = regressor([10, 30, 50], hrf="gamma")
        repr_str = repr(reg)
        
        assert "Regressor" in repr_str
        assert "n_events=3" in repr_str
        assert "gamma" in repr_str


class TestRegressorSet:
    """Test RegressorSet functionality."""
    
    def test_basic_creation(self):
        """Test basic regressor set creation."""
        onsets = [10, 20, 30, 40, 50, 60]
        conditions = ['A', 'B', 'C', 'A', 'B', 'C']
        
        rset = regressor_set(onsets, conditions)
        
        assert isinstance(rset, RegressorSet)
        assert len(rset.levels) == 3
        assert set(rset.levels) == {'A', 'B', 'C'}
        assert len(rset.regressors) == 3
    
    def test_numeric_conditions(self):
        """Test numeric condition levels."""
        onsets = [10, 20, 30, 40]
        conditions = [1, 2, 1, 2]
        
        rset = regressor_set(onsets, conditions)
        
        assert set(rset.levels) == {'1', '2'}
    
    def test_evaluate(self):
        """Test regressor set evaluation."""
        onsets = [10, 20, 30, 40, 50, 60]
        conditions = ['A', 'B', 'C', 'A', 'B', 'C']
        
        rset = regressor_set(onsets, conditions)
        grid = np.arange(0, 80, 1.0)
        
        design = rset.evaluate(grid)
        
        assert design.shape == (len(grid), 3)  # 3 conditions
    
    def test_empty_condition(self):
        """Test handling of empty condition."""
        onsets = [10, 20, 30]
        conditions = ['A', 'B', 'A']  # No 'C' events
        
        rset = regressor_set(onsets, conditions, hrf="spmg1")
        
        # Should still create regressor for each unique level
        assert len(rset.levels) == 2
        assert 'A' in rset.levels
        assert 'B' in rset.levels
    
    def test_repr(self):
        """Test string representation."""
        onsets = [10, 20, 30]
        conditions = ['A', 'B', 'C']
        
        rset = regressor_set(onsets, conditions)
        repr_str = repr(rset)
        
        assert "RegressorSet" in repr_str
        assert "n_conditions=3" in repr_str


class TestNeuralInput:
    """Test neural_input function."""
    
    def test_neural_input_function(self):
        """Test standalone neural_input function."""
        reg = regressor([10, 30, 50], duration=2)
        
        result = neural_input(reg, start=0, end=60, resolution=1.0)
        
        assert isinstance(result, dict)
        assert 'time' in result
        assert 'neural_input' in result
        assert len(result['time']) == len(result['neural_input'])


class TestRegressorDesign:
    """Test regressor_design function."""
    
    def test_single_regressor(self):
        """Test design matrix from single regressor."""
        reg = regressor([10, 30, 50])
        grid = np.arange(0, 80, 1.0)
        
        design = regressor_design(reg, grid)
        
        assert design.shape == (len(grid), 1)
    
    def test_regressor_set(self):
        """Test design matrix from regressor set."""
        onsets = [10, 20, 30, 40, 50, 60]
        conditions = ['A', 'B', 'C', 'A', 'B', 'C']
        rset = regressor_set(onsets, conditions)
        
        grid = np.arange(0, 80, 1.0)
        design = regressor_design(rset, grid)
        
        assert design.shape == (len(grid), 3)
    
    def test_list_of_regressors(self):
        """Test design matrix from list of regressors."""
        reg1 = regressor([10, 30, 50])
        reg2 = regressor([20, 40, 60])
        
        grid = np.arange(0, 80, 1.0)
        design = regressor_design([reg1, reg2], grid)
        
        assert design.shape == (len(grid), 2)
    
    def test_include_intercept(self):
        """Test including intercept column."""
        reg = regressor([10, 30, 50])
        grid = np.arange(0, 80, 1.0)
        
        design = regressor_design(reg, grid, include_intercept=True)
        
        assert design.shape == (len(grid), 2)
        assert_array_equal(design[:, 0], 1)  # Intercept column
    
    def test_sparse_output(self):
        """Test sparse output."""
        reg = regressor([10, 30, 50])
        grid = np.arange(0, 80, 1.0)
        
        design = regressor_design(reg, grid, sparse=True)
        
        assert issparse(design)
    
    def test_dataframe_output(self):
        """Test DataFrame output with column names."""
        onsets = [10, 20, 30]
        conditions = ['A', 'B', 'C']
        rset = regressor_set(onsets, conditions)
        
        grid = np.arange(0, 40, 1.0)
        design = regressor_design(
            rset, grid, 
            column_names=['Condition_A', 'Condition_B', 'Condition_C']
        )
        
        import pandas as pd
        assert isinstance(design, pd.DataFrame)
        assert list(design.columns) == ['Condition_A', 'Condition_B', 'Condition_C']
        assert_array_equal(design.index, grid)
    
    def test_column_name_mismatch(self):
        """Test error on column name mismatch."""
        reg = regressor([10, 30, 50])
        grid = np.arange(0, 80, 1.0)
        
        with pytest.raises(ValueError, match="Number of column names"):
            regressor_design(reg, grid, column_names=['A', 'B'])  # Wrong number
    
    def test_invalid_input_type(self):
        """Test error on invalid input type."""
        with pytest.raises(TypeError, match="regressors must be"):
            regressor_design("not a regressor", [0, 1, 2])
