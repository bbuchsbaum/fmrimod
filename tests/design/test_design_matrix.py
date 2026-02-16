"""Tests for design_matrix generic function."""

import pytest
import numpy as np
import pandas as pd

from fmrimod import event_model, design_matrix
from fmrimod.design.event_model import EventModel
from fmrimod.events.factor import EventFactor
from fmrimod.formula import Term
from fmrimod.sampling import SamplingFrame


class TestDesignMatrix:
    """Test design_matrix generic function."""
    
    def test_design_matrix_from_event_model(self):
        """Test extracting design matrix from EventModel."""
        # Create a simple model
        events = {
            'condition': EventFactor(
                name='condition',
                onsets=[5, 10, 15, 20],
                values=['A', 'B', 'A', 'B'],
                durations=1
            )
        }
        
        sampling = SamplingFrame(tr=2.0, n_scans=15)
        terms = [Term('condition')]
        
        model = EventModel(
            terms=terms,
            events=events,
            sampling_info=sampling
        )
        
        # Extract design matrix
        X = design_matrix(model)
        
        # Should be same as model.design_matrix
        assert np.array_equal(X, model.design_matrix)
        assert X.shape == (15, 2)
    
    def test_design_matrix_with_blockid(self):
        """Test subsetting design matrix by block."""
        # Create multi-block data
        df = pd.DataFrame({
            'onset': [5, 10, 15, 55, 60, 65],
            'condition': ['A', 'B', 'A', 'A', 'B', 'A'],
            'block': [1, 1, 1, 2, 2, 2],
            'duration': 1
        })
        
        # Create model with blocks
        model = event_model(
            "condition",
            data=df,
            tr=2.0,
            n_scans=40,  # 20 scans per block
            # Note: block handling might need more work
        )
        
        # Get full design matrix
        X_full = design_matrix(model)
        assert X_full.shape == (40, 2)
        
        # Single-block model: all timepoints belong to block 1
        # So requesting blockid=1 should return the full matrix
        X_block1 = design_matrix(model, blockid=1)
        assert X_block1.shape == (40, 2)
        np.testing.assert_array_equal(X_block1, X_full)

        # Requesting a non-existent block should warn and return empty
        with pytest.warns(UserWarning):
            X_block99 = design_matrix(model, blockid=99)
        assert X_block99.shape == (0, 2)
    
    def test_design_matrix_invalid_type(self):
        """Test error for unsupported types."""
        with pytest.raises(NotImplementedError, match="No design_matrix method"):
            design_matrix("not a model")
        
        with pytest.raises(NotImplementedError, match="No design_matrix method"):
            design_matrix([1, 2, 3])
    
    def test_design_matrix_from_formula(self):
        """Test design_matrix via event_model constructor."""
        df = pd.DataFrame({
            'onset': [5, 10, 15, 20, 25],
            'condition': ['A', 'B', 'A', 'B', 'A'],
            'rating': [1, 2, 3, 4, 5],
            'duration': 1
        })
        
        # Create model from formula
        model = event_model(
            "condition + rating",
            data=df,
            tr=1.0,
            n_scans=30
        )
        
        # Extract design matrix
        X = design_matrix(model)
        
        # Should have 3 columns (2 for condition, 1 for rating)
        assert X.shape == (30, 3)
        assert not np.all(X == 0)  # Should have some non-zero values
    
    def test_design_matrix_consistency(self):
        """Test that design_matrix() and model.design_matrix are consistent."""
        # Create various models and check consistency
        df = pd.DataFrame({
            'onset': np.arange(0, 50, 5),
            'cond': ['A', 'B'] * 5,
            'continuous': np.random.randn(10),
            'duration': 0.5
        })
        
        models = [
            event_model("cond", data=df, tr=2.0, n_scans=30),
            event_model("continuous", data=df, tr=2.0, n_scans=30),
            event_model("cond + continuous", data=df, tr=2.0, n_scans=30),
        ]
        
        for model in models:
            X_generic = design_matrix(model)
            X_direct = model.design_matrix
            
            assert np.array_equal(X_generic, X_direct)
            assert X_generic.shape == X_direct.shape
            
            # Both should be numpy arrays
            assert isinstance(X_generic, np.ndarray)
            assert isinstance(X_direct, np.ndarray)


class TestDesignMatrixIntegration:
    """Integration tests for design_matrix function."""
    
    def test_with_hrf_convolution(self):
        """Test design matrix extraction with HRF."""
        events = {
            'stimulus': EventFactor(
                name='stimulus',
                onsets=[5, 15, 25],
                values=['vis', 'aud', 'vis'],
                durations=2.0
            )
        }
        
        sampling = SamplingFrame(tr=1.0, n_scans=40)
        terms = [Term('stimulus', hrf='simple')]
        
        model = EventModel(
            terms=terms,
            events=events,
            sampling_info=sampling
        )
        
        X = design_matrix(model)
        
        # Should have convolved response
        assert X.shape == (40, 2)  # 2 stimulus types
        assert np.max(X) > 0
        assert not np.all(X[0:3] > 0)  # Shouldn't respond immediately
    
    def test_empty_result_with_invalid_blockid(self):
        """Test that invalid blockid returns empty array."""
        events = {
            'task': EventFactor(
                name='task',
                onsets=[5, 10],
                values=['A', 'B']
            )
        }
        
        model = EventModel(
            terms=[Term('task')],
            events=events,
            sampling_info=SamplingFrame(tr=2.0, n_scans=20)
        )
        
        # Test with definitely invalid blockid
        with pytest.warns(UserWarning, match="blockid"):
            X_empty = design_matrix(model, blockid=999)
        
        # Should return empty array with correct number of columns
        assert X_empty.shape[0] == 0
        assert X_empty.shape[1] == model.design_matrix.shape[1]