"""Tests for utility functions."""

import pytest
import numpy as np
import pandas as pd

from fmrimod.events.factor import EventFactor
from fmrimod.events.variable import EventVariable
from fmrimod.events.matrix import EventMatrix
from fmrimod.events.basis import EventBasis
from fmrimod import event_model, baseline_model
from fmrimod.basis import Poly, BSpline
from fmrimod.utils import term_indices, term_matrices, baseline_terms, split_onsets, split_by_block
from fmrimod.sampling import SamplingFrame


class TestTermUtils:
    """Test term utility functions."""
    
    def test_term_indices_simple(self):
        """Test term_indices with simple model."""
        # Create sample data
        onsets = np.array([1, 5, 10, 15, 20, 25])
        conditions = ['A', 'B', 'A', 'B', 'A', 'B']
        df = pd.DataFrame({
            'onset': onsets,
            'condition': conditions
        })
        
        # Create model
        model = event_model("condition", data=df, tr=2.0, n_scans=150)
        
        # Get indices
        indices = term_indices(model)
        
        # Should have condition term
        assert 'condition' in indices
        assert len(indices['condition']) == 2  # A and B levels
        
    def test_term_indices_multiple_terms(self):
        """Test term_indices with multiple terms."""
        # Create data
        onsets = np.array([1, 5, 10, 15])
        df = pd.DataFrame({
            'onset': onsets,
            'condition': ['A', 'B', 'A', 'B'],
            'rating': [1.2, 2.1, 3.2, 1.8]
        })
        
        # Create model with multiple terms
        model = event_model(
            "condition + rating",
            data=df,
            tr=2.0,
            n_scans=150
        )
        
        indices = term_indices(model)
        
        # Should have both terms
        assert 'condition' in indices
        assert 'rating' in indices
        assert len(indices['condition']) == 2  # A and B
        assert len(indices['rating']) == 1  # continuous variable
        
    def test_term_indices_basis_function(self):
        """Test term_indices with basis function expansion."""
        # Create data with basis function
        onsets = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        df = pd.DataFrame({
            'onset': onsets,
            'rating': np.linspace(0, 1, 8)
        })
        
        # Model with basis expansion
        from fmrimod.formula.base import Term
        from fmrimod.basis import Poly
        
        terms = [Term('rating', basis=Poly(degree=2))]
        model = event_model(terms, data=df, tr=2.0, n_scans=150)
        
        indices = term_indices(model)
        
        # Should have rating term - the exact number of columns depends on implementation
        assert 'rating' in indices
        assert len(indices['rating']) >= 1  # At least the term itself
        
    def test_term_matrices_extraction(self):
        """Test extracting term-specific matrices."""
        # Create data
        onsets = np.array([1, 5, 10, 15, 20])
        values = np.array([1.2, 2.3, 3.1, 1.8, 2.5])
        df = pd.DataFrame({
            'onset': onsets,
            'rating': values
        })
        
        # Model with continuous variable
        model = event_model("rating", data=df, tr=2.0, n_scans=150)
        
        # Extract matrices
        matrices = term_matrices(model)
        
        assert 'rating' in matrices
        assert matrices['rating'].shape[1] == 1  # Single column for continuous
        
    def test_term_matrices_subset(self):
        """Test extracting specific terms."""
        # Create multi-term model
        onsets = np.arange(10)
        df = pd.DataFrame({
            'onset': onsets,
            'cond': ['A', 'B'] * 5,
            'rating': np.random.randn(10)
        })
        
        model = event_model("cond + rating", data=df, tr=2.0, n_scans=150)
        
        # Extract only condition
        cond_only = term_matrices(model, 'cond')
        assert len(cond_only) == 1
        assert 'cond' in cond_only
        
        # Extract both terms  
        both = term_matrices(model, ['cond', 'rating'])
        assert len(both) == 2
        assert 'cond' in both
        assert 'rating' in both
        
    def test_baseline_terms_not_implemented(self):
        """Test baseline_terms returns None for models without baseline."""
        df = pd.DataFrame({
            'onset': [1, 5, 10],
            'cond': ['A', 'B', 'A']
        })
        
        # Model without baseline attribute
        model = event_model(
            "cond",
            data=df,
            tr=2.0,
            n_scans=150
        )
        
        baseline = baseline_terms(model)
        
        # Currently returns None as baseline not integrated
        assert baseline is None
        
    def test_baseline_terms_absent(self):
        """Test baseline_terms returns None when no baseline."""
        df = pd.DataFrame({
            'onset': [1, 5, 10],
            'cond': ['A', 'B', 'A'] 
        })
        
        model = event_model("cond", data=df, tr=2.0, n_scans=150)
        
        baseline = baseline_terms(model)
        assert baseline is None


class TestEventUtils:
    """Test event utility functions."""
    
    def test_split_onsets_by_values(self):
        """Test splitting onsets by event values."""
        onsets = np.array([1, 2, 3, 4, 5, 6])
        values = ['happy', 'sad', 'happy', 'neutral', 'sad', 'happy']
        
        event = EventFactor('emotion', onsets, values)
        
        splits = split_onsets(event, by='values')
        
        assert len(splits) == 3
        assert 'happy' in splits
        assert 'sad' in splits  
        assert 'neutral' in splits
        
        # Check correct assignment
        np.testing.assert_array_equal(splits['happy'], [1, 3, 6])
        np.testing.assert_array_equal(splits['sad'], [2, 5])
        np.testing.assert_array_equal(splits['neutral'], [4])
        
    def test_split_onsets_by_function(self):
        """Test splitting onsets by custom function."""
        onsets = np.array([1, 10, 20, 30, 40, 50])
        values = np.array([1, 10, 20, 30, 40, 50])
        event = EventVariable('timing', onsets, values=values)
        
        # Split by early/late - pass onsets as values since we want to split by onset time
        def early_late(onsets):
            threshold = 25
            return ['early' if o < threshold else 'late' for o in onsets]
        
        splits = split_onsets(event, by=early_late, values=onsets)
        
        assert len(splits) == 2
        np.testing.assert_array_equal(splits['early'], [1, 10, 20])
        np.testing.assert_array_equal(splits['late'], [30, 40, 50])
        
    def test_split_onsets_external_values(self):
        """Test splitting with external grouping values."""
        onsets = np.array([1, 2, 3, 4, 5])
        event = EventVariable('response', onsets, values=[1, 2, 3, 4, 5])
        
        # External block assignment
        blocks = ['A', 'A', 'B', 'B', 'A']
        
        splits = split_onsets(event, by=lambda x: blocks, values=blocks)
        
        assert len(splits) == 2
        np.testing.assert_array_equal(splits['A'], [1, 2, 5])
        np.testing.assert_array_equal(splits['B'], [3, 4])
        
    def test_split_by_block_labels(self):
        """Test splitting event by block labels."""
        onsets = np.array([1, 5, 10, 15, 20, 25])
        values = ['stim1', 'stim2', 'stim1', 'stim2', 'stim1', 'stim2']
        blocks = ['run1', 'run1', 'run1', 'run2', 'run2', 'run2']
        
        # Create event with block info
        event = EventFactor('stimulus', onsets, values)
        event.block = blocks
        
        # Split by blocks
        block_events = split_by_block(event, 'block')
        
        assert len(block_events) == 2
        assert 'run1' in block_events
        assert 'run2' in block_events
        
        # Check run1
        run1 = block_events['run1']
        assert len(run1.onsets) == 3
        np.testing.assert_array_equal(run1.onsets, [1, 5, 10])
        np.testing.assert_array_equal(run1.values, ['stim1', 'stim2', 'stim1'])
        
        # Check run2
        run2 = block_events['run2']
        assert len(run2.onsets) == 3
        np.testing.assert_array_equal(run2.onsets, [15, 20, 25])
        
    def test_split_by_block_timing(self):
        """Test splitting by block onset/duration timing."""
        # Events spread across time
        onsets = np.array([5, 15, 25, 50, 75, 85, 110, 125])
        event = EventVariable('response', onsets, values=np.arange(8))
        
        # Define blocks by timing
        block_starts = [0, 40, 80, 120]
        block_durations = [40, 40, 40, 40]
        block_names = ['block1', 'block2', 'block3', 'block4']
        
        blocks = split_by_block(
            event, 
            block_names,
            block_onsets=block_starts,
            block_durations=block_durations
        )
        
        # Check assignments
        assert len(blocks) == 4
        assert len(blocks['block1'].onsets) == 3  # 5, 15, 25
        assert len(blocks['block2'].onsets) == 2  # 50, 75
        assert len(blocks['block3'].onsets) == 2  # 85, 110 
        assert len(blocks['block4'].onsets) == 1  # 125
        
    def test_split_by_block_preserves_type(self):
        """Test that split_by_block preserves event type."""
        # Test with EventBasis
        onsets = np.array([1, 5, 10, 15])
        values = np.array([1.2, 2.1, 3.3, 1.8])
        basis = Poly(degree=2)
        event = EventBasis('poly_event', onsets, values, basis)
        
        # Use block labels array instead of attribute
        block_labels = ['A', 'A', 'B', 'B']
        blocks = split_by_block(event, block_labels)
        
        # Check types preserved
        assert isinstance(blocks['A'], EventBasis)
        assert isinstance(blocks['B'], EventBasis)
        assert blocks['A'].basis is basis
        assert blocks['B'].basis is basis