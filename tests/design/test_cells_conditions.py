"""Tests for cells and conditions functionality."""

import pytest
import numpy as np
import pandas as pd

from fmrimod.events.factor import EventFactor
from fmrimod.events.variable import EventVariable
from fmrimod.events.term import EventTerm
from fmrimod.design.event_model import EventModel, event_model
from fmrimod.formula.base import Term
from fmrimod.sampling import SamplingFrame


class TestCellsConditions:
    """Test cells and conditions extraction."""
    
    def test_cells_categorical_single(self):
        """Test cells extraction for single categorical event."""
        # Create categorical event
        onsets = np.array([1, 2, 3, 4, 5, 6])
        values = ['A', 'B', 'A', 'C', 'B', 'A']
        event = EventFactor('condition', onsets, values)
        
        # Create term
        term = EventTerm([event])
        
        # Get cells
        cells_df = term.cells(drop_empty=True)
        
        # Check structure
        assert 'condition' in cells_df.columns
        assert hasattr(cells_df, 'attrs')
        assert 'count' in cells_df.attrs
        
        # Check counts
        counts = cells_df.attrs['count']
        assert len(counts) == 3  # Three unique levels
        assert sum(counts) == 6  # Total observations
        
        # Check levels are represented
        levels = cells_df['condition'].tolist()
        assert set(levels) == {'A', 'B', 'C'}
    
    def test_cells_categorical_interaction(self):
        """Test cells extraction for categorical interaction."""
        # Create two categorical events
        onsets = np.array([1, 2, 3, 4, 5, 6])
        cond_values = ['A', 'B', 'A', 'B', 'A', 'B']
        block_values = ['1', '1', '2', '2', '3', '3']
        
        cond_event = EventFactor('condition', onsets, cond_values)
        block_event = EventFactor('block', onsets, block_values)
        
        # Create interaction term
        term = EventTerm([cond_event, block_event], interaction=True)
        
        # Get cells
        cells_df = term.cells(drop_empty=False)
        
        # Check structure
        assert 'condition' in cells_df.columns
        assert 'block' in cells_df.columns
        
        # Should have all combinations (2 conditions × 3 blocks = 6)
        assert len(cells_df) == 6
        
        # Check counts
        counts = cells_df.attrs['count']
        assert all(c == 1 for c in counts)  # Each combination appears once
    
    def test_cells_continuous(self):
        """Test cells extraction for continuous event."""
        # Create continuous event
        onsets = np.array([1, 2, 3, 4, 5])
        values = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
        event = EventVariable('rating', onsets, values)
        
        # Create term
        term = EventTerm([event])
        
        # Get cells
        cells_df = term.cells()
        
        # For continuous, should have single row
        assert len(cells_df) == 1
        assert 'rating' in cells_df.columns
        assert cells_df.attrs['count'][0] == 5  # Number of observations
    
    def test_conditions_categorical(self):
        """Test conditions extraction for categorical event."""
        # Create categorical event
        onsets = np.array([1, 2, 3, 4, 5, 6])
        values = ['face', 'house', 'face', 'house', 'face', 'house']
        event = EventFactor('stimulus', onsets, values)
        
        # Create term
        term = EventTerm([event])
        
        # Get conditions
        conditions = term.conditions()
        
        # Should have condition for each level
        assert len(conditions) == 2
        assert 'stimulus.face' in conditions
        assert 'stimulus.house' in conditions
    
    def test_conditions_continuous(self):
        """Test conditions extraction for continuous event."""
        # Create continuous event
        onsets = np.array([1, 2, 3, 4, 5])
        values = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
        event = EventVariable('rating', onsets, values)
        
        # Create term
        term = EventTerm([event])
        
        # Get conditions
        conditions = term.conditions()
        
        # Should have single condition
        assert len(conditions) == 1
        assert conditions[0] == 'rating'
    
    def test_conditions_interaction(self):
        """Test conditions extraction for interaction."""
        # Create events
        onsets = np.array([1, 2, 3, 4])
        cond_values = ['A', 'B', 'A', 'B']
        rating_values = np.array([0.1, 0.5, 0.3, 0.8])
        
        cond_event = EventFactor('condition', onsets, cond_values)
        rating_event = EventVariable('rating', onsets, rating_values)
        
        # Create interaction term
        term = EventTerm([cond_event, rating_event], interaction=True)
        
        # Get conditions
        conditions = term.conditions()
        
        # Should have conditions for each level × continuous
        assert len(conditions) == 2
        assert 'condition.A_rating' in conditions
        assert 'condition.B_rating' in conditions
    
    def test_conditions_expand_basis(self):
        """Test conditions with basis expansion."""
        # Create event and term with basis info
        onsets = np.array([1, 2, 3, 4, 5])
        values = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
        event = EventVariable('rating', onsets, values)
        
        term = EventTerm([event])
        term.nbasis = 3  # Simulate 3 basis functions
        
        # Get conditions without expansion
        conditions_no_expand = term.conditions(expand_basis=False)
        assert len(conditions_no_expand) == 1
        assert conditions_no_expand[0] == 'rating'
        
        # Get conditions with expansion
        conditions_expand = term.conditions(expand_basis=True)
        assert len(conditions_expand) == 3
        assert 'rating_b1' in conditions_expand
        assert 'rating_b2' in conditions_expand
        assert 'rating_b3' in conditions_expand
    
    def test_event_model_cells(self):
        """Test cells extraction from event model."""
        # Create events
        onsets = np.array([1, 2, 3, 4, 5, 6])
        cond_values = ['A', 'B', 'A', 'B', 'A', 'B']
        rating_values = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.6])
        
        events = {
            'condition': EventFactor('condition', onsets, cond_values),
            'rating': EventVariable('rating', onsets, rating_values)
        }
        
        # Create model with two terms
        terms = [
            Term('condition'),
            Term('rating')
        ]
        
        sampling_info = SamplingFrame(tr=2.0, n_scans=10)
        model = EventModel(terms, events, sampling_info)
        
        # Get cells for all terms
        all_cells = model.cells()
        
        # Should have list with 2 DataFrames
        assert len(all_cells) == 2
        assert isinstance(all_cells[0], pd.DataFrame)
        assert isinstance(all_cells[1], pd.DataFrame)
        
        # First should be categorical
        assert 'condition' in all_cells[0].columns
        assert len(all_cells[0]) == 2  # Two levels
        
        # Second should be continuous
        assert 'rating' in all_cells[1].columns
        assert len(all_cells[1]) == 1  # Single row
    
    def test_event_model_conditions(self):
        """Test conditions extraction from event model."""
        # Create events
        onsets = np.array([1, 2, 3, 4, 5, 6])
        cond_values = ['face', 'house', 'face', 'house', 'face', 'house']
        rating_values = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.6])
        
        events = {
            'stimulus': EventFactor('stimulus', onsets, cond_values),
            'rating': EventVariable('rating', onsets, rating_values)
        }
        
        # Create model
        terms = [
            Term('stimulus'),
            Term('rating')
        ]
        
        sampling_info = SamplingFrame(tr=2.0, n_scans=10)
        model = EventModel(terms, events, sampling_info)
        
        # Get conditions for all terms
        all_conditions = model.conditions()
        
        # Should have list with 2 lists
        assert len(all_conditions) == 2
        
        # First term conditions
        assert len(all_conditions[0]) == 2
        assert 'stimulus.face' in all_conditions[0]
        assert 'stimulus.house' in all_conditions[0]
        
        # Second term conditions
        assert len(all_conditions[1]) == 1
        assert all_conditions[1][0] == 'rating'
    
    def test_shortnames_longnames(self):
        """Test short and long name generation."""
        # Create a simple model
        onsets = np.array([1, 2, 3, 4])
        cond_values = ['A', 'B', 'A', 'B']
        
        events = {
            'condition': EventFactor('condition', onsets, cond_values)
        }
        
        terms = [Term('condition')]
        sampling_info = SamplingFrame(tr=2.0, n_scans=10)
        model = EventModel(terms, events, sampling_info)
        
        # Get long names (same as column_names)
        long_names = model.longnames()
        assert isinstance(long_names, list)
        assert len(long_names) > 0
        
        # Get short names
        short_names = model.shortnames()
        assert isinstance(short_names, list)
        assert len(short_names) == len(long_names)
        
        # Short names should be shorter or equal
        for short, long in zip(short_names, long_names):
            assert len(short) <= len(long)
        
        # Test with custom acronym
        short_with_acronym = model.shortnames(acronym='TEST')
        assert all('TEST' in name or name == long 
                  for name, long in zip(short_with_acronym, long_names))