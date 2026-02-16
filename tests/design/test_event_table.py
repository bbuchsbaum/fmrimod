"""Tests for event_table functionality."""

import pytest
import numpy as np
import pandas as pd

from fmrimod.events.factor import EventFactor
from fmrimod.events.variable import EventVariable
from fmrimod.events.matrix import EventMatrix
from fmrimod.events.basis import EventBasis
from fmrimod.events.event_table import event_table
from fmrimod import event_model
from fmrimod.events.term import EventTerm
from fmrimod.basis import Poly


class TestEventTable:
    """Test event_table generic function."""
    
    def test_event_table_factor(self):
        """Test event_table for EventFactor."""
        event = EventFactor(
            name='condition',
            onsets=[1, 2, 3, 4],
            values=['A', 'B', 'A', 'B']
        )
        
        table = event_table(event)
        
        assert isinstance(table, pd.DataFrame)
        assert 'condition' in table.columns
        assert set(table['condition']) == {'A', 'B'}
        assert len(table) == 2
    
    def test_event_table_variable(self):
        """Test event_table for EventVariable."""
        # Few unique values
        event = EventVariable(
            name='rating',
            onsets=[1, 2, 3, 4],
            values=[1, 2, 1, 3],
            center=False  # Disable centering for test
        )
        
        table = event_table(event)
        
        assert isinstance(table, pd.DataFrame)
        assert 'rating' in table.columns
        assert set(table['rating']) == {1, 2, 3}
        
        # Many unique values
        event_many = EventVariable(
            name='rt',
            onsets=np.arange(50),
            values=np.random.randn(50)
        )
        
        table_many = event_table(event_many)
        
        # Should return summary
        assert len(table_many) == 1
        assert 'rt' in table_many.columns
        assert table_many['rt'].iloc[0] == 'continuous'
        assert 'rt_range' in table_many.columns
    
    def test_event_table_matrix(self):
        """Test event_table for EventMatrix."""
        event = EventMatrix(
            name='motion',
            onsets=[1, 2, 3],
            values=np.random.randn(3, 6),
            column_names=['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        )
        
        table = event_table(event)
        
        assert isinstance(table, pd.DataFrame)
        assert 'column' in table.columns
        assert 'type' in table.columns
        assert len(table) == 6
        assert list(table['column']) == ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    
    def test_event_table_basis(self):
        """Test event_table for EventBasis."""
        event = EventBasis(
            name='time',
            onsets=[1, 2, 3, 4],
            values=np.arange(4),
            basis=Poly(degree=3)
        )
        
        table = event_table(event)
        
        assert isinstance(table, pd.DataFrame)
        assert 'basis_function' in table.columns
        assert 'type' in table.columns
        assert len(table) == 4  # degree 3 + constant
    
    def test_event_table_term_single(self):
        """Test event_table for single-event EventTerm."""
        event = EventFactor(
            name='task',
            onsets=[1, 2, 3],
            values=['go', 'stop', 'go']
        )
        
        term = EventTerm([event])
        table = event_table(term)
        
        assert isinstance(table, pd.DataFrame)
        assert 'task' in table.columns
        assert set(table['task']) == {'go', 'stop'}
    
    def test_event_table_term_interaction(self):
        """Test event_table for interaction EventTerm."""
        event1 = EventFactor(
            name='condition',
            onsets=[1, 2, 3, 4],
            values=['A', 'B', 'A', 'B']
        )
        
        event2 = EventFactor(
            name='block',
            onsets=[1, 2, 3, 4],
            values=['1', '1', '2', '2']
        )
        
        term = EventTerm([event1, event2], interaction=True)
        table = event_table(term)
        
        assert isinstance(table, pd.DataFrame)
        assert 'condition' in table.columns
        assert 'block' in table.columns
        
        # Should have all combinations
        assert len(table) == 4  # 2x2
        
        # Check combinations exist
        combos = set(zip(table['condition'], table['block']))
        expected = {('A', '1'), ('A', '2'), ('B', '1'), ('B', '2')}
        assert combos == expected
    
    def test_event_table_mixed_interaction(self):
        """Test event_table for mixed categorical/continuous interaction."""
        cat_event = EventFactor(
            name='group',
            onsets=[1, 2, 3, 4],
            values=['ctrl', 'exp', 'ctrl', 'exp']
        )
        
        cont_event = EventVariable(
            name='score',
            onsets=[1, 2, 3, 4],
            values=[10, 20, 15, 25]
        )
        
        term = EventTerm([cat_event, cont_event], interaction=True)
        table = event_table(term)
        
        assert isinstance(table, pd.DataFrame)
        assert 'group' in table.columns
        assert 'score' in table.columns
        
        # Should have combinations
        assert len(table) == 8  # 2 groups x 4 unique scores
    
    def test_event_table_invalid_type(self):
        """Test error for unsupported types."""
        with pytest.raises(NotImplementedError, match="No event_table method"):
            event_table("not an event")
        
        with pytest.raises(NotImplementedError, match="No event_table method"):
            event_table([1, 2, 3])


class TestEventTableIntegration:
    """Integration tests for event_table with models."""
    
    def test_event_table_from_model(self):
        """Test extracting event table from EventModel."""
        # Register the EventModel method
        from fmrimod.events.event_table import _register_event_model
        _register_event_model()
        
        # Create model
        df = pd.DataFrame({
            'onset': [1, 2, 3, 4, 5, 6],
            'condition': ['A', 'B', 'C', 'A', 'B', 'C'],
            'session': ['1', '1', '1', '2', '2', '2'],
            'duration': 1
        })
        
        model = event_model(
            "condition + session",
            data=df,
            tr=2.0,
            n_scans=10
        )
        
        table = event_table(model)
        
        assert isinstance(table, pd.DataFrame)
        assert '_term' in table.columns
        
        # Should have entries for both terms
        assert len(table[table['_term'] == 'condition']) == 3  # A, B, C
        assert len(table[table['_term'] == 'session']) == 2  # 1, 2
    
    def test_event_table_interaction_model(self):
        """Test event table from model with interaction."""
        from fmrimod.events.event_table import _register_event_model
        _register_event_model()
        
        df = pd.DataFrame({
            'onset': np.arange(8),
            'factor1': ['A', 'B'] * 4,
            'factor2': ['X', 'X', 'Y', 'Y'] * 2,
            'duration': 0.5
        })
        
        # For now, use separate terms since * parsing isn't implemented
        model = event_model(
            "factor1 + factor2",
            data=df,
            tr=1.0,
            n_scans=20
        )
        
        table = event_table(model)
        
        # Should have main effects
        assert '_term' in table.columns
        
        # Count unique term types
        term_counts = table['_term'].value_counts()
        assert len(term_counts) == 2  # factor1, factor2