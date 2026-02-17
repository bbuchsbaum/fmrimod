"""Tests for event components."""

import pytest
import numpy as np
import pandas as pd

from fmrimod.events import (
    EventFactor,
    EventVariable,
    EventMatrix,
    EventBasis,
    create_event,
    events_from_dataframe,
)
from fmrimod.events.term import EventTerm, create_interaction


class TestEventFactor:
    """Test EventFactor class."""
    
    def test_simple_factor(self):
        """Test creating a simple categorical event."""
        event = EventFactor(
            name='condition',
            onsets=[1, 2, 3, 4],
            values=['A', 'B', 'A', 'B']
        )
        
        assert event.name == 'condition'
        assert event.n_events == 4
        assert event.n_levels == 2
        assert event.levels == ['A', 'B']
        assert event.event_type == 'categorical'
    
    def test_factor_with_explicit_levels(self):
        """Test factor with explicit level ordering."""
        event = EventFactor(
            name='difficulty',
            onsets=[1, 2, 3],
            values=['easy', 'hard', 'medium'],
            levels=['easy', 'medium', 'hard']
        )
        
        assert event.levels == ['easy', 'medium', 'hard']
        assert list(event.values.categories) == ['easy', 'medium', 'hard']
    
    def test_factor_with_durations(self):
        """Test factor with varying durations."""
        event = EventFactor(
            name='task',
            onsets=[0, 5, 10],
            values=['A', 'B', 'A'],
            durations=[2, 3, 2]
        )
        
        assert np.array_equal(event.durations, [2, 3, 2])
    
    def test_get_level_methods(self):
        """Test methods for accessing level-specific data."""
        event = EventFactor(
            name='condition',
            onsets=[1, 2, 3, 4],
            values=['A', 'B', 'A', 'B'],
            durations=[1, 1, 2, 2]
        )
        
        # Test level indices
        a_indices = event.get_level_indices('A')
        assert np.array_equal(a_indices, [True, False, True, False])
        
        # Test level onsets
        a_onsets = event.get_level_onsets('A')
        assert np.array_equal(a_onsets, [1, 3])
        
        # Test level durations
        b_durations = event.get_level_durations('B')
        assert np.array_equal(b_durations, [1, 2])
    
    def test_design_matrix(self):
        """Test design matrix generation."""
        event = EventFactor(
            name='condition',
            onsets=[0, 2, 4],
            values=['A', 'B', 'A'],
            durations=1
        )
        
        # Sample at 1 Hz
        sampling_points = np.arange(0, 6, 1)
        X = event.design_matrix(sampling_points)
        
        assert X.shape == (6, 2)  # 6 timepoints, 2 levels
        
        # Check indicators
        expected_A = [1, 0, 0, 0, 1, 0]
        expected_B = [0, 0, 1, 0, 0, 0]
        
        assert np.array_equal(X[:, 0], expected_A)
        assert np.array_equal(X[:, 1], expected_B)

    def test_design_matrix_duplicate_impulses_superpose_same_level(self):
        """Duplicate impulses for a level should accumulate counts."""
        event = EventFactor(
            name='condition',
            onsets=[1, 1],
            values=['A', 'A'],
            durations=0
        )

        sampling_points = np.arange(0, 4, 1)
        X = event.design_matrix(sampling_points)

        assert X.shape == (4, 1)
        assert X[1, 0] == 2.0

    def test_design_matrix_overlapping_durations_superpose_same_level(self):
        """Overlapping duration windows for a level should accumulate."""
        event = EventFactor(
            name='condition',
            onsets=[1, 2],
            values=['A', 'A'],
            durations=[3, 2]
        )

        sampling_points = np.arange(0, 6, 1)
        X = event.design_matrix(sampling_points)

        # First event active at 1,2,3. Second active at 2,3.
        assert X[1, 0] == 1.0
        assert X[2, 0] == 2.0
        assert X[3, 0] == 2.0
    
    def test_split_by_level(self):
        """Test splitting into separate events by level."""
        event = EventFactor(
            name='condition',
            onsets=[1, 2, 3, 4],
            values=['A', 'B', 'A', 'B']
        )
        
        split = event.split_by_level()
        
        assert len(split) == 2
        assert 'A' in split and 'B' in split
        assert split['A'].n_events == 2
        assert split['B'].n_events == 2
    
    def test_from_dataframe(self):
        """Test creating from DataFrame."""
        df = pd.DataFrame({
            'onset': [1, 2, 3, 4],
            'condition': ['A', 'B', 'A', 'B'],
            'duration': [1, 1, 2, 2]
        })
        
        event = EventFactor.from_dataframe(df, 'condition')
        
        assert event.name == 'condition'
        assert event.n_events == 4
        assert list(event.values) == ['A', 'B', 'A', 'B']


class TestEventVariable:
    """Test EventVariable class."""
    
    def test_simple_variable(self):
        """Test creating a simple continuous event."""
        event = EventVariable(
            name='rating',
            onsets=[1, 2, 3, 4],
            values=[7.5, 3.2, 8.1, 5.5],
            center=False
        )
        
        assert event.name == 'rating'
        assert event.n_events == 4
        assert event.event_type == 'continuous'
        assert np.array_equal(event.values, event.raw_values)
    
    def test_centering(self):
        """Test value centering."""
        values = [2, 4, 6, 8]
        event = EventVariable(
            name='rating',
            onsets=[1, 2, 3, 4],
            values=values,
            center=True,
            scale=False
        )
        
        assert np.mean(event.values) == pytest.approx(0, abs=1e-10)
        assert event.mean == 5.0  # Mean of raw values
    
    def test_scaling(self):
        """Test value scaling."""
        values = [2, 4, 6, 8]
        event = EventVariable(
            name='rating',
            onsets=[1, 2, 3, 4],
            values=values,
            center=True,
            scale=True
        )
        
        assert np.std(event.values) == pytest.approx(1.0, abs=1e-10)
    
    def test_design_matrix_impulse(self):
        """Test design matrix for impulse events."""
        event = EventVariable(
            name='rating',
            onsets=[0, 2, 4],
            values=[1, 2, 3],
            durations=0,  # Impulse
            center=False
        )
        
        sampling_points = np.arange(0, 6, 1)
        X = event.design_matrix(sampling_points)
        
        assert X.shape == (6, 1)
        expected = [[1], [0], [2], [0], [3], [0]]
        assert np.array_equal(X, expected)
    
    def test_design_matrix_extended(self):
        """Test design matrix for extended events."""
        event = EventVariable(
            name='rating',
            onsets=[0, 3],
            values=[1, 2],
            durations=2,
            center=False
        )
        
        sampling_points = np.arange(0, 6, 1)
        X = event.design_matrix(sampling_points)
        
        assert X.shape == (6, 1)
        expected = [[1], [1], [0], [2], [2], [0]]
        assert np.array_equal(X, expected)
    
    def test_bin_values(self):
        """Test binning continuous values."""
        event = EventVariable(
            name='rating',
            onsets=[1, 2, 3, 4, 5, 6],
            values=[1, 2, 3, 4, 5, 6],
            center=False
        )
        
        binned = event.bin_values(n_bins=3)
        
        assert isinstance(binned, EventFactor)
        assert binned.n_levels == 3
        assert binned.n_events == 6


class TestEventMatrix:
    """Test EventMatrix class."""
    
    def test_simple_matrix(self):
        """Test creating a matrix event."""
        values = [[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]
        
        event = EventMatrix(
            name='motion',
            onsets=[0, 1, 2],
            values=values,
            column_names=['x', 'y', 'z']
        )
        
        assert event.name == 'motion'
        assert event.n_events == 3
        assert event.n_columns == 3
        assert event.column_names == ['x', 'y', 'z']
        assert event.event_type == 'matrix'
    
    def test_get_column(self):
        """Test column access."""
        values = [[1, 2], [3, 4], [5, 6]]
        event = EventMatrix(
            name='params',
            onsets=[0, 1, 2],
            values=values,
            column_names=['a', 'b']
        )
        
        # By index
        col0 = event.get_column(0)
        assert np.array_equal(col0, [1, 3, 5])
        
        # By name
        col_b = event.get_column('b')
        assert np.array_equal(col_b, [2, 4, 6])
    
    def test_design_matrix(self):
        """Test design matrix generation."""
        values = [[1, 2], [3, 4]]
        event = EventMatrix(
            name='params',
            onsets=[0, 2],
            values=values,
            durations=1
        )
        
        sampling_points = np.arange(0, 4, 1)
        X = event.design_matrix(sampling_points)
        
        assert X.shape == (4, 2)
        expected = [[1, 2], [0, 0], [3, 4], [0, 0]]
        assert np.array_equal(X, expected)
    
    def test_split_columns(self):
        """Test splitting into separate variables."""
        values = [[1, 2], [3, 4]]
        event = EventMatrix(
            name='params',
            onsets=[0, 1],
            values=values,
            column_names=['a', 'b']
        )
        
        split = event.split_columns()
        
        assert len(split) == 2
        assert 'a' in split and 'b' in split
        assert isinstance(split['a'], EventVariable)
        assert np.array_equal(split['a'].values, [1, 3])


class TestEventBasis:
    """Test EventBasis class."""
    
    def test_with_poly_basis(self):
        """Test event with polynomial basis."""
        from fmrimod.basis import Poly
        
        basis = Poly(degree=2)
        event = EventBasis(
            name='time',
            onsets=[0, 1, 2, 3],
            values=[0, 1, 2, 3],
            basis=basis
        )
        
        assert event.name == 'time'
        assert event.n_events == 4
        assert event.event_type == 'basis'
        assert event.n_basis == 3  # Intercept + linear + quadratic
    
    def test_expanded_values(self):
        """Test basis expansion."""
        from fmrimod.basis import Poly
        
        basis = Poly(degree=1)  # Linear
        event = EventBasis(
            name='x',
            onsets=[0, 1, 2],
            values=[1, 2, 3],
            basis=basis
        )
        
        # Should have intercept and linear terms
        assert event.expanded_values.shape == (3, 2)
    
    def test_to_matrix(self):
        """Test conversion to EventMatrix."""
        from fmrimod.basis import Poly
        
        basis = Poly(degree=2)
        event = EventBasis(
            name='param',
            onsets=[0, 1, 2],
            values=[0, 1, 2],
            basis=basis
        )
        
        matrix = event.to_matrix()
        
        assert isinstance(matrix, EventMatrix)
        assert matrix.n_columns == event.n_basis
        assert matrix.n_events == event.n_events


class TestEventTerm:
    """Test EventTerm for combinations."""
    
    def test_single_event_term(self):
        """Test term with single event."""
        event = EventFactor(
            name='condition',
            onsets=[0, 1],
            values=['A', 'B']
        )
        
        term = EventTerm([event])
        
        assert term.name == 'condition'
        assert term.n_events == 1
        assert not term.interaction
        assert term.is_categorical
    
    def test_categorical_interaction(self):
        """Test interaction between categorical events."""
        cond = EventFactor(
            name='condition',
            onsets=[0, 1, 2, 3],
            values=['A', 'B', 'A', 'B']
        )
        
        block = EventFactor(
            name='block',
            onsets=[0, 1, 2, 3],
            values=['1', '1', '2', '2']
        )
        
        term = create_interaction(cond, block)
        
        assert term.name == 'condition:block'
        assert term.interaction
        assert term.is_categorical
        
        # Should have 2x2 = 4 level combinations
        levels = term.get_levels()
        assert len(levels) == 4
        assert ('A', '1') in levels
        
        # Column names
        col_names = term.get_column_names()
        assert len(col_names) == 4
        assert 'A:1' in col_names
    
    def test_continuous_interaction(self):
        """Test interaction between continuous events."""
        rating = EventVariable(
            name='rating',
            onsets=[0, 1, 2],
            values=[1, 2, 3],
            center=False
        )
        
        rt = EventVariable(
            name='rt',
            onsets=[0, 1, 2],
            values=[0.5, 0.6, 0.7],
            center=False
        )
        
        term = create_interaction(rating, rt)
        
        assert term.is_continuous
        assert not term.is_categorical
        
        # Test design matrix
        sampling_points = np.arange(0, 3, 1)
        X = term.design_matrix(sampling_points)
        
        # Should multiply values
        expected = [[0.5], [1.2], [2.1]]  # 1*0.5, 2*0.6, 3*0.7
        assert np.allclose(X, expected)
    
    def test_mixed_interaction(self):
        """Test interaction between categorical and continuous."""
        cond = EventFactor(
            name='condition',
            onsets=[0, 1],
            values=['A', 'B']
        )
        
        rating = EventVariable(
            name='rating',
            onsets=[0, 1],
            values=[2, 4],
            center=False
        )
        
        term = create_interaction(cond, rating)
        
        assert term.is_mixed
        assert not term.is_categorical
        assert not term.is_continuous


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_create_event(self):
        """Test create_event factory."""
        # Factor
        event = create_event(
            'factor',
            name='condition',
            onsets=[0, 1],
            values=['A', 'B']
        )
        assert isinstance(event, EventFactor)
        
        # Variable
        event = create_event(
            'variable',
            name='rating',
            onsets=[0, 1],
            values=[1, 2]
        )
        assert isinstance(event, EventVariable)
    
    def test_events_from_dataframe(self):
        """Test creating multiple events from DataFrame."""
        df = pd.DataFrame({
            'onset': [0, 1, 2, 3],
            'condition': ['A', 'B', 'A', 'B'],
            'rating': [1, 2, 3, 4],
            'x': [0.1, 0.2, 0.3, 0.4],
            'y': [0.5, 0.6, 0.7, 0.8]
        })
        
        specs = {
            'condition': {'type': 'factor'},
            'rating': {'type': 'variable', 'center': True},
            'motion': {'type': 'matrix', 'value_cols': ['x', 'y']}
        }
        
        events = events_from_dataframe(df, specs)
        
        assert len(events) == 3
        assert isinstance(events['condition'], EventFactor)
        assert isinstance(events['rating'], EventVariable)
        assert isinstance(events['motion'], EventMatrix)
        assert events['motion'].n_columns == 2
