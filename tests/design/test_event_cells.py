"""Tests for event cells, conditions, matrix, and basis components."""

import pytest
import numpy as np
import pandas as pd

from fmrimod.events import (
    EventFactor,
    EventVariable,
    EventMatrix,
    EventBasis,
    EventTerm,
)
from fmrimod.events.cells import (
    cells_event_term,
    conditions_event_term,
    cells_event_model,
    conditions_event_model,
)
from fmrimod.events.term import create_interaction
from fmrimod.basis import Poly, BSpline
from fmrimod import event_model
from fmrimod.naming import level_token, continuous_token


# ============================================================================
# Test cells_event_term - P2-01
# ============================================================================

class TestCellsEventTerm:
    """Test cells_event_term function."""

    def test_single_categorical_factor(self):
        """Test cells extraction from single categorical factor."""
        event = EventFactor(
            name='condition',
            onsets=[0, 1, 2, 3, 4, 5],
            values=['A', 'B', 'C', 'A', 'B', 'C']
        )
        term = EventTerm([event])

        cells = cells_event_term(term, drop_empty=True)

        # Should have 3 rows (one per level)
        assert len(cells) == 3
        assert 'condition' in cells.columns
        assert set(cells['condition']) == {'A', 'B', 'C'}

        # Check counts
        counts = cells.attrs['count']
        assert len(counts) == 3
        assert all(counts == 2)  # Each level appears twice

    def test_two_categorical_factors_interaction(self):
        """Test cells from interaction of two categorical factors."""
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

        cells = cells_event_term(term, drop_empty=True)

        # Should have 4 rows (2x2 combinations)
        assert len(cells) == 4
        assert 'condition' in cells.columns
        assert 'block' in cells.columns

        # Check all combinations are present
        combo_set = set(zip(cells['condition'], cells['block']))
        expected = {('A', '1'), ('A', '2'), ('B', '1'), ('B', '2')}
        assert combo_set == expected

        # Check counts
        counts = cells.attrs['count']
        assert len(counts) == 4
        assert all(counts == 1)  # Each combination appears once

    def test_single_continuous_event(self):
        """Test cells from single continuous event (EventVariable)."""
        event = EventVariable(
            name='rating',
            onsets=[0, 1, 2, 3],
            values=[1.5, 2.5, 3.5, 4.5],
            center=False
        )
        term = EventTerm([event])

        cells = cells_event_term(term, drop_empty=True)

        # Should have 1 row with continuous variable name
        assert len(cells) == 1
        assert 'rating' in cells.columns
        assert cells['rating'].iloc[0] == ''  # Empty string for continuous

        # Count should be number of observations
        counts = cells.attrs['count']
        assert len(counts) == 1
        assert counts[0] == 4

    def test_mixed_term_categorical_continuous(self):
        """Test cells from mixed term (categorical + continuous)."""
        cond = EventFactor(
            name='condition',
            onsets=[0, 1, 2, 3],
            values=['A', 'B', 'A', 'B']
        )
        rating = EventVariable(
            name='rating',
            onsets=[0, 1, 2, 3],
            values=[1, 2, 3, 4],
            center=False
        )
        term = EventTerm([cond, rating])

        cells = cells_event_term(term, drop_empty=True)

        # Should have rows for each categorical level
        # Note: continuous events don't create separate columns in cells
        assert len(cells) == 2
        assert 'condition' in cells.columns
        assert set(cells['condition']) == {'A', 'B'}

        # Check counts
        counts = cells.attrs['count']
        assert len(counts) == 2
        assert all(counts == 2)  # Each level appears twice

    def test_drop_empty_false(self):
        """Test drop_empty=False includes zero-count cells."""
        event = EventFactor(
            name='condition',
            onsets=[0, 1],
            values=['A', 'A'],  # Only 'A' appears
            levels=['A', 'B', 'C']  # But three levels defined
        )
        term = EventTerm([event])

        # With drop_empty=False, should include all levels
        cells = cells_event_term(term, drop_empty=False)
        assert len(cells) == 3
        assert set(cells['condition']) == {'A', 'B', 'C'}

        counts = cells.attrs['count']
        assert counts[0] == 2  # A appears twice
        assert counts[1] == 0  # B appears zero times
        assert counts[2] == 0  # C appears zero times

        # With drop_empty=True, should only include observed levels
        cells_dropped = cells_event_term(term, drop_empty=True)
        assert len(cells_dropped) == 1
        assert cells_dropped['condition'].iloc[0] == 'A'

    def test_matrix_event(self):
        """Test cells from EventMatrix."""
        event = EventMatrix(
            name='motion',
            onsets=[0, 1, 2],
            values=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            column_names=['x', 'y', 'z']
        )
        term = EventTerm([event])

        cells = cells_event_term(term, drop_empty=True)

        # Should have 1 row with all column names
        assert len(cells) == 1
        assert 'x' in cells.columns
        assert 'y' in cells.columns
        assert 'z' in cells.columns
        assert all(cells.iloc[0] == '')  # All continuous

        counts = cells.attrs['count']
        assert counts[0] == 3

    def test_caching(self):
        """Test that cells are cached for performance."""
        event = EventFactor(
            name='condition',
            onsets=[0, 1, 2],
            values=['A', 'B', 'A']
        )
        term = EventTerm([event])

        # First call
        cells1 = cells_event_term(term, drop_empty=True)

        # Second call should use cache
        cells2 = cells_event_term(term, drop_empty=True)

        # Should be the same object (from cache)
        assert cells1 is cells2

        # Different parameter should not use cache
        cells3 = cells_event_term(term, drop_empty=False)
        assert cells3 is not cells1


# ============================================================================
# Test conditions_event_term - P2-01
# ============================================================================

class TestConditionsEventTerm:
    """Test conditions_event_term function."""

    def test_single_categorical_factor(self):
        """Test condition names from single categorical factor."""
        event = EventFactor(
            name='condition',
            onsets=[0, 1, 2],
            values=['A', 'B', 'C']
        )
        term = EventTerm([event])

        conditions = conditions_event_term(term)

        # Should have 3 condition names (one per level)
        assert len(conditions) == 3
        assert set(conditions) == {
            level_token('condition', 'A'),
            level_token('condition', 'B'),
            level_token('condition', 'C')
        }

    def test_two_categorical_factors_interaction(self):
        """Test condition names from categorical interaction."""
        cond = EventFactor(
            name='condition',
            onsets=[0, 1, 2, 3],
            values=['A', 'B', 'A', 'B']
        )
        block = EventFactor(
            name='block',
            onsets=[0, 1, 2, 3],
            values=['1', '2', '1', '2']
        )
        term = create_interaction(cond, block)

        conditions = conditions_event_term(term)

        # Should have 4 combinations
        assert len(conditions) == 4

        # Check format: should use level_token and combine with _
        expected_patterns = ['condition.A_block.1', 'condition.A_block.2',
                            'condition.B_block.1', 'condition.B_block.2']
        assert set(conditions) == set(expected_patterns)

    def test_single_continuous_variable(self):
        """Test condition names from continuous variable."""
        event = EventVariable(
            name='rating',
            onsets=[0, 1, 2],
            values=[1, 2, 3],
            center=False
        )
        term = EventTerm([event])

        conditions = conditions_event_term(term)

        # Should have 1 condition using continuous_token
        assert len(conditions) == 1
        assert conditions[0] == continuous_token('rating')

    def test_matrix_event_columns(self):
        """Test condition names from EventMatrix."""
        event = EventMatrix(
            name='motion',
            onsets=[0, 1],
            values=[[1, 2], [3, 4]],
            column_names=['x', 'y']
        )
        term = EventTerm([event])

        conditions = conditions_event_term(term)

        # Should have one condition per column
        assert len(conditions) == 2
        assert set(conditions) == {
            continuous_token('x'),
            continuous_token('y')
        }

    def test_basis_event_no_expand(self):
        """Test condition names from EventBasis without expansion."""
        basis = Poly(degree=2)
        event = EventBasis(
            name='time',
            onsets=[0, 1, 2],
            values=[0, 1, 2],
            basis=basis
        )
        term = EventTerm([event])

        conditions = conditions_event_term(term, expand_basis=False)

        # Should have 1 condition (no basis expansion)
        assert len(conditions) == 1
        assert conditions[0] == continuous_token('time')

    def test_basis_event_with_expand(self):
        """Test condition names from EventBasis with expansion."""
        basis = Poly(degree=2)
        event = EventBasis(
            name='time',
            onsets=[0, 1, 2],
            values=[0, 1, 2],
            basis=basis
        )
        term = EventTerm([event])

        # Manually set nbasis on term for expansion
        term.nbasis = 3

        conditions = conditions_event_term(term, expand_basis=True)

        # Should have 3 conditions (basis expansion)
        assert len(conditions) == 3
        assert conditions[0] == continuous_token('time') + '_b1'
        assert conditions[1] == continuous_token('time') + '_b2'
        assert conditions[2] == continuous_token('time') + '_b3'

    def test_mixed_categorical_continuous(self):
        """Test condition names from mixed term."""
        cond = EventFactor(
            name='condition',
            onsets=[0, 1, 2, 3],
            values=['A', 'B', 'A', 'B']
        )
        rating = EventVariable(
            name='rating',
            onsets=[0, 1, 2, 3],
            values=[1, 2, 3, 4],
            center=False
        )
        term = EventTerm([cond, rating])

        conditions = conditions_event_term(term)

        # Should combine categorical levels with continuous name
        assert len(conditions) == 2
        expected = {
            level_token('condition', 'A') + '_' + continuous_token('rating'),
            level_token('condition', 'B') + '_' + continuous_token('rating')
        }
        assert set(conditions) == expected

    def test_caching(self):
        """Test that conditions are cached."""
        event = EventFactor(
            name='condition',
            onsets=[0, 1],
            values=['A', 'B']
        )
        term = EventTerm([event])

        # First call
        cond1 = conditions_event_term(term)

        # Second call should use cache
        cond2 = conditions_event_term(term)

        assert cond1 is cond2

        # Different parameters should not use cache
        cond3 = conditions_event_term(term, expand_basis=True)
        assert cond3 is not cond1


# ============================================================================
# Test cells_event_model and conditions_event_model - P2-01
# ============================================================================

class TestEventModelCells:
    """Test cells and conditions extraction from EventModel."""

    def test_cells_event_model(self):
        """Test cells_event_model extracts cells from all terms."""
        # Create event data
        df = pd.DataFrame({
            'onset': [0, 1, 2, 3, 4, 5],
            'condition': ['A', 'B', 'A', 'B', 'A', 'B'],
            'block': ['1', '1', '2', '2', '3', '3'],
            'rating': [1, 2, 3, 4, 5, 6]
        })

        # Create model with multiple terms
        model = event_model('condition + block', data=df, tr=2.0, n_scans=10)

        cells_list = cells_event_model(model, drop_empty=True)

        # Should have one DataFrame per term
        assert len(cells_list) >= 2

        # Each should be a DataFrame
        assert all(isinstance(c, pd.DataFrame) for c in cells_list)

    def test_conditions_event_model(self):
        """Test conditions_event_model extracts conditions from all terms."""
        df = pd.DataFrame({
            'onset': [0, 1, 2, 3],
            'condition': ['A', 'B', 'A', 'B'],
            'rating': [1, 2, 3, 4]
        })

        model = event_model('condition + rating', data=df, tr=2.0, n_scans=10)

        conditions_list = conditions_event_model(model, drop_empty=True)

        # Should have one list per term
        assert len(conditions_list) >= 2

        # Each should be a list of strings
        assert all(isinstance(c, list) for c in conditions_list)
        assert all(all(isinstance(s, str) for s in c) for c in conditions_list)


# ============================================================================
# Test EventMatrix - P2-02
# ============================================================================

class TestEventMatrixConstruction:
    """Test EventMatrix construction and properties."""

    def test_construction_with_array(self):
        """Test EventMatrix construction with array values."""
        values = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])

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
        assert np.array_equal(event.values, values)

    def test_construction_with_dataframe(self):
        """Test EventMatrix construction with DataFrame values."""
        df_values = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6],
            'z': [7, 8, 9]
        })

        event = EventMatrix(
            name='motion',
            onsets=[0, 1, 2],
            values=df_values
        )

        assert event.n_columns == 3
        # Column names should be auto-detected from DataFrame
        assert event.column_names == ['x', 'y', 'z']
        assert np.array_equal(event.values, df_values.values)

    def test_auto_generated_column_names(self):
        """Test auto-generated column names when not provided."""
        event = EventMatrix(
            name='params',
            onsets=[0, 1],
            values=[[1, 2], [3, 4]]
        )

        # Should auto-generate names
        assert event.column_names == ['params_1', 'params_2']

    def test_explicit_column_names(self):
        """Test explicit column names."""
        event = EventMatrix(
            name='motion',
            onsets=[0, 1],
            values=[[1, 2], [3, 4]],
            column_names=['pitch', 'roll']
        )

        assert event.column_names == ['pitch', 'roll']

    def test_n_columns_property(self):
        """Test n_columns property."""
        event = EventMatrix(
            name='params',
            onsets=[0, 1, 2],
            values=np.random.randn(3, 5)
        )

        assert event.n_columns == 5

    def test_event_type(self):
        """Test event_type is 'matrix'."""
        event = EventMatrix(
            name='test',
            onsets=[0],
            values=[[1, 2]]
        )

        assert event.event_type == 'matrix'

    def test_design_matrix_impulse(self):
        """Test design_matrix method with impulse events."""
        event = EventMatrix(
            name='params',
            onsets=[0, 2, 4],
            values=[[1, 2], [3, 4], [5, 6]],
            durations=0  # Impulse events
        )

        sampling_points = np.arange(0, 6, 1)
        X = event.design_matrix(sampling_points)

        assert X.shape == (6, 2)  # 6 timepoints, 2 columns

        # Check values at onset times
        assert np.array_equal(X[0, :], [1, 2])
        assert np.array_equal(X[2, :], [3, 4])
        assert np.array_equal(X[4, :], [5, 6])

        # Check zeros elsewhere
        assert np.array_equal(X[1, :], [0, 0])
        assert np.array_equal(X[3, :], [0, 0])
        assert np.array_equal(X[5, :], [0, 0])

    def test_design_matrix_extended(self):
        """Test design_matrix method with extended events."""
        event = EventMatrix(
            name='params',
            onsets=[0, 3],
            values=[[1, 2], [3, 4]],
            durations=2
        )

        sampling_points = np.arange(0, 6, 1)
        X = event.design_matrix(sampling_points)

        assert X.shape == (6, 2)

        # First event (onset=0, duration=2): fills indices 0, 1
        assert np.array_equal(X[0, :], [1, 2])
        assert np.array_equal(X[1, :], [1, 2])

        # Second event (onset=3, duration=2): fills indices 3, 4
        assert np.array_equal(X[3, :], [3, 4])
        assert np.array_equal(X[4, :], [3, 4])

        # Other indices should be zero
        assert np.array_equal(X[2, :], [0, 0])
        assert np.array_equal(X[5, :], [0, 0])

    def test_interaction_with_event_factor(self):
        """Test EventMatrix in interaction with EventFactor."""
        matrix = EventMatrix(
            name='params',
            onsets=[0, 1, 2, 3],
            values=[[1, 2], [3, 4], [5, 6], [7, 8]],
            column_names=['a', 'b']
        )

        factor = EventFactor(
            name='condition',
            onsets=[0, 1, 2, 3],
            values=['A', 'B', 'A', 'B']
        )

        # Create interaction term
        term = EventTerm([factor, matrix])

        # Should be a mixed term
        assert term.is_mixed
        # Both events should be in the term
        assert len(term.events) == 2
        event_names = [e.name for e in term.events]
        assert 'condition' in event_names
        assert 'params' in event_names

    def test_1d_array_conversion(self):
        """Test that 1D arrays are converted to 2D."""
        event = EventMatrix(
            name='param',
            onsets=[0, 1],
            values=[1, 2]  # 1D array
        )

        assert event.values.ndim == 2
        assert event.values.shape == (2, 1)
        assert event.n_columns == 1

    def test_validation_shape_mismatch(self):
        """Test validation catches shape mismatch."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            EventMatrix(
                name='test',
                onsets=[0, 1, 2],  # 3 onsets
                values=[[1, 2], [3, 4]]  # 2 rows
            )

    def test_validation_non_finite(self):
        """Test validation catches non-finite values."""
        with pytest.raises(ValueError, match="finite"):
            EventMatrix(
                name='test',
                onsets=[0, 1],
                values=[[1, np.nan], [3, 4]]
            )

    def test_validation_column_names_mismatch(self):
        """Test validation catches column name count mismatch."""
        with pytest.raises(ValueError, match="Number of column names"):
            EventMatrix(
                name='test',
                onsets=[0, 1],
                values=[[1, 2], [3, 4]],
                column_names=['a', 'b', 'c']  # 3 names for 2 columns
            )


# ============================================================================
# Test EventBasis - P2-03
# ============================================================================

class TestEventBasisConstruction:
    """Test EventBasis construction and properties."""

    def test_construction_with_poly(self):
        """Test EventBasis construction with Poly basis."""
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

    def test_construction_with_bspline(self):
        """Test EventBasis construction with BSpline basis."""
        basis = BSpline(df=4)
        event = EventBasis(
            name='param',
            onsets=[0, 1, 2, 3, 4],
            values=np.linspace(0, 1, 5),
            basis=basis
        )

        assert event.name == 'param'
        assert event.n_events == 5
        assert event.event_type == 'basis'
        assert event.n_basis == 4

    def test_expanded_values_property(self):
        """Test expanded_values / basis_matrix property."""
        basis = Poly(degree=1)
        event = EventBasis(
            name='x',
            onsets=[0, 1, 2],
            values=[1, 2, 3],
            basis=basis
        )

        # expanded_values should be a 2D array
        assert event.expanded_values.ndim == 2
        assert event.expanded_values.shape[0] == 3  # n_events
        assert event.expanded_values.shape[1] == 2  # Intercept + linear

    def test_design_matrix_method(self):
        """Test design_matrix method."""
        basis = Poly(degree=1)
        event = EventBasis(
            name='x',
            onsets=[0, 2],
            values=[1, 2],
            basis=basis,
            durations=1
        )

        sampling_points = np.arange(0, 4, 1)
        X = event.design_matrix(sampling_points)

        # Should have shape (n_timepoints, n_basis)
        assert X.shape == (4, 2)

        # Check that values are filled at onset times
        # First event at onset=0, duration=1: fills index 0
        assert X[0, 0] != 0  # Should have intercept
        assert X[0, 1] != 0  # Should have linear term

        # Second event at onset=2, duration=1: fills index 2
        assert X[2, 0] != 0
        assert X[2, 1] != 0

    def test_n_basis_property(self):
        """Test n_basis property."""
        basis = Poly(degree=3)
        event = EventBasis(
            name='x',
            onsets=[0, 1, 2],
            values=[0, 1, 2],
            basis=basis
        )

        # Poly(degree=3) should give 3 basis functions (linear, quadratic, cubic - no intercept)
        assert event.n_basis == 3

    def test_basis_names_property(self):
        """Test basis_names property."""
        basis = Poly(degree=2)
        event = EventBasis(
            name='time',
            onsets=[0, 1, 2],
            values=[0, 1, 2],
            basis=basis
        )

        names = event.basis_names
        assert len(names) == 3  # Should match n_basis
        assert all(isinstance(n, str) for n in names)

    def test_event_type(self):
        """Test event_type is 'basis'."""
        basis = Poly(degree=1)
        event = EventBasis(
            name='x',
            onsets=[0],
            values=[1],
            basis=basis
        )

        assert event.event_type == 'basis'

    def test_validation_1d_values(self):
        """Test validation ensures 1D values."""
        basis = Poly(degree=1)

        # 2D values should raise error
        with pytest.raises(ValueError, match="1-dimensional"):
            EventBasis(
                name='x',
                onsets=[0, 1],
                values=[[1, 2], [3, 4]],  # 2D
                basis=basis
            )

    def test_validation_length_mismatch(self):
        """Test validation catches length mismatch."""
        basis = Poly(degree=1)

        with pytest.raises(ValueError, match="Length mismatch"):
            EventBasis(
                name='x',
                onsets=[0, 1, 2],  # 3 onsets
                values=[1, 2],  # 2 values
                basis=basis
            )

    def test_validation_non_finite(self):
        """Test validation catches non-finite values."""
        basis = Poly(degree=1)

        with pytest.raises(ValueError, match="finite"):
            EventBasis(
                name='x',
                onsets=[0, 1],
                values=[1, np.inf],
                basis=basis
            )

    def test_validation_basis_protocol(self):
        """Test validation checks for evaluate method."""
        # Object without evaluate method
        class FakeBasis:
            pass

        with pytest.raises(TypeError, match="evaluate"):
            EventBasis(
                name='x',
                onsets=[0],
                values=[1],
                basis=FakeBasis()
            )

    def test_bspline_basic(self):
        """Test BSpline with basic configuration."""
        # Use sufficient data points for stable spline fitting
        # df=4 is a safe default for BSpline
        n_points = 20
        basis = BSpline(df=4)
        event = EventBasis(
            name='param',
            onsets=np.arange(n_points),
            values=np.linspace(0, 1, n_points),
            basis=basis
        )

        assert event.n_basis == 4
        assert event.expanded_values.shape == (n_points, 4)

    def test_design_matrix_impulse_events(self):
        """Test design matrix for impulse events (duration=0)."""
        basis = Poly(degree=1)
        event = EventBasis(
            name='x',
            onsets=[0, 2, 4],
            values=[1, 2, 3],
            basis=basis,
            durations=0  # Impulse
        )

        sampling_points = np.arange(0, 6, 1)
        X = event.design_matrix(sampling_points)

        assert X.shape == (6, 2)

        # Check that only onset times are filled
        assert X[0, :].sum() != 0  # Onset at 0
        assert X[1, :].sum() == 0  # No event
        assert X[2, :].sum() != 0  # Onset at 2
        assert X[3, :].sum() == 0  # No event
        assert X[4, :].sum() != 0  # Onset at 4
        assert X[5, :].sum() == 0  # No event


# ============================================================================
# Integration tests
# ============================================================================

class TestEventSystemIntegration:
    """Integration tests across multiple components."""

    def test_full_workflow_categorical(self):
        """Test full workflow with categorical events."""
        # Create events
        event = EventFactor(
            name='condition',
            onsets=[0, 1, 2, 3, 4, 5],
            values=['A', 'B', 'C', 'A', 'B', 'C']
        )

        # Create term
        term = EventTerm([event])

        # Extract cells
        cells = cells_event_term(term)
        assert len(cells) == 3

        # Extract conditions
        conditions = conditions_event_term(term)
        assert len(conditions) == 3
        assert all('condition.' in c for c in conditions)

    def test_full_workflow_interaction(self):
        """Test full workflow with interaction."""
        cond = EventFactor(
            name='condition',
            onsets=[0, 1, 2, 3],
            values=['A', 'B', 'A', 'B']
        )
        block = EventFactor(
            name='block',
            onsets=[0, 1, 2, 3],
            values=['1', '2', '1', '2']
        )

        term = create_interaction(cond, block)

        # Extract cells
        cells = cells_event_term(term)
        # Only 2 cells because each combination appears once: (A,1), (B,2)
        assert len(cells) == 2

        # Extract conditions
        conditions = conditions_event_term(term)
        # Should have 4 possible combinations even if not all observed
        assert len(conditions) >= 2

    def test_full_workflow_mixed(self):
        """Test full workflow with mixed term."""
        cond = EventFactor(
            name='condition',
            onsets=[0, 1, 2, 3],
            values=['A', 'B', 'A', 'B']
        )
        rating = EventVariable(
            name='rating',
            onsets=[0, 1, 2, 3],
            values=[1, 2, 3, 4],
            center=False
        )

        term = EventTerm([cond, rating])

        cells = cells_event_term(term)
        # Categorical events create columns, continuous events don't
        assert 'condition' in cells.columns
        assert len(cells) == 2  # Two levels: A and B

        conditions = conditions_event_term(term)
        assert len(conditions) == 2  # One per level of condition

    def test_full_workflow_matrix(self):
        """Test full workflow with matrix event."""
        event = EventMatrix(
            name='motion',
            onsets=[0, 1, 2],
            values=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            column_names=['x', 'y', 'z']
        )

        term = EventTerm([event])

        cells = cells_event_term(term)
        assert len(cells) == 1
        assert all(col in cells.columns for col in ['x', 'y', 'z'])

        conditions = conditions_event_term(term)
        assert len(conditions) == 3  # One per column

        # Test design matrix
        sampling_points = np.arange(0, 4, 1)
        X = event.design_matrix(sampling_points)
        assert X.shape == (4, 3)

    def test_full_workflow_basis(self):
        """Test full workflow with basis event."""
        basis = Poly(degree=2)
        event = EventBasis(
            name='time',
            onsets=[0, 1, 2, 3],
            values=[0, 1, 2, 3],
            basis=basis
        )

        term = EventTerm([event])

        cells = cells_event_term(term)
        assert len(cells) == 1

        conditions_no_expand = conditions_event_term(term, expand_basis=False)
        assert len(conditions_no_expand) == 1

        # Test with expansion
        term.nbasis = 3
        conditions_expand = conditions_event_term(term, expand_basis=True)
        assert len(conditions_expand) == 3

        # Test design matrix
        sampling_points = np.arange(0, 5, 1)
        X = event.design_matrix(sampling_points)
        assert X.shape == (5, 3)
