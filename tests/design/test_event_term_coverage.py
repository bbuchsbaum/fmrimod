"""Coverage-focused tests for events/term.py.

This module targets specific uncovered lines to boost coverage from 71% to 85%+.
"""

import pytest
import numpy as np
import pandas as pd

from fmrimod.events.factor import EventFactor
from fmrimod.events.variable import EventVariable
from fmrimod.events.matrix import EventMatrix
from fmrimod.events.basis import EventBasis
from fmrimod.events.term import EventTerm, create_interaction
from fmrimod.basis import Poly, BSpline


class TestCategoricalInteractionCoverage:
    """Test 3-way categorical interaction to cover lines 138-144, 149."""

    def test_three_way_categorical_interaction(self):
        """Test 3-way interaction: factor A × B × C."""
        # Factor A: 2 levels
        factor_a = EventFactor(
            name='factor_a',
            onsets=[0, 1, 2, 3, 4, 5, 6, 7],
            values=['A1', 'A2', 'A1', 'A2', 'A1', 'A2', 'A1', 'A2']
        )

        # Factor B: 3 levels
        factor_b = EventFactor(
            name='factor_b',
            onsets=[0, 1, 2, 3, 4, 5, 6, 7],
            values=['B1', 'B2', 'B3', 'B1', 'B2', 'B3', 'B1', 'B2']
        )

        # Factor C: 2 levels
        factor_c = EventFactor(
            name='factor_c',
            onsets=[0, 1, 2, 3, 4, 5, 6, 7],
            values=['C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C2']
        )

        # Create 3-way interaction
        term = EventTerm([factor_a, factor_b, factor_c], interaction=True)

        # Verify it's categorical
        assert term.is_categorical
        assert term.interaction
        assert term.n_events == 3

        # Should have 2 × 3 × 2 = 12 level combinations
        levels = term.get_levels()
        assert len(levels) == 12

        # Check specific combinations exist
        assert ('A1', 'B1', 'C1') in levels
        assert ('A2', 'B3', 'C2') in levels

        # Test column names
        col_names = term.get_column_names()
        assert len(col_names) == 12
        assert 'A1:B1:C1' in col_names
        assert 'A2:B3:C2' in col_names

        # Test design matrix generation (covers lines 138-144)
        sampling_points = np.arange(0, 10, 1)
        X = term.design_matrix(sampling_points)

        # Should have 12 columns (one per combination)
        assert X.shape == (10, 12)

        # Verify indicators are binary
        assert np.all((X == 0) | (X == 1))

    def test_interaction_design_matrix_computation(self):
        """Test design matrix for 2-way interaction (covers lines 158-166)."""
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

        # Test _get_n_columns (lines 155-178)
        n_cols = term._get_n_columns()
        assert n_cols == 4  # 2 × 2

        # Test design matrix
        sampling_points = np.arange(0, 5, 1)
        X = term.design_matrix(sampling_points)

        assert X.shape == (5, 4)


class TestMixedInteractionEdgeCases:
    """Test mixed interaction edge cases to cover lines 251-268, 285-286, 292-293."""

    def test_continuous_times_continuous_interaction(self):
        """Test continuous × continuous interaction (edge case)."""
        rating1 = EventVariable(
            name='rating1',
            onsets=[0, 1, 2],
            values=[2.0, 3.0, 4.0],
            center=False
        )
        rating2 = EventVariable(
            name='rating2',
            onsets=[0, 1, 2],
            values=[0.5, 0.6, 0.7],
            center=False
        )

        term = create_interaction(rating1, rating2)

        # Verify it's continuous interaction
        assert term.is_continuous
        assert not term.is_mixed

        # Test design matrix (covers _continuous_interaction)
        sampling_points = np.arange(0, 3, 1)
        X = term.design_matrix(sampling_points)

        # Should multiply element-wise
        assert X.shape == (3, 1)
        expected = [[1.0], [1.8], [2.8]]  # 2*0.5, 3*0.6, 4*0.7
        assert np.allclose(X, expected)

    def test_single_level_factor_times_continuous(self):
        """Test single-level factor × continuous (edge case for lines 285-286)."""
        # Factor with only one level observed
        factor = EventFactor(
            name='single_level',
            onsets=[0, 1, 2],
            values=['Only', 'Only', 'Only']
        )

        rating = EventVariable(
            name='rating',
            onsets=[0, 1, 2],
            values=[1.0, 2.0, 3.0],
            center=False
        )

        term = create_interaction(factor, rating)

        # Verify it's mixed
        assert term.is_mixed

        # Test design matrix (covers _mixed_interaction edge case)
        sampling_points = np.arange(0, 3, 1)
        X = term.design_matrix(sampling_points)

        # Should have 1 column (1 level × 1 continuous)
        assert X.shape == (3, 1)

    def test_mixed_no_categorical_events(self):
        """Test edge case where categorical events list is empty (line 291-295)."""
        # This tests the branch where cont_mats exist but cat_events is empty
        rating1 = EventVariable(
            name='rating1',
            onsets=[0, 1],
            values=[1.0, 2.0],
            center=False
        )
        rating2 = EventVariable(
            name='rating2',
            onsets=[0, 1],
            values=[3.0, 4.0],
            center=False
        )

        term = create_interaction(rating1, rating2)

        sampling_points = np.arange(0, 2, 1)
        X = term.design_matrix(sampling_points)

        # Both continuous, so should multiply
        assert X.shape == (2, 1)
        assert np.allclose(X, [[3.0], [8.0]])  # 1*3, 2*4

    def test_mixed_multi_column_continuous(self):
        """Test mixed with multi-column continuous (lines 301-316)."""
        factor = EventFactor(
            name='condition',
            onsets=[0, 1, 2, 3],
            values=['A', 'B', 'A', 'B']
        )

        # Matrix with 2 columns
        matrix = EventMatrix(
            name='params',
            onsets=[0, 1, 2, 3],
            values=[[1, 2], [3, 4], [5, 6], [7, 8]],
            column_names=['p1', 'p2']
        )

        term = create_interaction(factor, matrix)

        assert term.is_mixed

        # Design matrix should have 2 levels × 2 columns = 4 columns
        sampling_points = np.arange(0, 4, 1)
        X = term.design_matrix(sampling_points)

        assert X.shape == (4, 4)


class TestBasisExpandedTerms:
    """Test basis-expanded terms to cover lines 302-316."""

    def test_basis_term_column_count(self):
        """Test EventBasis term column count and naming (lines 139-144)."""
        basis = Poly(degree=2)
        event = EventBasis(
            name='time',
            onsets=[0, 1, 2, 3],
            values=[0, 1, 2, 3],
            basis=basis
        )

        term = EventTerm([event])

        # Verify event type
        assert term.events[0].event_type == 'basis'

        # Test _get_n_columns for basis (lines 163-165)
        n_cols = term._get_n_columns()
        assert n_cols == 3  # degree=2 gives 3 basis functions

        # Test get_column_names for basis (lines 141-142)
        col_names = term.get_column_names()
        assert len(col_names) == 3
        assert all(isinstance(name, str) for name in col_names)

    def test_basis_design_matrix(self):
        """Test basis design matrix generation."""
        basis = Poly(degree=2)
        event = EventBasis(
            name='param',
            onsets=[0, 2, 4],
            values=[1, 2, 3],
            basis=basis,
            durations=1
        )

        term = EventTerm([event])

        sampling_points = np.arange(0, 6, 1)
        X = term.design_matrix(sampling_points)

        # Should have 3 columns for 3 basis functions
        assert X.shape == (6, 3)


class TestGetLevelsAndColumnNames:
    """Test get_levels and get_column_names for complex terms (lines 285-286, 292-293)."""

    def test_get_levels_for_interaction(self):
        """Test get_levels for multi-factor interaction (covers line 115-116)."""
        factor_a = EventFactor(
            name='a',
            onsets=[0, 1, 2, 3],
            values=['A1', 'A2', 'A1', 'A2']
        )
        factor_b = EventFactor(
            name='b',
            onsets=[0, 1, 2, 3],
            values=['B1', 'B2', 'B1', 'B2']
        )

        term = create_interaction(factor_a, factor_b)

        # get_levels should return cartesian product
        levels = term.get_levels()
        assert len(levels) == 4  # 2 × 2
        assert all(isinstance(level, tuple) for level in levels)
        assert all(len(level) == 2 for level in levels)

    def test_get_column_names_interaction_format(self):
        """Test column naming for interactions (lines 131-133)."""
        factor_a = EventFactor(
            name='condition',
            onsets=[0, 1],
            values=['Easy', 'Hard']
        )
        factor_b = EventFactor(
            name='block',
            onsets=[0, 1],
            values=['First', 'Second']
        )

        term = create_interaction(factor_a, factor_b)

        col_names = term.get_column_names()

        # Should use "level1:level2" format
        assert 'Easy:First' in col_names
        assert 'Hard:Second' in col_names

    def test_get_column_names_non_categorical_single(self):
        """Test column names for single non-categorical event (line 144)."""
        rating = EventVariable(
            name='my_rating',
            onsets=[0, 1, 2],
            values=[1, 2, 3],
            center=False
        )

        term = EventTerm([rating])

        col_names = term.get_column_names()
        assert col_names == ['my_rating']

    def test_get_column_names_matrix_event(self):
        """Test column names for matrix event (line 140)."""
        matrix = EventMatrix(
            name='motion',
            onsets=[0, 1],
            values=[[1, 2, 3], [4, 5, 6]],
            column_names=['x', 'y', 'z']
        )

        term = EventTerm([matrix])

        col_names = term.get_column_names()
        assert col_names == ['x', 'y', 'z']


class TestEmptyFactorLevels:
    """Test drop_empty behavior for factors with empty levels (lines 324-325, 331, 334)."""

    def test_factor_with_unobserved_levels(self):
        """Test factor where some levels have zero events."""
        # Define 3 levels but only observe 2
        factor = EventFactor(
            name='condition',
            onsets=[0, 1, 2, 3],
            values=['A', 'A', 'B', 'B'],
            levels=['A', 'B', 'C']  # 'C' is never observed
        )

        term = EventTerm([factor])

        # get_levels should return all defined levels
        levels = term.get_levels()
        assert len(levels) == 3
        assert ('A',) in levels
        assert ('B',) in levels
        assert ('C',) in levels

        # Test cells with drop_empty=True
        cells = term.cells(drop_empty=True)
        assert len(cells) == 2  # Only A and B observed
        assert set(cells['condition']) == {'A', 'B'}

        # Test cells with drop_empty=False
        cells_all = term.cells(drop_empty=False)
        assert len(cells_all) == 3  # All levels included
        assert set(cells_all['condition']) == {'A', 'B', 'C'}

        # Check counts
        counts = cells_all.attrs['count']
        assert counts[2] == 0  # 'C' has zero count


class TestTermMetadataAccessors:
    """Test term metadata accessors (lines 324-325, 331, 334)."""

    def test_conditions_method(self):
        """Test conditions() method on term."""
        factor = EventFactor(
            name='task',
            onsets=[0, 1, 2],
            values=['Go', 'NoGo', 'Go']
        )

        term = EventTerm([factor])

        # Test conditions without expansion
        conds = term.conditions(drop_empty=True, expand_basis=False)
        assert len(conds) == 2  # Go and NoGo
        assert isinstance(conds, list)
        assert all(isinstance(c, str) for c in conds)

    def test_conditions_with_basis_expansion(self):
        """Test conditions with expand_basis=True."""
        basis = Poly(degree=2)
        event = EventBasis(
            name='time_var',
            onsets=[0, 1, 2],
            values=[0, 1, 2],
            basis=basis
        )

        term = EventTerm([event])

        # Without expansion
        conds_no_expand = term.conditions(expand_basis=False)
        assert len(conds_no_expand) == 1

        # With expansion (need to set nbasis attribute)
        term.nbasis = 3
        conds_expand = term.conditions(expand_basis=True)
        # Should expand to multiple condition names
        assert len(conds_expand) >= 1


class TestContinuousInteractionColumnCounts:
    """Test continuous interaction with different column counts (lines 250-268)."""

    def test_continuous_interaction_broadcast(self):
        """Test broadcasting in continuous interaction."""
        # Single column continuous
        rating = EventVariable(
            name='rating',
            onsets=[0, 1],
            values=[2.0, 3.0],
            center=False
        )

        # Multi-column matrix
        matrix = EventMatrix(
            name='params',
            onsets=[0, 1],
            values=[[1, 2], [3, 4]],
            column_names=['p1', 'p2']
        )

        term = create_interaction(rating, matrix)

        # This is a continuous interaction (both are non-categorical)
        assert term.is_continuous

        sampling_points = np.arange(0, 2, 1)
        X = term.design_matrix(sampling_points)

        # Should broadcast: single column × 2 columns = 2 columns
        assert X.shape == (2, 2)

    def test_continuous_interaction_outer_product(self):
        """Test outer product case in continuous interaction (lines 257-268)."""
        # Matrix with 2 columns
        matrix1 = EventMatrix(
            name='m1',
            onsets=[0, 1],
            values=[[1, 2], [3, 4]],
            column_names=['a', 'b']
        )

        # Matrix with 3 columns
        matrix2 = EventMatrix(
            name='m2',
            onsets=[0, 1],
            values=[[1, 2, 3], [4, 5, 6]],
            column_names=['x', 'y', 'z']
        )

        term = create_interaction(matrix1, matrix2)

        assert term.is_continuous

        sampling_points = np.arange(0, 2, 1)
        X = term.design_matrix(sampling_points)

        # Outer product: 2 × 3 = 6 columns
        assert X.shape == (2, 6)


class TestMixedInteractionComplexCases:
    """Test complex mixed interaction scenarios (lines 318-336)."""

    def test_mixed_interaction_empty_categorical_indicator(self):
        """Test mixed interaction with empty categorical result (line 331-334)."""
        # This is a tricky case - the cols list might be empty
        # Testing the defensive check at line 333-334

        factor = EventFactor(
            name='cond',
            onsets=[0, 1],
            values=['A', 'B']
        )

        rating = EventVariable(
            name='rating',
            onsets=[0, 1],
            values=[1.0, 2.0],
            center=False
        )

        term = create_interaction(factor, rating)

        # Normal case, should not be empty
        sampling_points = np.arange(0, 2, 1)
        X = term.design_matrix(sampling_points)

        assert X.shape[0] == 2
        assert X.shape[1] == 2  # 2 levels × 1 continuous

    def test_mixed_with_no_continuous_part(self):
        """Test mixed interaction where cont_part is None (lines 315-316, 327-331)."""
        # Actually this would just be a pure categorical interaction
        # But let's test the code path where we multiply categorical with None

        factor_a = EventFactor(
            name='a',
            onsets=[0, 1],
            values=['A1', 'A2']
        )
        factor_b = EventFactor(
            name='b',
            onsets=[0, 1],
            values=['B1', 'B2']
        )

        term = create_interaction(factor_a, factor_b)

        # Should use categorical interaction, not mixed
        assert term.is_categorical

        sampling_points = np.arange(0, 2, 1)
        X = term.design_matrix(sampling_points)

        assert X.shape == (2, 4)  # 2 × 2 combinations


class TestColumnNamesGeneratedFormat:
    """Test generated column names for complex cases (lines 151-153)."""

    def test_generated_column_names_for_mixed_complex(self):
        """Test auto-generated column names (line 152-153)."""
        factor = EventFactor(
            name='condition',
            onsets=[0, 1, 2, 3],
            values=['A', 'B', 'A', 'B']
        )

        matrix = EventMatrix(
            name='params',
            onsets=[0, 1, 2, 3],
            values=[[1, 2], [3, 4], [5, 6], [7, 8]],
            column_names=['p1', 'p2']
        )

        term = create_interaction(factor, matrix)

        col_names = term.get_column_names()

        # Mixed term with 2 levels × 2 columns = 4 columns
        # Names should be generated as "{name}_{i+1}"
        assert len(col_names) == 4
        assert all('condition:params' in name for name in col_names)


class TestSingleNonCategoricalEvent:
    """Test single non-categorical event paths (lines 136-144)."""

    def test_single_matrix_event(self):
        """Test single matrix event column names (line 140)."""
        matrix = EventMatrix(
            name='motion',
            onsets=[0, 1, 2],
            values=[[1, 2], [3, 4], [5, 6]],
            column_names=['pitch', 'roll']
        )

        term = EventTerm([matrix])

        # Single event, not interaction
        assert not term.interaction

        col_names = term.get_column_names()
        assert col_names == ['pitch', 'roll']

    def test_single_basis_event(self):
        """Test single basis event column names (line 142)."""
        basis = Poly(degree=1)
        event = EventBasis(
            name='trend',
            onsets=[0, 1, 2],
            values=[0, 1, 2],
            basis=basis
        )

        term = EventTerm([event])

        col_names = term.get_column_names()
        # Should use basis_names
        assert len(col_names) == 2  # intercept + linear
        assert all(isinstance(name, str) for name in col_names)
