"""Comprehensive tests for F-contrast and basis filtering systems.

This test file covers:
- contrast/fcontrast.py: F-contrast matrix generation (currently 13% coverage)
- contrast/basis_filter.py: Basis function filtering (currently 6% coverage)
"""

import pytest
import numpy as np
import pandas as pd

from fmrimod.events.factor import EventFactor
from fmrimod.events.variable import EventVariable
from fmrimod.sampling import SamplingFrame
from fmrimod.contrast import Fcontrasts
from fmrimod.contrast.fcontrast import (
    _contrast_sum_matrix,
    _unit_vector,
)
from fmrimod.contrast.basis_filter import (
    filter_basis,
    apply_basis_filter,
)
from fmrimod.events.term import EventTerm, create_interaction
from fmrimod.design.event_model import EventModel
from fmrimod.formula.base import Term


class TestContrastSumMatrix:
    """Test _contrast_sum_matrix helper function."""

    def test_two_levels(self):
        """Test contrast matrix for 2 levels."""
        C = _contrast_sum_matrix(2)

        # Should be 2x1
        assert C.shape == (2, 1)

        # Should be [1, -1]
        assert C[0, 0] == 1
        assert C[1, 0] == -1

        # Sum to zero
        assert np.sum(C[:, 0]) == 0

    def test_three_levels(self):
        """Test contrast matrix for 3 levels."""
        C = _contrast_sum_matrix(3)

        # Should be 3x2
        assert C.shape == (3, 2)

        # First two rows should be identity-like
        assert C[0, 0] == 1 and C[0, 1] == 0
        assert C[1, 0] == 0 and C[1, 1] == 1

        # Last row should sum to -1
        assert C[2, 0] == -1
        assert C[2, 1] == -1

        # Each column should sum to zero
        assert np.allclose(C.sum(axis=0), 0)

    def test_four_levels(self):
        """Test contrast matrix for 4 levels."""
        C = _contrast_sum_matrix(4)

        # Should be 4x3
        assert C.shape == (4, 3)

        # Each column should sum to zero
        for i in range(3):
            assert np.allclose(C[:, i].sum(), 0)

        # Last row should be all -1
        assert np.allclose(C[3, :], -1)

    def test_invalid_input(self):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError, match="at least 2 levels"):
            _contrast_sum_matrix(1)

        with pytest.raises(ValueError, match="at least 2 levels"):
            _contrast_sum_matrix(0)


class TestUnitVector:
    """Test _unit_vector helper function."""

    def test_basic_functionality(self):
        """Test unit vector creation."""
        u = _unit_vector(3)

        assert u.shape == (3, 1)
        assert np.allclose(u, np.ones((3, 1)))

    def test_different_sizes(self):
        """Test unit vectors of different sizes."""
        for n in [1, 2, 5, 10]:
            u = _unit_vector(n)
            assert u.shape == (n, 1)
            assert np.all(u == 1)


class TestFcontrastsEventFactor:
    """Test Fcontrasts for EventFactor (single categorical event)."""

    def test_single_factor_three_levels(self):
        """Test F-contrasts for single 3-level factor."""
        onsets = np.array([1, 2, 3, 4, 5, 6])
        values = ['A', 'B', 'C', 'A', 'B', 'C']

        event = EventFactor('condition', onsets, values)
        fcon = Fcontrasts(event)

        # Should have one contrast matrix
        assert len(fcon) == 1
        assert 'condition' in fcon

        # Check dimensions - 3 levels -> 2 contrast columns
        con_mat = fcon['condition']
        assert con_mat.shape == (3, 2)

        # Check sum-to-zero property
        assert np.allclose(con_mat.sum(axis=0), 0)

        # Verify structure matches _contrast_sum_matrix
        expected = _contrast_sum_matrix(3)
        assert np.allclose(con_mat, expected)

    def test_single_factor_two_levels(self):
        """Test F-contrasts for 2-level factor."""
        onsets = np.array([1, 2, 3, 4])
        values = ['A', 'B', 'A', 'B']

        event = EventFactor('group', onsets, values)
        fcon = Fcontrasts(event)

        assert len(fcon) == 1
        assert 'group' in fcon

        # 2 levels -> 1 contrast column
        con_mat = fcon['group']
        assert con_mat.shape == (2, 1)
        assert con_mat[0, 0] == 1
        assert con_mat[1, 0] == -1

    def test_single_level_factor(self):
        """Test that single-level factors return empty dict."""
        onsets = np.array([1, 2, 3, 4])
        values = ['A', 'A', 'A', 'A']

        event = EventFactor('constant', onsets, values)
        fcon = Fcontrasts(event)

        # Can't create contrasts for single level
        assert len(fcon) == 0

    def test_four_levels(self):
        """Test F-contrasts for 4-level factor."""
        onsets = np.arange(8)
        values = ['A', 'B', 'C', 'D'] * 2

        event = EventFactor('category', onsets, values)
        fcon = Fcontrasts(event)

        assert len(fcon) == 1
        con_mat = fcon['category']
        assert con_mat.shape == (4, 3)

        # Each column sums to zero
        for i in range(3):
            assert np.allclose(con_mat[:, i].sum(), 0)


class TestFcontrastsEventVariable:
    """Test Fcontrasts for EventVariable (continuous events)."""

    def test_continuous_event_returns_empty(self):
        """Test that continuous events return empty F-contrasts."""
        onsets = np.array([1, 2, 3, 4])
        values = np.array([1.5, 2.3, 3.1, 1.8])

        event = EventVariable('rating', onsets, values)
        fcon = Fcontrasts(event)

        # Continuous events have no F-contrasts
        assert len(fcon) == 0
        assert fcon == {}

    def test_multiple_continuous_events(self):
        """Test F-contrasts with different continuous events."""
        for values in [
            np.random.randn(10),
            np.arange(10, dtype=float),
            np.ones(5) * 2.5,
        ]:
            onsets = np.arange(len(values))
            event = EventVariable('var', onsets, values)
            fcon = Fcontrasts(event)
            assert len(fcon) == 0


class TestFcontrastsEventTerm:
    """Test Fcontrasts for EventTerm (interactions)."""

    def test_single_factor_term(self):
        """Test EventTerm with single factor matches EventFactor result."""
        onsets = np.array([1, 2, 3, 4, 5, 6])
        values = ['A', 'B', 'C', 'A', 'B', 'C']

        event = EventFactor('condition', onsets, values)
        term = EventTerm([event], name='condition')

        fcon_event = Fcontrasts(event)
        fcon_term = Fcontrasts(term)

        # Should be identical
        assert fcon_event.keys() == fcon_term.keys()
        assert np.allclose(fcon_event['condition'], fcon_term['condition'])

    def test_two_factor_interaction(self):
        """Test F-contrasts for 2x2 factorial design."""
        onsets = np.array([1, 2, 3, 4])

        # Two factors, 2 levels each
        factor1 = ['A', 'B', 'A', 'B']
        factor2 = ['X', 'X', 'Y', 'Y']

        ev1 = EventFactor('f1', onsets, factor1)
        ev2 = EventFactor('f2', onsets, factor2)

        term = create_interaction(ev1, ev2)
        fcon = Fcontrasts(term)

        # Should have main effects and interaction
        assert len(fcon) == 3
        assert 'f1' in fcon
        assert 'f2' in fcon
        assert 'f1:f2' in fcon

        # 2x2 = 4 cells
        # Each main effect: 1 df (2 levels - 1)
        # Interaction: 1 df ((2-1) * (2-1))
        assert fcon['f1'].shape == (4, 1)
        assert fcon['f2'].shape == (4, 1)
        assert fcon['f1:f2'].shape == (4, 1)

    def test_2x3_factorial(self):
        """Test F-contrasts for 2x3 factorial design."""
        onsets = np.arange(12)

        # 2 levels x 3 levels
        factor1 = ['A', 'B'] * 6
        factor2 = ['X', 'X', 'Y', 'Y', 'Z', 'Z'] * 2

        ev1 = EventFactor('condition', onsets, factor1)
        ev2 = EventFactor('group', onsets, factor2)

        term = create_interaction(ev1, ev2)
        fcon = Fcontrasts(term)

        assert len(fcon) == 3

        # 2x3 = 6 cells
        # condition: 1 df (2-1)
        # group: 2 df (3-1)
        # interaction: 2 df ((2-1) * (3-1))
        assert fcon['condition'].shape == (6, 1)
        assert fcon['group'].shape == (6, 2)
        assert fcon['condition:group'].shape == (6, 2)

    def test_three_factor_interaction(self):
        """Test F-contrasts for 3-way interaction."""
        onsets = np.arange(8)

        # 2x2x2 design
        f1 = ['A', 'B'] * 4
        f2 = ['X', 'X', 'Y', 'Y'] * 2
        f3 = ['1', '1', '1', '1', '2', '2', '2', '2']

        ev1 = EventFactor('f1', onsets, f1)
        ev2 = EventFactor('f2', onsets, f2)
        ev3 = EventFactor('f3', onsets, f3)

        term = EventTerm([ev1, ev2, ev3], name='f1:f2:f3')
        fcon = Fcontrasts(term, max_inter=3)

        # Should have all main effects, 2-way, and 3-way
        expected = ['f1', 'f2', 'f3', 'f1:f2', 'f1:f3', 'f2:f3', 'f1:f2:f3']
        assert sorted(fcon.keys()) == sorted(expected)

        # 2x2x2 = 8 cells
        assert fcon['f1'].shape == (8, 1)
        assert fcon['f2'].shape == (8, 1)
        assert fcon['f3'].shape == (8, 1)
        assert fcon['f1:f2'].shape == (8, 1)
        assert fcon['f1:f3'].shape == (8, 1)
        assert fcon['f2:f3'].shape == (8, 1)
        assert fcon['f1:f2:f3'].shape == (8, 1)  # (2-1)*(2-1)*(2-1) = 1

    def test_max_inter_parameter(self):
        """Test that max_inter limits interaction order."""
        onsets = np.arange(8)

        f1 = ['A', 'B'] * 4
        f2 = ['X', 'Y'] * 4
        f3 = ['1', '2'] * 4

        ev1 = EventFactor('f1', onsets, f1)
        ev2 = EventFactor('f2', onsets, f2)
        ev3 = EventFactor('f3', onsets, f3)

        term = EventTerm([ev1, ev2, ev3])

        # Limit to 2-way interactions
        fcon = Fcontrasts(term, max_inter=2)

        # Should NOT have 3-way
        assert 'f1:f2:f3' not in fcon

        # Should have 2-way
        assert 'f1:f2' in fcon
        assert 'f1:f3' in fcon
        assert 'f2:f3' in fcon

        # Should have main effects
        assert 'f1' in fcon
        assert 'f2' in fcon
        assert 'f3' in fcon

    def test_max_inter_one(self):
        """Test max_inter=1 gives only main effects."""
        onsets = np.arange(4)

        ev1 = EventFactor('f1', onsets, ['A', 'B', 'A', 'B'])
        ev2 = EventFactor('f2', onsets, ['X', 'X', 'Y', 'Y'])

        term = create_interaction(ev1, ev2)
        fcon = Fcontrasts(term, max_inter=1)

        # Only main effects
        assert len(fcon) == 2
        assert 'f1' in fcon
        assert 'f2' in fcon
        assert 'f1:f2' not in fcon

    def test_mixed_categorical_continuous(self):
        """Test term with both categorical and continuous events."""
        onsets = np.arange(8)

        # Categorical and continuous
        conditions = ['A', 'B'] * 4
        ratings = np.random.randn(8)

        cat_event = EventFactor('condition', onsets, conditions)
        cont_event = EventVariable('rating', onsets, ratings)

        term = EventTerm([cat_event, cont_event])
        fcon = Fcontrasts(term)

        # Only categorical gets F-contrast
        assert len(fcon) == 1
        assert 'condition' in fcon
        assert 'rating' not in fcon

    def test_kronecker_product_main_effect(self):
        """Test that main effects use correct Kronecker product pattern."""
        onsets = np.array([1, 2, 3, 4])

        # 2x2 design
        f1_vals = ['A', 'B', 'A', 'B']
        f2_vals = ['X', 'X', 'Y', 'Y']

        ev1 = EventFactor('f1', onsets, f1_vals)
        ev2 = EventFactor('f2', onsets, f2_vals)

        term = create_interaction(ev1, ev2)
        fcon = Fcontrasts(term)

        # Manually compute expected main effect for f1
        # Main effect of f1: D1 ⊗ C2
        D1 = _contrast_sum_matrix(2)  # 2x1: [[1], [-1]]
        C2 = _unit_vector(2)           # 2x1: [[1], [1]]

        expected_f1 = np.kron(D1, C2)

        assert np.allclose(fcon['f1'], expected_f1)

        # Main effect of f2: C1 ⊗ D2
        C1 = _unit_vector(2)
        D2 = _contrast_sum_matrix(2)

        expected_f2 = np.kron(C1, D2)

        assert np.allclose(fcon['f2'], expected_f2)

    def test_kronecker_product_interaction(self):
        """Test that interaction uses D1 ⊗ D2."""
        onsets = np.array([1, 2, 3, 4])

        ev1 = EventFactor('f1', onsets, ['A', 'B', 'A', 'B'])
        ev2 = EventFactor('f2', onsets, ['X', 'X', 'Y', 'Y'])

        term = create_interaction(ev1, ev2)
        fcon = Fcontrasts(term)

        # Interaction: D1 ⊗ D2
        D1 = _contrast_sum_matrix(2)
        D2 = _contrast_sum_matrix(2)

        expected_inter = np.kron(D1, D2)

        assert np.allclose(fcon['f1:f2'], expected_inter)


class TestFcontrastsEventModel:
    """Test Fcontrasts for EventModel (full model integration)."""

    def test_simple_model_single_term(self):
        """Test F-contrasts for simple model with one categorical term."""
        # Create events
        events = {
            'condition': EventFactor(
                name='condition',
                onsets=[1, 3, 5, 7, 9, 11],
                values=['A', 'B', 'C', 'A', 'B', 'C'],
                durations=1.0
            )
        }

        sampling = SamplingFrame(tr=2.0, n_scans=20)
        terms = [Term('condition')]

        model = EventModel(
            terms=terms,
            events=events,
            sampling_info=sampling
        )

        # Get F-contrasts
        fcon = Fcontrasts(model)

        # Should have one F-contrast
        assert len(fcon) >= 1

        # Check that contrast is mapped to full design matrix
        for name, con_mat in fcon.items():
            # Number of rows should match design matrix columns
            assert con_mat.shape[0] == model.design_matrix.shape[1]

    def test_model_with_two_factors(self):
        """Test F-contrasts for model with two categorical terms."""
        # Create events
        events = {
            'condition': EventFactor(
                name='condition',
                onsets=[1, 3, 5, 7, 9, 11],
                values=['A', 'B', 'C', 'A', 'B', 'C'],
                durations=1.0
            ),
            'group': EventFactor(
                name='group',
                onsets=[1, 3, 5, 7, 9, 11],
                values=['X', 'Y', 'X', 'Y', 'X', 'Y'],
                durations=1.0
            )
        }

        sampling = SamplingFrame(tr=2.0, n_scans=20)
        terms = [Term('condition'), Term('group')]

        model = EventModel(
            terms=terms,
            events=events,
            sampling_info=sampling
        )

        # Get F-contrasts
        fcon = Fcontrasts(model)

        # Should have F-contrasts for both terms
        assert len(fcon) >= 2

        # Check shape consistency
        ncols = model.design_matrix.shape[1]
        for name, con_mat in fcon.items():
            assert con_mat.shape[0] == ncols


class TestBasisFilter:
    """Test filter_basis function."""

    def test_no_filtering_when_basis_none(self):
        """Test that basis=None returns all conditions."""
        condnames = ['A_b01', 'A_b02', 'B_b01', 'B_b02']
        result = filter_basis(condnames, basis=None, nbasis=2)

        assert result == condnames

    def test_no_filtering_when_basis_all(self):
        """Test that basis='all' returns all conditions."""
        condnames = ['A_b01', 'A_b02', 'B_b01', 'B_b02']
        result = filter_basis(condnames, basis='all', nbasis=2)

        assert result == condnames

    def test_no_filtering_when_nbasis_one(self):
        """Test that nbasis=1 returns all conditions."""
        condnames = ['A', 'B', 'C']
        result = filter_basis(condnames, basis=[1], nbasis=1)

        assert result == condnames

    def test_filter_first_basis(self):
        """Test filtering for first basis function only."""
        condnames = ['A_b01', 'A_b02', 'A_b03', 'B_b01', 'B_b02', 'B_b03']
        result = filter_basis(condnames, basis=[1], nbasis=3)

        expected = ['A_b01', 'B_b01']
        assert result == expected

    def test_filter_second_basis(self):
        """Test filtering for second basis function."""
        condnames = ['A_b01', 'A_b02', 'A_b03', 'B_b01', 'B_b02', 'B_b03']
        result = filter_basis(condnames, basis=[2], nbasis=3)

        expected = ['A_b02', 'B_b02']
        assert result == expected

    def test_filter_multiple_bases(self):
        """Test filtering for multiple basis functions."""
        condnames = ['A_b01', 'A_b02', 'A_b03', 'B_b01', 'B_b02', 'B_b03']
        result = filter_basis(condnames, basis=[1, 3], nbasis=3)

        expected = ['A_b01', 'A_b03', 'B_b01', 'B_b03']
        assert result == expected

    def test_filter_handles_no_suffix(self):
        """Test that conditions without basis suffix are kept if basis=[1]."""
        condnames = ['A', 'B', 'C_b01', 'C_b02']
        result = filter_basis(condnames, basis=[1], nbasis=2)

        # A and B have no suffix, so kept if 1 in basis
        # C_b01 matches
        expected = ['A', 'B', 'C_b01']
        assert result == expected

    def test_filter_excludes_no_suffix_when_not_basis_one(self):
        """Test that no-suffix conditions excluded when basis doesn't include 1."""
        condnames = ['A', 'B', 'C_b01', 'C_b02']
        result = filter_basis(condnames, basis=[2], nbasis=2)

        # Only C_b02 should match
        expected = ['C_b02']
        assert result == expected

    def test_filter_integer_input(self):
        """Test that integer basis input is converted to list."""
        condnames = ['A_b01', 'A_b02', 'B_b01', 'B_b02']
        result = filter_basis(condnames, basis=1, nbasis=2)

        expected = ['A_b01', 'B_b01']
        assert result == expected

    def test_filter_empty_result_warning(self):
        """Test warning when filtering removes all conditions."""
        condnames = ['A_b01', 'A_b02']

        with pytest.warns(UserWarning, match="removed all conditions"):
            result = filter_basis(condnames, basis=[3], nbasis=2, contrast_name='test')

            # Result should be empty
            assert result == []

    def test_filter_realistic_spmg3(self):
        """Test realistic scenario with SPM canonical + derivatives (3 bases)."""
        condnames = [
            'face_b01', 'face_b02', 'face_b03',
            'scene_b01', 'scene_b02', 'scene_b03',
            'object_b01', 'object_b02', 'object_b03'
        ]

        # Filter for canonical only (b01)
        result = filter_basis(condnames, basis=[1], nbasis=3)
        expected = ['face_b01', 'scene_b01', 'object_b01']
        assert result == expected

        # Filter for derivatives only (b02, b03)
        result2 = filter_basis(condnames, basis=[2, 3], nbasis=3)
        expected2 = [
            'face_b02', 'face_b03',
            'scene_b02', 'scene_b03',
            'object_b02', 'object_b03'
        ]
        assert result2 == expected2


class TestApplyBasisFilter:
    """Test apply_basis_filter function."""

    def test_no_filtering_nbasis_one(self):
        """Test that nbasis=1 returns unchanged weights."""
        weights = np.array([[1.0], [0.5], [-0.5]])
        condnames = ['A', 'B', 'C']

        result = apply_basis_filter(weights, condnames, basis_spec=None, nbasis=1)

        assert np.allclose(result['weights'], weights)
        assert result['condnames'] == condnames
        assert result['nbasis'] == 1

    def test_nbasis_one_with_basis_spec_warns(self):
        """Test warning when basis filtering requested but nbasis=1."""
        weights = np.array([[1.0], [0.5]])
        condnames = ['A', 'B']

        with pytest.warns(UserWarning, match="no multi-basis HRF"):
            result = apply_basis_filter(
                weights, condnames, basis_spec=[1], nbasis=1, contrast_name='test'
            )

            # Weights unchanged
            assert np.allclose(result['weights'], weights)

    def test_no_filtering_basis_none(self):
        """Test that basis_spec=None returns unchanged weights."""
        weights = np.array([[1.0], [0.5], [0.3], [0.2]])
        condnames = ['A_b01', 'A_b02', 'B_b01', 'B_b02']

        result = apply_basis_filter(weights, condnames, basis_spec=None, nbasis=2)

        assert np.allclose(result['weights'], weights)
        assert result['condnames'] == condnames

    def test_no_filtering_basis_all(self):
        """Test that basis_spec='all' returns unchanged weights."""
        weights = np.array([[1.0], [0.5], [0.3], [0.2]])
        condnames = ['A_b01', 'A_b02', 'B_b01', 'B_b02']

        result = apply_basis_filter(weights, condnames, basis_spec='all', nbasis=2)

        assert np.allclose(result['weights'], weights)
        assert result['condnames'] == condnames

    def test_filter_first_basis_zeros_others(self):
        """Test filtering first basis zeros out other basis weights."""
        weights = np.array([[1.0], [0.5], [0.3], [0.2]])
        condnames = ['A_b01', 'A_b02', 'B_b01', 'B_b02']

        result = apply_basis_filter(
            weights, condnames, basis_spec=[1], nbasis=2, contrast_name='test'
        )

        # A_b01 and B_b01 should keep weights
        # A_b02 and B_b02 should be zeroed
        expected_weights = np.array([[1.0], [0.0], [0.3], [0.0]])

        assert np.allclose(result['weights'], expected_weights)
        assert result['condnames'] == ['A_b01', 'B_b01']

    def test_filter_second_basis(self):
        """Test filtering second basis function."""
        weights = np.array([[1.0], [0.5], [0.3], [0.2]])
        condnames = ['A_b01', 'A_b02', 'B_b01', 'B_b02']

        result = apply_basis_filter(
            weights, condnames, basis_spec=[2], nbasis=2
        )

        # Only b02 should keep weights
        expected_weights = np.array([[0.0], [0.5], [0.0], [0.2]])

        assert np.allclose(result['weights'], expected_weights)
        assert result['condnames'] == ['A_b02', 'B_b02']

    def test_filter_multiple_bases(self):
        """Test filtering multiple basis functions."""
        weights = np.array([[1.0], [0.5], [0.3], [0.2], [0.1], [0.05]])
        condnames = ['A_b01', 'A_b02', 'A_b03', 'B_b01', 'B_b02', 'B_b03']

        result = apply_basis_filter(
            weights, condnames, basis_spec=[1, 3], nbasis=3
        )

        # Keep b01 and b03, zero b02
        expected_weights = np.array([[1.0], [0.0], [0.3], [0.2], [0.0], [0.05]])

        assert np.allclose(result['weights'], expected_weights)
        assert result['condnames'] == ['A_b01', 'A_b03', 'B_b01', 'B_b03']

    def test_basis_weights_application(self):
        """Test that basis_weights are applied to filtered conditions."""
        weights = np.array([[1.0], [1.0], [1.0], [1.0]])
        condnames = ['A_b01', 'A_b02', 'B_b01', 'B_b02']

        # Apply weights [0.7, 0.3] to two bases
        basis_weights = np.array([0.7, 0.3])

        result = apply_basis_filter(
            weights,
            condnames,
            basis_spec=[1, 2],
            nbasis=2,
            basis_weights=basis_weights
        )

        # b01 should get 0.7, b02 should get 0.3
        expected_weights = np.array([[0.7], [0.3], [0.7], [0.3]])

        assert np.allclose(result['weights'], expected_weights)

    def test_basis_weights_normalization(self):
        """Test that basis_weights are normalized to sum to 1."""
        weights = np.array([[1.0], [1.0], [1.0], [1.0]])
        condnames = ['A_b01', 'A_b02', 'B_b01', 'B_b02']

        # Non-normalized weights (sum to 2)
        basis_weights = np.array([1.0, 1.0])

        with pytest.warns(UserWarning, match="normalizing"):
            result = apply_basis_filter(
                weights,
                condnames,
                basis_spec=[1, 2],
                nbasis=2,
                basis_weights=basis_weights,
                contrast_name='test'
            )

            # Weights should be normalized to 0.5 each
            expected_weights = np.array([[0.5], [0.5], [0.5], [0.5]])
            assert np.allclose(result['weights'], expected_weights)

    def test_basis_weights_length_mismatch_error(self):
        """Test error when basis_weights length doesn't match selected bases."""
        weights = np.array([[1.0], [1.0], [1.0], [1.0]])
        condnames = ['A_b01', 'A_b02', 'B_b01', 'B_b02']

        # Selecting 2 bases but providing 3 weights
        basis_weights = np.array([0.5, 0.3, 0.2])

        with pytest.raises(ValueError, match="basis_weights length"):
            apply_basis_filter(
                weights,
                condnames,
                basis_spec=[1, 2],
                nbasis=2,
                basis_weights=basis_weights,
                contrast_name='test'
            )

    def test_multi_column_contrast(self):
        """Test filtering works with multi-column contrast matrices."""
        # F-contrast with 2 columns
        weights = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-0.5, -0.5],
            [0.5, 0.5],
            [0.3, 0.2],
            [-0.3, -0.2]
        ])
        condnames = ['A_b01', 'A_b02', 'A_b03', 'B_b01', 'B_b02', 'B_b03']

        result = apply_basis_filter(
            weights, condnames, basis_spec=[1], nbasis=3
        )

        # Should zero out b02 and b03 rows
        expected_weights = np.array([
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.5],
            [0.0, 0.0],
            [0.0, 0.0]
        ])

        assert np.allclose(result['weights'], expected_weights)
        assert result['condnames'] == ['A_b01', 'B_b01']

    def test_realistic_spmg3_filtering(self):
        """Test realistic SPM canonical + derivatives scenario."""
        # 3 conditions x 3 basis functions = 9 rows
        weights = np.ones((9, 2))  # 2-column F-contrast
        condnames = [
            'face_b01', 'face_b02', 'face_b03',
            'scene_b01', 'scene_b02', 'scene_b03',
            'object_b01', 'object_b02', 'object_b03'
        ]

        # Filter for canonical HRF only
        result = apply_basis_filter(
            weights, condnames, basis_spec=[1], nbasis=3, contrast_name='canonical_only'
        )

        # Only rows 0, 3, 6 (b01) should be non-zero
        expected_nonzero_indices = [0, 3, 6]
        for i in range(9):
            if i in expected_nonzero_indices:
                assert np.all(result['weights'][i, :] == 1.0)
            else:
                assert np.all(result['weights'][i, :] == 0.0)

        assert result['condnames'] == ['face_b01', 'scene_b01', 'object_b01']
        assert result['nbasis'] == 3


class TestFcontrastsIntegration:
    """Integration tests combining Fcontrasts with basis filtering concepts."""

    def test_fcontrasts_with_basis_naming(self):
        """Test F-contrasts work correctly when conditions have basis suffixes."""
        # This tests the integration: Fcontrasts generates matrices,
        # then basis_filter can be used to select specific bases

        onsets = np.arange(6)
        values = ['A', 'B', 'C', 'A', 'B', 'C']

        event = EventFactor('condition', onsets, values)
        fcon = Fcontrasts(event)

        # Get the contrast matrix
        con_mat = fcon['condition']
        assert con_mat.shape == (3, 2)

        # Simulate expanded basis names (what would happen with nbasis=3)
        expanded_names = [
            'A_b01', 'A_b02', 'A_b03',
            'B_b01', 'B_b02', 'B_b03',
            'C_b01', 'C_b02', 'C_b03'
        ]

        # Expand contrast matrix for 3 bases per condition
        expanded_con = np.kron(con_mat, np.ones((3, 1)))

        # Now filter for basis 1 only
        result = apply_basis_filter(
            expanded_con,
            expanded_names,
            basis_spec=[1],
            nbasis=3
        )

        # Should keep only b01 entries
        assert result['condnames'] == ['A_b01', 'B_b01', 'C_b01']

        # First 3 rows should match original, others zeroed
        for i in range(9):
            if i % 3 == 0:  # b01 rows
                assert not np.allclose(result['weights'][i, :], 0)
            else:
                assert np.allclose(result['weights'][i, :], 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
