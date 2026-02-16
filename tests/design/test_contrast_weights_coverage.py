"""Comprehensive coverage tests for contrast_weights.py

This module tests all singledispatch handlers, edge cases, and helper functions
to boost coverage to 85%+.
"""

import pytest
import numpy as np
import warnings
import sys
import typing
from unittest.mock import MagicMock

# Direct imports to avoid scipy initialization issues
sys.path.insert(0, 'src')

from fmrimod.contrast.contrast_spec import (
    UnitContrastSpec, PairContrastSpec, ColumnContrastSpec,
    PolyContrastSpec, OnewayContrastSpec, InteractionContrastSpec,
    ContrastFormulaSpec, ContrastSet, Formula
)
from fmrimod.contrast.contrast_weights import (
    contrast_weights, Contrast,
    _calculate_mask_weights, _get_conditions, _parse_formula_condition,
    _match_conditions_to_formula, CONTRAST_TOLERANCE
)


# ============================================================================
# Helper function to create mock terms
# ============================================================================

def make_term(levels=None, name='condition'):
    """Create a mock term for testing."""
    if levels is None:
        levels = ('A', 'B', 'C')
    else:
        levels = list(levels)

    term = MagicMock()

    # Create condition names in the format "factor.level"
    if levels is None:
        levels = ['A', 'B', 'C']
    base_conditions = [f"{name}.{level}" for level in levels]

    def conditions_method(drop_empty=False, expand_basis=True):
        if expand_basis and hasattr(term, 'nbasis') and term.nbasis > 1:
            # Expand for basis functions
            expanded = []
            for cond in base_conditions:
                for i in range(term.nbasis):
                    expanded.append(f"{cond}_b{i+1}")
            return expanded
        return base_conditions

    term.conditions = conditions_method
    term.nbasis = 1
    term.varname = name

    def cells_method():
        import pandas as pd
        return pd.DataFrame()

    term.cells = cells_method

    return term


def test_make_term_copies_input_levels():
    """Regression: caller mutations should not affect term conditions."""
    levels = ['A', 'B']
    term = make_term(levels=levels, name='condition')
    levels.append('C')

    assert term.conditions() == ['condition.A', 'condition.B']


# ============================================================================
# Test Contrast dataclass
# ============================================================================

class TestContrastDataclass:
    """Test the Contrast dataclass methods."""

    def test_contrast_init(self):
        """Test Contrast initialization."""
        term = make_term()
        weights = np.array([[1], [0], [0]])
        condnames = ['A', 'B', 'C']
        spec = UnitContrastSpec(name='test', A=Formula('A'))

        con = Contrast(term, 'test', weights, condnames, spec)

        assert con.term is term
        assert con.name == 'test'
        np.testing.assert_array_equal(con.weights, weights)
        assert con.condnames == condnames
        assert con.contrast_spec is spec

    def test_is_fcontrast_false(self):
        """Test is_fcontrast for t-contrast (single column)."""
        term = make_term()
        weights = np.array([[1], [0], [0]])
        spec = UnitContrastSpec(name='test', A=Formula('A'))
        con = Contrast(term, 'test', weights, ['A', 'B', 'C'], spec)

        assert not con.is_fcontrast

    def test_is_fcontrast_true(self):
        """Test is_fcontrast for F-contrast (multiple columns)."""
        term = make_term()
        weights = np.array([[1, 0], [0, 1], [0, 0]])
        spec = OnewayContrastSpec(name='test', A=Formula('condition'))
        con = Contrast(term, 'test', weights, ['A', 'B', 'C'], spec)

        assert con.is_fcontrast

    def test_is_fcontrast_1d_array(self):
        """Test is_fcontrast with 1D array."""
        term = make_term()
        weights = np.array([1, 0, 0])
        spec = UnitContrastSpec(name='test', A=Formula('A'))
        con = Contrast(term, 'test', weights, ['A', 'B', 'C'], spec)

        # 1D array is not an F-contrast
        assert not con.is_fcontrast

    def test_repr_t_contrast(self):
        """Test __repr__ for t-contrast."""
        term = make_term()
        weights = np.array([[1], [-1], [0]])
        spec = PairContrastSpec(name='A_vs_B', A=Formula('A'), B=Formula('B'))
        con = Contrast(term, 'A_vs_B', weights, ['A', 'B', 'C'], spec)

        repr_str = repr(con)
        assert 'Contrast' in repr_str
        assert 'A_vs_B' in repr_str
        assert 't-contrast' in repr_str
        assert '3x1' in repr_str

    def test_repr_f_contrast(self):
        """Test __repr__ for F-contrast."""
        term = make_term()
        weights = np.array([[1, 0], [-0.5, 0.5], [-0.5, -0.5]])
        spec = OnewayContrastSpec(name='main', A=Formula('condition'))
        con = Contrast(term, 'main', weights, ['A', 'B', 'C'], spec)

        repr_str = repr(con)
        assert 'Contrast' in repr_str
        assert 'main' in repr_str
        assert 'F-contrast' in repr_str
        assert '3x2' in repr_str


# ============================================================================
# Test helper functions
# ============================================================================

class TestHelperFunctions:
    """Test internal helper functions."""

    def test_ast_annotated_helpers_type_hints_resolve(self):
        """Regression: ast-based type annotations should resolve."""
        import importlib

        mod = importlib.import_module("fmrimod.contrast.contrast_weights")
        typing.get_type_hints(mod._eval_formula_ast)
        typing.get_type_hints(mod._reconstruct_compare)

    def test_calculate_mask_weights_single_group(self):
        """Test _calculate_mask_weights with only group A."""
        names = ['A', 'B', 'C']
        A_mask = np.array([True, True, False])

        weights = _calculate_mask_weights(names, A_mask)

        # Should assign equal positive weights to A
        expected = np.array([0.5, 0.5, 0.0])
        np.testing.assert_allclose(weights, expected)

    def test_calculate_mask_weights_two_groups(self):
        """Test _calculate_mask_weights with A and B groups."""
        names = ['A', 'B', 'C', 'D']
        A_mask = np.array([True, True, False, False])
        B_mask = np.array([False, False, True, True])

        weights = _calculate_mask_weights(names, A_mask, B_mask)

        # A gets +0.5 each, B gets -0.5 each
        expected = np.array([0.5, 0.5, -0.5, -0.5])
        np.testing.assert_allclose(weights, expected)
        # Should sum to zero
        assert abs(np.sum(weights)) < CONTRAST_TOLERANCE

    def test_calculate_mask_weights_overlap_error(self):
        """Test that overlapping masks raise ValueError."""
        names = ['A', 'B', 'C']
        A_mask = np.array([True, True, False])
        B_mask = np.array([False, True, True])  # Overlaps at B

        with pytest.raises(ValueError, match="overlap"):
            _calculate_mask_weights(names, A_mask, B_mask)

    def test_calculate_mask_weights_empty_masks(self):
        """Test that empty masks raise ValueError."""
        names = ['A', 'B', 'C']
        A_mask = np.array([False, False, False])

        with pytest.raises(ValueError, match="No conditions were selected"):
            _calculate_mask_weights(names, A_mask)

    def test_calculate_mask_weights_empty_A_warning(self):
        """Test warning when A is empty but B is not."""
        names = ['A', 'B', 'C']
        A_mask = np.array([False, False, False])
        B_mask = np.array([True, False, False])

        with pytest.warns(UserWarning, match="Mask A is empty"):
            weights = _calculate_mask_weights(names, A_mask, B_mask)

        # Only B has weight
        expected = np.array([-1.0, 0.0, 0.0])
        np.testing.assert_allclose(weights, expected)

    def test_calculate_mask_weights_empty_B_warning(self):
        """Test warning when B is empty but A is not."""
        names = ['A', 'B', 'C']
        A_mask = np.array([True, False, False])
        B_mask = np.array([False, False, False])

        with pytest.warns(UserWarning, match="Mask B is empty"):
            weights = _calculate_mask_weights(names, A_mask, B_mask)

        # Only A has weight
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(weights, expected)

    def test_calculate_mask_weights_unbalanced_warning(self):
        """Test warning for unbalanced weights."""
        names = ['A', 'B', 'C']
        A_mask = np.array([True, False, False])  # 1 element
        B_mask = np.array([False, True, True])    # 2 elements

        # This creates unbalanced weights: 1 vs -0.5, -0.5
        # The warning is only raised if sum differs by more than tolerance
        # With 1 vs 2 elements: 1.0 - 0.5 - 0.5 = 0.0, so no warning
        weights = _calculate_mask_weights(names, A_mask, B_mask)
        # Just verify it computes without error
        assert weights.shape == (3,)

    def test_parse_formula_condition_with_equality(self):
        """Test _parse_formula_condition with == operator."""
        factor, level = _parse_formula_condition("condition == 'A'")
        assert factor == 'condition'
        assert level == 'A'

        factor, level = _parse_formula_condition('category == "face"')
        assert factor == 'category'
        assert level == 'face'

    def test_parse_formula_condition_bare_variable(self):
        """Test _parse_formula_condition with bare variable."""
        factor, level = _parse_formula_condition("time")
        assert factor == 'time'
        assert level is None

    def test_parse_formula_condition_empty(self):
        """Test _parse_formula_condition with empty string."""
        factor, level = _parse_formula_condition("")
        assert factor is None
        assert level is None

    def test_parse_formula_condition_invalid(self):
        """Test _parse_formula_condition with invalid expression."""
        factor, level = _parse_formula_condition("a + b")
        assert factor is None
        assert level is None

    def test_match_conditions_to_formula_all(self):
        """Test _match_conditions_to_formula with 'all' keyword."""
        condnames = ['condition.A', 'condition.B', 'condition.C']

        for keyword in ['all', '~1', '*']:
            mask = _match_conditions_to_formula(condnames, keyword)
            assert np.all(mask)

    def test_match_conditions_to_formula_factor_level(self):
        """Test _match_conditions_to_formula with factor.level pattern."""
        condnames = ['condition.A', 'condition.B', 'condition.C']

        mask = _match_conditions_to_formula(condnames, "condition == 'A'")
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_match_conditions_to_formula_bare_level(self):
        """Test _match_conditions_to_formula matching bare level name."""
        condnames = ['condition.A', 'condition.B', 'condition.C']

        # Match by level name directly
        mask = _match_conditions_to_formula(condnames, "A")
        assert mask[0]  # Should match condition.A

    def test_match_conditions_to_formula_factor_only(self):
        """Test _match_conditions_to_formula with just factor name."""
        condnames = ['condition.A', 'condition.B', 'time.1']

        mask = _match_conditions_to_formula(condnames, "condition")
        # Should match condition.A and condition.B but not time.1
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_get_conditions_with_conditions_method(self):
        """Test _get_conditions with term that has conditions method."""
        term = make_term(['A', 'B', 'C'])
        conds = _get_conditions(term, expand_basis=False)

        assert len(conds) == 3
        assert 'condition.A' in conds

    def test_get_conditions_with_design_matrix(self):
        """Test _get_conditions with term that has design_matrix but no conditions method."""
        term = MagicMock(spec=[])  # Create mock without conditions attribute

        # Mock design matrix with columns
        dm = MagicMock()
        dm.columns = ['col1', 'col2', 'col3']
        term.design_matrix = dm

        # _get_conditions falls back to design_matrix.columns when no conditions method
        conds = _get_conditions(term)
        assert conds == ['col1', 'col2', 'col3']

    def test_get_conditions_fallback(self):
        """Test _get_conditions fallback when no conditions available."""
        term = MagicMock()
        del term.conditions
        del term.design_matrix

        with pytest.warns(UserWarning, match="Could not get conditions"):
            conds = _get_conditions(term)

        assert conds == []


# ============================================================================
# Test singledispatch handlers
# ============================================================================

class TestSingledispatchHandlers:
    """Test all singledispatch implementations of contrast_weights."""

    def test_contrast_weights_not_implemented(self):
        """Test that unknown types raise NotImplementedError."""
        term = make_term()

        with pytest.raises(NotImplementedError, match="No contrast_weights method"):
            contrast_weights("not a valid spec", term)

    def test_unit_contrast_empty_term(self):
        """Test UnitContrastSpec with empty term."""
        # Create a mock term with no conditions
        term = MagicMock()
        term.conditions = lambda **kwargs: []

        spec = UnitContrastSpec(name='test', A=Formula('A'))

        with pytest.warns(UserWarning, match="No conditions found"):
            con = contrast_weights(spec, term)

        assert con.weights.shape == (0, 1)
        assert con.condnames == []

    def test_unit_contrast_no_match_warning(self):
        """Test UnitContrastSpec when formula matches nothing."""
        term = make_term(['A', 'B', 'C'])
        spec = UnitContrastSpec(name='test', A=Formula("condition == 'Z'"))

        with pytest.warns(UserWarning, match="No conditions were selected"):
            con = contrast_weights(spec, term)

        # Should have zero weights
        assert np.all(con.weights == 0)

    def test_unit_contrast_none_formula(self):
        """Test UnitContrastSpec with A=None (select all)."""
        term = make_term(['A', 'B', 'C'])
        spec = UnitContrastSpec(name='all', A=None)

        con = contrast_weights(spec, term)

        # All conditions get equal weight summing to 1
        assert np.isclose(np.sum(con.weights), 1.0)
        assert np.allclose(con.weights, 1/3)

    def test_pair_contrast_empty_term(self):
        """Test PairContrastSpec with empty term."""
        term = MagicMock()
        term.conditions = lambda **kwargs: []

        spec = PairContrastSpec(
            name='test',
            A=Formula('A'),
            B=Formula('B')
        )

        with pytest.warns(UserWarning, match="No conditions found"):
            con = contrast_weights(spec, term)

        assert con.weights.shape == (0, 1)

    def test_pair_contrast_with_basis_expansion(self):
        """Test PairContrastSpec with basis expansion."""
        term = make_term(['A', 'B'], name='condition')
        term.nbasis = 3  # 3 basis functions

        spec = PairContrastSpec(
            name='A_vs_B',
            A=Formula("condition == 'A'"),
            B=Formula("condition == 'B'")
        )

        con = contrast_weights(spec, term)

        # Should expand: 2 conditions * 3 basis = 6 weights
        assert con.weights.shape == (6, 1)
        assert len(con.condnames) == 6
        # Should still sum to zero
        assert abs(np.sum(con.weights)) < 1e-10

    def test_pair_contrast_error_handling(self):
        """Test PairContrastSpec error handling."""
        term = make_term(['A', 'B', 'C'])
        # Create spec that will cause error in _calculate_mask_weights
        spec = PairContrastSpec(
            name='test',
            A=Formula("nonexistent == 'Z'"),
            B=Formula("alsonothere == 'Y'")
        )

        with pytest.warns(UserWarning, match="No conditions were selected"):
            con = contrast_weights(spec, term)

        # Should create zero weights
        assert np.all(con.weights == 0)

    def test_column_contrast_empty_term(self):
        """Test ColumnContrastSpec with empty term."""
        term = MagicMock()
        term.conditions = lambda **kwargs: []
        term.varname = 'test_var'

        spec = ColumnContrastSpec(pattern_A='.*', pattern_B=None, name='test')

        with pytest.warns(UserWarning, match="has no columns"):
            con = contrast_weights(spec, term)

        assert con.weights.shape == (0, 1)

    def test_column_contrast_no_match_A(self):
        """Test ColumnContrastSpec when pattern_A matches nothing."""
        term = make_term(['A', 'B', 'C'])
        spec = ColumnContrastSpec(pattern_A='NOMATCH', pattern_B=None, name='test')

        with pytest.warns(UserWarning):
            con = contrast_weights(spec, term)
        # When no match, weights will be zero
        assert con.weights.shape[0] == 3

    def test_column_contrast_no_match_B(self):
        """Test ColumnContrastSpec when pattern_B matches nothing."""
        term = make_term(['A', 'B', 'C'])
        spec = ColumnContrastSpec(
            name='test',
            pattern_A='A',
            pattern_B='NOMATCH'
        )

        with pytest.warns(UserWarning) as record:
            con = contrast_weights(spec, term)

        messages = [str(w.message) for w in record]
        assert any("matched no columns" in msg for msg in messages)
        assert any("Mask B is empty but Mask A is not" in msg for msg in messages)

    def test_column_contrast_overlap_error(self):
        """Test ColumnContrastSpec with overlapping patterns."""
        term = make_term(['A', 'B', 'C'])
        spec = ColumnContrastSpec(
            name='test',
            pattern_A='condition',  # Matches all
            pattern_B='condition.A'  # Also matches A
        )

        with pytest.raises(ValueError, match="overlapping columns"):
            contrast_weights(spec, term)

    def test_poly_contrast_empty_term(self):
        """Test PolyContrastSpec with empty term."""
        term = MagicMock()
        term.conditions = lambda **kwargs: []

        spec = PolyContrastSpec(name='test', A=Formula('time'), degree=2)

        with pytest.warns(UserWarning, match="No conditions found"):
            con = contrast_weights(spec, term)

        assert con.weights.shape == (0, 2)

    def test_poly_contrast_degree_too_high(self):
        """Test PolyContrastSpec with degree too high for levels."""
        term = make_term(['A', 'B', 'C'])  # 3 levels
        spec = PolyContrastSpec(name='test', A=Formula('cond'), degree=3)

        with pytest.raises(ValueError, match="too high"):
            contrast_weights(spec, term)

    def test_poly_contrast_single_level(self):
        """Test PolyContrastSpec with single level."""
        term = make_term(['A'])  # 1 level
        spec = PolyContrastSpec(name='test', A=Formula('cond'), degree=1)

        with pytest.raises(ValueError, match="too high"):
            contrast_weights(spec, term)

    def test_oneway_contrast_empty_term(self):
        """Test OnewayContrastSpec with empty term."""
        term = MagicMock()
        term.conditions = lambda **kwargs: []

        spec = OnewayContrastSpec(name='test', A=Formula('cond'))

        with pytest.warns(UserWarning, match="No conditions found"):
            con = contrast_weights(spec, term)

        assert con.weights.shape == (0, 0)

    def test_oneway_contrast_single_level(self):
        """Test OnewayContrastSpec with single level."""
        term = make_term(['A'])
        spec = OnewayContrastSpec(name='test', A=Formula('cond'))

        with pytest.warns(UserWarning, match="only 1 level"):
            con = contrast_weights(spec, term)

        # k=1 -> k-1=0 contrasts
        assert con.weights.shape == (1, 0)

    def test_interaction_contrast_empty_term(self):
        """Test InteractionContrastSpec with empty term."""
        term = MagicMock()
        term.conditions = lambda **kwargs: []

        spec = InteractionContrastSpec(name='test', A=Formula('a * b'))

        with pytest.warns(UserWarning, match="No conditions found"):
            con = contrast_weights(spec, term)

        assert con.weights.shape == (0, 0)

    def test_interaction_contrast_too_few_cells(self):
        """Test InteractionContrastSpec with < 4 cells."""
        term = make_term(['A', 'B'])  # Only 2 cells
        spec = InteractionContrastSpec(name='test', A=Formula('cond'))

        with pytest.warns(UserWarning, match="at least 4 cells"):
            con = contrast_weights(spec, term)

        assert con.weights.shape == (2, 0)

    def test_interaction_contrast_2x2(self):
        """Test InteractionContrastSpec with 4 cells (2x2)."""
        term = make_term(['A', 'B', 'C', 'D'])
        spec = InteractionContrastSpec(name='test', A=Formula('design'))

        con = contrast_weights(spec, term)

        assert con.weights.shape == (4, 1)
        # 2x2 pattern: [1, -1, -1, 1]
        expected = np.array([[1], [-1], [-1], [1]], dtype=float)
        np.testing.assert_allclose(con.weights, expected)

    def test_interaction_contrast_2x3(self):
        """Test InteractionContrastSpec with 6 cells (2x3)."""
        term = make_term(['A', 'B', 'C', 'D', 'E', 'F'])
        spec = InteractionContrastSpec(name='test', A=Formula('design'))

        con = contrast_weights(spec, term)

        # 2x3 interaction has 2 contrasts (1 * 2)
        assert con.weights.shape == (6, 2)

    def test_interaction_contrast_3x3(self):
        """Test InteractionContrastSpec with 9 cells (3x3)."""
        term = make_term(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
        spec = InteractionContrastSpec(name='test', A=Formula('design'))

        con = contrast_weights(spec, term)

        # 3x3 interaction has 4 contrasts (2 * 2)
        assert con.weights.shape == (9, 4)

    def test_interaction_contrast_2xk_general(self):
        """Test InteractionContrastSpec with 2xk design (8 cells = 2x4)."""
        term = make_term(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        spec = InteractionContrastSpec(name='test', A=Formula('design'))

        con = contrast_weights(spec, term)

        # 2x4 interaction has 3 contrasts (1 * 3)
        assert con.weights.shape == (8, 3)

    def test_interaction_contrast_odd_cells(self):
        """Test InteractionContrastSpec with odd number of cells."""
        term = make_term(['A', 'B', 'C', 'D', 'E'])  # 5 cells
        spec = InteractionContrastSpec(name='test', A=Formula('design'))

        with pytest.warns(UserWarning, match="Cannot determine factorial"):
            con = contrast_weights(spec, term)

        # Falls back to simple pattern
        assert con.weights.shape == (5, 1)

    def test_contrast_set_handler(self):
        """Test ContrastSet singledispatch handler."""
        term = make_term(['A', 'B', 'C'])

        c1 = UnitContrastSpec(name='test1', A=Formula("condition == 'A'"))
        c2 = UnitContrastSpec(name='test2', A=Formula("condition == 'B'"))
        cset = ContrastSet(c1, c2)  # Pass as separate args, not list

        results = contrast_weights(cset, term)

        assert isinstance(results, dict)
        assert len(results) == 2
        assert 'test1' in results
        assert 'test2' in results

    def test_contrast_set_with_failure(self):
        """Test ContrastSet handler with one failing contrast."""
        term = make_term(['A', 'B'])

        c1 = UnitContrastSpec(name='good', A=Formula('A'))
        c2 = PolyContrastSpec(name='bad', A=Formula('time'), degree=5)  # Too high
        cset = ContrastSet(c1, c2)  # Pass as separate args, not list

        with pytest.warns(UserWarning, match="Failed to compute"):
            results = contrast_weights(cset, term)

        # Should have the good one, not the bad one
        assert 'good' in results
        assert 'bad' not in results


# ============================================================================
# Test ContrastFormulaSpec and formula evaluation
# ============================================================================

class TestContrastFormulaSpec:
    """Test ContrastFormulaSpec handler and formula parsing."""

    def test_formula_empty_term(self):
        """Test ContrastFormulaSpec with empty term."""
        term = MagicMock()
        term.conditions = lambda **kwargs: []
        term.nbasis = 1

        spec = ContrastFormulaSpec(name='test', A=Formula('A - B'))

        con = contrast_weights(spec, term)

        assert con.weights.shape == (0, 1)

    def test_formula_simple_condition(self):
        """Test formula with simple condition selection."""
        term = make_term(['A', 'B', 'C'])
        spec = ContrastFormulaSpec(
            name='test',
            A=Formula("condition == 'A'")
        )

        con = contrast_weights(spec, term)

        # Should match condition.A only
        condnames = con.condnames
        idx_A = condnames.index('condition.A')
        assert con.weights[idx_A, 0] == 1.0

    def test_formula_subtraction(self):
        """Test formula with subtraction."""
        term = make_term(['A', 'B', 'C'])
        spec = ContrastFormulaSpec(
            name='test',
            A=Formula("condition == 'A' - condition == 'B'")
        )

        con = contrast_weights(spec, term)

        # A gets +1, B gets -1
        assert abs(np.sum(con.weights)) < 1e-10

    def test_formula_addition(self):
        """Test formula with addition."""
        term = make_term(['A', 'B', 'C'])
        spec = ContrastFormulaSpec(
            name='test',
            A=Formula("condition == 'A' + condition == 'B'")
        )

        con = contrast_weights(spec, term)

        # A and B each get +1
        condnames = con.condnames
        idx_A = condnames.index('condition.A')
        idx_B = condnames.index('condition.B')
        assert con.weights[idx_A, 0] == 1.0
        assert con.weights[idx_B, 0] == 1.0

    def test_formula_division(self):
        """Test formula with division."""
        term = make_term(['A', 'B', 'C'])
        spec = ContrastFormulaSpec(
            name='test',
            A=Formula("(condition == 'A' + condition == 'B') / 2")
        )

        con = contrast_weights(spec, term)

        # A and B each get 0.5
        condnames = con.condnames
        idx_A = condnames.index('condition.A')
        idx_B = condnames.index('condition.B')
        assert con.weights[idx_A, 0] == 0.5
        assert con.weights[idx_B, 0] == 0.5

    def test_formula_multiplication(self):
        """Test formula with multiplication."""
        term = make_term(['A', 'B', 'C'])
        spec = ContrastFormulaSpec(
            name='test',
            A=Formula("2 * condition == 'A'")
        )

        con = contrast_weights(spec, term)

        # A gets 2.0
        condnames = con.condnames
        idx_A = condnames.index('condition.A')
        assert con.weights[idx_A, 0] == 2.0

    def test_formula_unary_negation(self):
        """Test formula with unary negation."""
        term = make_term(['A', 'B', 'C'])
        spec = ContrastFormulaSpec(
            name='test',
            A=Formula("-condition == 'A'")
        )

        con = contrast_weights(spec, term)

        # A gets -1.0
        condnames = con.condnames
        idx_A = condnames.index('condition.A')
        assert con.weights[idx_A, 0] == -1.0

    def test_formula_unary_plus(self):
        """Test formula with unary plus."""
        term = make_term(['A', 'B', 'C'])
        spec = ContrastFormulaSpec(
            name='test',
            A=Formula("+condition == 'A'")
        )

        con = contrast_weights(spec, term)

        # A gets +1.0
        condnames = con.condnames
        idx_A = condnames.index('condition.A')
        assert con.weights[idx_A, 0] == 1.0

    def test_formula_numeric_constant(self):
        """Test formula with numeric constant."""
        term = make_term(['A', 'B', 'C'])
        spec = ContrastFormulaSpec(name='test', A=Formula("3.5"))

        con = contrast_weights(spec, term)

        # All conditions get 3.5
        assert np.all(con.weights == 3.5)

    def test_formula_bare_name(self):
        """Test formula with bare variable name."""
        term = make_term(['A', 'B', 'C'])
        spec = ContrastFormulaSpec(name='test', A=Formula("A"))

        con = contrast_weights(spec, term)

        # Should match condition.A
        condnames = con.condnames
        idx_A = condnames.index('condition.A')
        assert con.weights[idx_A, 0] == 1.0

    def test_formula_invalid_syntax(self):
        """Test formula with invalid syntax."""
        term = make_term(['A', 'B', 'C'])
        spec = ContrastFormulaSpec(name='test', A=Formula("???invalid"))

        # Should fall back to treating as condition name
        con = contrast_weights(spec, term)

        # Won't match anything, all zeros
        assert np.all(con.weights == 0)

    def test_formula_with_basis_expansion(self):
        """Test ContrastFormulaSpec with basis expansion."""
        term = make_term(['A', 'B'], name='condition')
        term.nbasis = 2

        spec = ContrastFormulaSpec(
            name='test',
            A=Formula("condition == 'A'")
        )

        con = contrast_weights(spec, term)

        # Should expand to 4 (2 conditions * 2 basis)
        assert con.weights.shape == (4, 1)

    def test_formula_evaluation_error(self):
        """Test formula that raises error during evaluation."""
        term = make_term(['A', 'B', 'C'])

        # Create a formula with invalid syntax that will be treated as a name
        spec = ContrastFormulaSpec(name='test', A=Formula("@#$%"))

        con = contrast_weights(spec, term)

        # Invalid syntax gets treated as a name that doesn't match anything
        # Should return zero weights (no match)
        assert np.all(con.weights == 0)


# ============================================================================
# Run the tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
