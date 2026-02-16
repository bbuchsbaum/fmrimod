"""Comprehensive tests for the contrast system with real EventTerm objects.

This module tests the contrast specification and weight computation system
using actual EventFactor and EventTerm objects, not mocks.
"""

import pytest
import numpy as np
import pandas as pd

from fmrimod.events import EventFactor, EventVariable, EventTerm
from fmrimod.contrast import (
    contrast,
    unit_contrast,
    pair_contrast,
    column_contrast,
    poly_contrast,
    oneway_contrast,
    interaction_contrast,
    contrast_set,
    pairwise_contrasts,
    one_against_all_contrast,
    contrast_weights,
)
from fmrimod.contrast.contrast_spec import (
    ContrastSpec,
    UnitContrastSpec,
    PairContrastSpec,
    ColumnContrastSpec,
    PolyContrastSpec,
    OnewayContrastSpec,
    InteractionContrastSpec,
    ContrastFormulaSpec,
    ContrastSet,
    Formula,
    sliding_window_contrasts,
)


# ============================================================================
# Fixtures for creating real EventFactor and EventTerm objects
# ============================================================================

@pytest.fixture
def simple_factor_3_levels():
    """Create a simple 3-level categorical factor."""
    return EventFactor(
        name='condition',
        onsets=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        values=['A', 'B', 'C', 'A', 'B', 'C'],
        durations=1.0
    )


@pytest.fixture
def simple_term_3_levels(simple_factor_3_levels):
    """Create a simple EventTerm with 3 levels."""
    return EventTerm([simple_factor_3_levels])


@pytest.fixture
def factor_4_levels():
    """Create a 4-level categorical factor."""
    return EventFactor(
        name='time',
        onsets=np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]),
        values=['t1', 't2', 't3', 't4', 't1', 't2', 't3', 't4'],
        durations=1.0
    )


@pytest.fixture
def term_4_levels(factor_4_levels):
    """Create an EventTerm with 4 levels."""
    return EventTerm([factor_4_levels])


@pytest.fixture
def category_factor():
    """Create a category factor with face/scene/object levels."""
    return EventFactor(
        name='category',
        onsets=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        values=['face', 'scene', 'object', 'face', 'scene', 'object'],
        durations=0.5
    )


@pytest.fixture
def category_term(category_factor):
    """Create EventTerm from category factor."""
    return EventTerm([category_factor])


# ============================================================================
# Test ContrastSpec Construction
# ============================================================================

class TestContrastSpecConstruction:
    """Test creation of contrast specification objects."""

    def test_base_contrast_spec(self):
        """Test base ContrastSpec class."""
        spec = ContrastSpec(
            name="test",
            A=Formula("condition == 'A'"),
            B=Formula("condition == 'B'")
        )

        assert spec.name == "test"
        assert spec.A.expr == "condition == 'A'"
        assert spec.B.expr == "condition == 'B'"
        assert spec.where is None

    def test_unit_contrast_spec(self):
        """Test UnitContrastSpec construction."""
        spec = unit_contrast(Formula("all"), name="all_conditions")

        assert isinstance(spec, UnitContrastSpec)
        assert spec.name == "all_conditions"
        assert spec.A.expr == "all"
        assert spec.B is None

    def test_pair_contrast_spec(self):
        """Test PairContrastSpec construction."""
        spec = pair_contrast(
            Formula("condition == 'A'"),
            Formula("condition == 'B'"),
            name="A_vs_B"
        )

        assert isinstance(spec, PairContrastSpec)
        assert spec.name == "A_vs_B"
        assert spec.A.expr == "condition == 'A'"
        assert spec.B.expr == "condition == 'B'"

    def test_column_contrast_spec(self):
        """Test ColumnContrastSpec construction."""
        spec = column_contrast(
            pattern_A="^cond_A",
            pattern_B="^cond_B",
            name="A_vs_B_columns"
        )

        assert isinstance(spec, ColumnContrastSpec)
        assert spec.name == "A_vs_B_columns"
        assert spec.pattern_A == "^cond_A"
        assert spec.pattern_B == "^cond_B"

    def test_poly_contrast_spec(self):
        """Test PolyContrastSpec construction."""
        spec = poly_contrast(
            Formula("time"),
            name="linear_time",
            degree=2
        )

        assert isinstance(spec, PolyContrastSpec)
        assert spec.name == "linear_time"
        assert spec.degree == 2
        assert spec.A.expr == "time"

    def test_oneway_contrast_spec(self):
        """Test OnewayContrastSpec construction."""
        spec = oneway_contrast(Formula("condition"), name="main_effect")

        assert isinstance(spec, OnewayContrastSpec)
        assert spec.name == "main_effect"
        assert spec.A.expr == "condition"

    def test_interaction_contrast_spec(self):
        """Test InteractionContrastSpec construction."""
        spec = interaction_contrast(
            Formula("condition * block"),
            name="cond_by_block"
        )

        assert isinstance(spec, InteractionContrastSpec)
        assert spec.name == "cond_by_block"
        assert spec.A.expr == "condition * block"

    def test_contrast_set_construction(self):
        """Test ContrastSet construction and iteration."""
        c1 = unit_contrast(Formula("condition == 'A'"), name="A")
        c2 = unit_contrast(Formula("condition == 'B'"), name="B")
        c3 = pair_contrast(
            Formula("condition == 'A'"),
            Formula("condition == 'B'"),
            name="A_vs_B"
        )

        cset = contrast_set(c1, c2, c3)

        assert isinstance(cset, ContrastSet)
        assert len(cset) == 3
        assert cset[0].name == "A"
        assert cset[1].name == "B"
        assert cset[2].name == "A_vs_B"

        # Test iteration
        names = [c.name for c in cset]
        assert names == ["A", "B", "A_vs_B"]

    def test_contrast_set_type_validation(self):
        """Test that ContrastSet validates types."""
        with pytest.raises(TypeError):
            contrast_set("not a contrast spec")

    def test_pairwise_contrasts_generation(self):
        """Test pairwise_contrasts helper."""
        levels = ["A", "B", "C"]
        cset = pairwise_contrasts(levels, facname="condition")

        assert isinstance(cset, ContrastSet)
        assert len(cset) == 3  # 3 choose 2

        names = [c.name for c in cset]
        assert "con_A_B" in names
        assert "con_A_C" in names
        assert "con_B_C" in names

        # All should be PairContrastSpec
        for c in cset:
            assert isinstance(c, PairContrastSpec)

    def test_one_against_all_contrasts_generation(self):
        """Test one_against_all_contrast helper."""
        levels = ["A", "B", "C"]
        cset = one_against_all_contrast(levels, facname="condition")

        assert isinstance(cset, ContrastSet)
        assert len(cset) == 3  # One for each level

        names = [c.name for c in cset]
        assert "con_A_vs_other" in names
        assert "con_B_vs_other" in names
        assert "con_C_vs_other" in names

    def test_sliding_window_contrasts(self):
        """Test sliding window contrast generation."""
        levels = ["t1", "t2", "t3", "t4"]
        cset = sliding_window_contrasts(levels, facname="time", window_size=2)

        assert isinstance(cset, ContrastSet)
        # With window_size=2, we get: (t1,t2) vs (t3,t4)
        assert len(cset) == 1

        # Test smaller window
        cset2 = sliding_window_contrasts(levels, facname="time", window_size=1)
        # With window_size=1: t1 vs t2, t2 vs t3, t3 vs t4
        assert len(cset2) == 3

    def test_contrast_subtraction_operator(self):
        """Test ContrastSpec.__sub__ operator."""
        c1 = unit_contrast(Formula("condition == 'A'"), name="A")
        c2 = unit_contrast(Formula("condition == 'B'"), name="B")

        c_diff = c1 - c2

        assert isinstance(c_diff, PairContrastSpec)
        assert c_diff.name == "A-B"
        assert c_diff.A.expr == "condition == 'A'"
        assert c_diff.B.expr == "condition == 'B'"

    def test_input_validation(self):
        """Test input validation for contrast constructors."""
        # Non-Formula A
        with pytest.raises(TypeError):
            unit_contrast("not a formula", name="test")

        # Non-string name
        with pytest.raises(TypeError):
            unit_contrast(Formula("A"), name=123)

        # Invalid degree
        with pytest.raises(ValueError):
            poly_contrast(Formula("time"), name="test", degree=0)

        # Too few levels for pairwise
        with pytest.raises(ValueError):
            pairwise_contrasts(["A"], facname="condition")


# ============================================================================
# Test Contrast Weight Computation with Real EventTerms
# ============================================================================

class TestContrastWeightsReal:
    """Test contrast weight computation with real EventFactor/EventTerm objects."""

    def test_unit_contrast_all_conditions(self, simple_term_3_levels):
        """Test unit contrast selecting all conditions."""
        spec = unit_contrast(Formula("all"), name="all_cond")
        con = contrast_weights(spec, simple_term_3_levels)

        assert con.name == "all_cond"
        assert con.weights.shape == (3, 1)

        # Unit contrast should sum to 1
        assert np.isclose(np.sum(con.weights), 1.0)

        # Each condition should get equal weight
        expected = np.array([[1/3], [1/3], [1/3]])
        np.testing.assert_allclose(con.weights, expected, atol=1e-10)

        # Check condition names
        assert len(con.condnames) == 3
        assert set(con.condnames) == {'condition.A', 'condition.B', 'condition.C'}

    def test_unit_contrast_single_condition(self, simple_term_3_levels):
        """Test unit contrast selecting single condition."""
        spec = unit_contrast(Formula("condition == 'A'"), name="test_A")
        con = contrast_weights(spec, simple_term_3_levels)

        assert con.name == "test_A"
        assert con.weights.shape == (3, 1)

        # Only condition A should have weight 1
        condnames = con.condnames
        idx_A = condnames.index('condition.A')
        idx_B = condnames.index('condition.B')
        idx_C = condnames.index('condition.C')

        assert con.weights[idx_A, 0] == 1.0
        assert con.weights[idx_B, 0] == 0.0
        assert con.weights[idx_C, 0] == 0.0

        # Sum should still be 1
        assert np.isclose(np.sum(con.weights), 1.0)

    def test_pair_contrast_basic(self, simple_term_3_levels):
        """Test basic pairwise contrast."""
        spec = pair_contrast(
            Formula("condition == 'A'"),
            Formula("condition == 'B'"),
            name="A_vs_B"
        )
        con = contrast_weights(spec, simple_term_3_levels)

        assert con.name == "A_vs_B"
        assert con.weights.shape == (3, 1)

        # Should sum to zero
        assert np.abs(np.sum(con.weights)) < 1e-10

        # A gets +1, B gets -1, C gets 0
        condnames = con.condnames
        idx_A = condnames.index('condition.A')
        idx_B = condnames.index('condition.B')
        idx_C = condnames.index('condition.C')

        assert con.weights[idx_A, 0] == 1.0
        assert con.weights[idx_B, 0] == -1.0
        assert con.weights[idx_C, 0] == 0.0

    def test_pair_contrast_category_levels(self, category_term):
        """Test pair contrast with realistic category names."""
        spec = pair_contrast(
            Formula("category == 'face'"),
            Formula("category == 'scene'"),
            name="face_vs_scene"
        )
        con = contrast_weights(spec, category_term)

        assert con.name == "face_vs_scene"
        assert con.weights.shape == (3, 1)

        # Should sum to zero
        assert np.abs(np.sum(con.weights)) < 1e-10

        # face=+1, scene=-1, object=0
        condnames = con.condnames
        idx_face = condnames.index('category.face')
        idx_scene = condnames.index('category.scene')
        idx_object = condnames.index('category.object')

        assert con.weights[idx_face, 0] == 1.0
        assert con.weights[idx_scene, 0] == -1.0
        assert con.weights[idx_object, 0] == 0.0

    def test_column_contrast_regex_matching(self):
        """Test column contrast with regex pattern matching."""
        # Create term with named columns
        factor = EventFactor(
            name='condition',
            onsets=np.array([1.0, 2.0, 3.0, 4.0]),
            values=['A', 'B', 'A', 'B'],
            durations=1.0
        )
        term = EventTerm([factor])

        # Match columns containing 'A'
        spec = column_contrast(pattern_A="A", name="A_cols")
        con = contrast_weights(spec, term)

        assert con.name == "A_cols"
        # Should match condition.A
        assert con.weights.shape[0] == 2  # 2 conditions

        # Find index of condition.A
        condnames = con.condnames
        if 'condition.A' in condnames:
            idx_A = condnames.index('condition.A')
            idx_B = condnames.index('condition.B')
            assert con.weights[idx_A, 0] == 1.0  # A matched
            assert con.weights[idx_B, 0] == 0.0  # B not matched

    def test_column_contrast_A_vs_B(self):
        """Test column contrast with both A and B patterns."""
        factor = EventFactor(
            name='condition',
            onsets=np.array([1.0, 2.0, 3.0, 4.0]),
            values=['A', 'B', 'A', 'B'],
            durations=1.0
        )
        term = EventTerm([factor])

        spec = column_contrast(
            pattern_A="^condition\\.A",
            pattern_B="^condition\\.B",
            name="A_vs_B"
        )
        con = contrast_weights(spec, term)

        assert con.name == "A_vs_B"
        assert con.weights.shape[0] == 2

        # Should sum to zero
        assert np.abs(np.sum(con.weights)) < 1e-10

        condnames = con.condnames
        idx_A = condnames.index('condition.A')
        idx_B = condnames.index('condition.B')

        # A positive, B negative
        assert con.weights[idx_A, 0] > 0
        assert con.weights[idx_B, 0] < 0

    def test_poly_contrast_linear(self, term_4_levels):
        """Test linear polynomial contrast."""
        spec = poly_contrast(Formula("time"), name="linear_time", degree=1)
        con = contrast_weights(spec, term_4_levels)

        assert con.name == "linear_time"
        assert con.weights.shape == (4, 1)  # 4 levels, degree 1

        # Column should be normalized
        assert np.abs(np.linalg.norm(con.weights[:, 0]) - 1.0) < 1e-10

    def test_poly_contrast_quadratic(self, term_4_levels):
        """Test quadratic polynomial contrast."""
        spec = poly_contrast(Formula("time"), name="quad_time", degree=2)
        con = contrast_weights(spec, term_4_levels)

        assert con.name == "quad_time"
        assert con.weights.shape == (4, 2)  # 4 levels, degree 2 -> 2 columns

        # Each column should be normalized
        for i in range(2):
            norm = np.linalg.norm(con.weights[:, i])
            assert np.abs(norm - 1.0) < 1e-10

        # Columns should be orthogonal
        dot_product = np.dot(con.weights[:, 0], con.weights[:, 1])
        assert np.abs(dot_product) < 1e-10

    def test_poly_contrast_cubic(self, term_4_levels):
        """Test cubic polynomial contrast."""
        spec = poly_contrast(Formula("time"), name="cubic_time", degree=3)
        con = contrast_weights(spec, term_4_levels)

        assert con.name == "cubic_time"
        assert con.weights.shape == (4, 3)  # 4 levels, degree 3 -> 3 columns

        # All columns should be orthonormal
        for i in range(3):
            # Normalized
            assert np.abs(np.linalg.norm(con.weights[:, i]) - 1.0) < 1e-10

        # Pairwise orthogonal
        for i in range(3):
            for j in range(i+1, 3):
                dot = np.dot(con.weights[:, i], con.weights[:, j])
                assert np.abs(dot) < 1e-10

    def test_poly_contrast_degree_too_high(self, simple_term_3_levels):
        """Test that polynomial degree validation works."""
        # 3 levels, degree 3 should fail (need at least degree+1 levels)
        spec = poly_contrast(Formula("condition"), name="too_high", degree=3)

        with pytest.raises(ValueError, match="too high"):
            contrast_weights(spec, simple_term_3_levels)

    def test_oneway_contrast_3_levels(self, simple_term_3_levels):
        """Test one-way contrast with 3 levels (Helmert coding)."""
        spec = oneway_contrast(Formula("condition"), name="main_effect")
        con = contrast_weights(spec, simple_term_3_levels)

        assert con.name == "main_effect"
        # 3 levels -> 2 contrasts (k-1)
        assert con.weights.shape == (3, 2)

        # Each contrast should sum to zero
        for i in range(2):
            assert np.abs(np.sum(con.weights[:, i])) < 1e-10

    def test_oneway_contrast_4_levels(self, term_4_levels):
        """Test one-way contrast with 4 levels."""
        spec = oneway_contrast(Formula("time"), name="time_main")
        con = contrast_weights(spec, term_4_levels)

        assert con.name == "time_main"
        # 4 levels -> 3 contrasts
        assert con.weights.shape == (4, 3)

        # Each contrast should sum to zero
        for i in range(3):
            col_sum = np.sum(con.weights[:, i])
            assert np.abs(col_sum) < 1e-10

    def test_oneway_contrast_single_level(self):
        """Test one-way contrast with only 1 level."""
        factor = EventFactor(
            name='condition',
            onsets=np.array([1.0, 2.0]),
            values=['A', 'A'],
            durations=1.0
        )
        term = EventTerm([factor])

        spec = oneway_contrast(Formula("condition"), name="single_level")
        with pytest.warns(UserWarning, match="only 1 level"):
            con = contrast_weights(spec, term)

        # Should return empty contrast matrix (no contrasts possible)
        assert con.weights.shape == (1, 0)

    def test_interaction_contrast_2x2(self):
        """Test 2x2 interaction contrast."""
        # Create 2x2 factorial design
        factor = EventFactor(
            name='design',
            onsets=np.array([1.0, 2.0, 3.0, 4.0]),
            values=['A1', 'A2', 'B1', 'B2'],  # Simulating 2x2
            durations=1.0
        )
        term = EventTerm([factor])

        spec = interaction_contrast(Formula("design"), name="interaction_2x2")
        con = contrast_weights(spec, term)

        assert con.name == "interaction_2x2"
        assert con.weights.shape == (4, 1)

        # 2x2 interaction pattern: [1, -1, -1, 1]
        assert con.weights[0, 0] == 1.0
        assert con.weights[1, 0] == -1.0
        assert con.weights[2, 0] == -1.0
        assert con.weights[3, 0] == 1.0

    def test_contrast_set_weights_computation(self, simple_term_3_levels):
        """Test computing weights for entire contrast set."""
        c1 = unit_contrast(Formula("condition == 'A'"), name="test_A")
        c2 = unit_contrast(Formula("condition == 'B'"), name="test_B")
        c3 = pair_contrast(
            Formula("condition == 'A'"),
            Formula("condition == 'B'"),
            name="A_vs_B"
        )

        cset = contrast_set(c1, c2, c3)
        results = contrast_weights(cset, simple_term_3_levels)

        assert isinstance(results, dict)
        assert len(results) == 3
        assert "test_A" in results
        assert "test_B" in results
        assert "A_vs_B" in results

        # Check individual contrasts
        condnames = results["test_A"].condnames
        idx_A = condnames.index('condition.A')
        idx_B = condnames.index('condition.B')

        assert results["test_A"].weights[idx_A, 0] == 1.0
        assert results["test_B"].weights[idx_B, 0] == 1.0
        assert results["A_vs_B"].weights[idx_A, 0] == 1.0
        assert results["A_vs_B"].weights[idx_B, 0] == -1.0

    def test_pairwise_contrasts_with_real_term(self, simple_term_3_levels):
        """Test pairwise contrasts applied to real term."""
        levels = ["A", "B", "C"]
        cset = pairwise_contrasts(levels, facname="condition")

        # Compute all contrasts
        for spec in cset:
            con = contrast_weights(spec, simple_term_3_levels)
            # Each pairwise contrast should sum to zero
            assert np.abs(np.sum(con.weights)) < 1e-10

    def test_one_against_all_with_real_term(self, simple_term_3_levels):
        """Test one-against-all contrasts applied to real term.

        Note: Currently the != operator in formulas is not implemented,
        so one_against_all_contrast creates contrasts that only select
        the positive condition (not a proper one-vs-all). This is a known
        limitation - the contrasts won't sum to zero.
        """
        levels = ["A", "B", "C"]
        cset = one_against_all_contrast(levels, facname="condition")

        # Currently these act as unit contrasts (selecting single condition)
        # because the != formula doesn't match anything
        for spec in cset:
            with pytest.warns(UserWarning, match="Mask B is empty"):
                con = contrast_weights(spec, simple_term_3_levels)
            # Each contrast selects one condition with weight 1
            # (not a true one-vs-all because != is not implemented)
            assert con.weights.shape == (3, 1)
            # Sum is 1, not 0 (unit contrast behavior)
            assert np.abs(np.sum(con.weights) - 1.0) < 1e-10

    def test_contrast_is_fcontrast_property(self, term_4_levels):
        """Test Contrast.is_fcontrast property."""
        # t-contrast (single column)
        spec_t = pair_contrast(
            Formula("time == 't1'"),
            Formula("time == 't2'"),
            name="t1_vs_t2"
        )
        con_t = contrast_weights(spec_t, term_4_levels)
        assert not con_t.is_fcontrast

        # F-contrast (multiple columns)
        spec_f = poly_contrast(Formula("time"), name="poly", degree=2)
        con_f = contrast_weights(spec_f, term_4_levels)
        assert con_f.is_fcontrast


# ============================================================================
# Test Edge Cases and Error Handling
# ============================================================================

class TestContrastEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_term_handling(self):
        """Test handling of terms with no matching conditions."""
        # Create term with valid events but test non-matching formula
        factor = EventFactor(
            name='condition',
            onsets=np.array([1.0, 2.0]),
            values=['A', 'B'],
            durations=1.0
        )
        term = EventTerm([factor])

        # Use formula that doesn't match any conditions
        spec = unit_contrast(Formula("nonexistent == 'Z'"), name="test")
        with pytest.warns(UserWarning, match="No conditions were selected"):
            con = contrast_weights(spec, term)

        # Should handle gracefully - creates zero weights
        assert con.weights.shape[0] == 2  # Still has 2 conditions
        assert np.all(con.weights == 0)  # But all weights are zero

    def test_no_matching_conditions_unit_contrast(self, simple_term_3_levels):
        """Test unit contrast when no conditions match."""
        spec = unit_contrast(Formula("condition == 'Z'"), name="nonexistent")
        with pytest.warns(UserWarning, match="No conditions were selected"):
            con = contrast_weights(spec, simple_term_3_levels)

        # Should create zero weights
        assert con.weights.shape == (3, 1)
        assert np.all(con.weights == 0)

    def test_contrast_repr(self, simple_term_3_levels):
        """Test Contrast.__repr__ method."""
        spec = pair_contrast(
            Formula("condition == 'A'"),
            Formula("condition == 'B'"),
            name="A_vs_B"
        )
        con = contrast_weights(spec, simple_term_3_levels)

        repr_str = repr(con)
        assert "Contrast" in repr_str
        assert "A_vs_B" in repr_str
        assert "t-contrast" in repr_str or "F-contrast" in repr_str

    def test_contrast_set_repr(self):
        """Test ContrastSet.__repr__ method."""
        c1 = unit_contrast(Formula("condition == 'A'"), name="A")
        c2 = unit_contrast(Formula("condition == 'B'"), name="B")
        cset = contrast_set(c1, c2)

        repr_str = repr(cset)
        assert "ContrastSet" in repr_str
        assert "n=2" in repr_str
        assert "A" in repr_str
        assert "B" in repr_str

    def test_contrast_set_empty_repr(self):
        """Test empty ContrastSet repr."""
        cset = ContrastSet()
        repr_str = repr(cset)
        assert "empty" in repr_str

    def test_formula_repr(self):
        """Test Formula.__repr__ and __str__."""
        f = Formula("condition == 'A'")
        assert str(f) == "condition == 'A'"
        assert "Formula" in repr(f)
        assert "condition == 'A'" in repr(f)


# ============================================================================
# Test Integration Scenarios
# ============================================================================

class TestContrastIntegration:
    """Test realistic integration scenarios."""

    def test_full_factorial_design_contrasts(self):
        """Test comprehensive contrast set for factorial design."""
        # Create 3-level factor
        factor = EventFactor(
            name='stimulus',
            onsets=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            values=['face', 'house', 'object', 'face', 'house', 'object'],
            durations=1.0
        )
        term = EventTerm([factor])

        # Create comprehensive contrast set
        c1 = unit_contrast(Formula("all"), name="overall_mean")
        c2 = oneway_contrast(Formula("stimulus"), name="main_effect")
        c3 = pair_contrast(
            Formula("stimulus == 'face'"),
            Formula("stimulus == 'house'"),
            name="face_vs_house"
        )

        cset = contrast_set(c1, c2, c3)
        results = contrast_weights(cset, term)

        assert len(results) == 3
        assert results["overall_mean"].weights.shape == (3, 1)
        assert results["main_effect"].weights.shape == (3, 2)  # 2 contrasts for 3 levels
        assert results["face_vs_house"].weights.shape == (3, 1)

    def test_polynomial_trend_analysis(self, term_4_levels):
        """Test polynomial trend analysis with multiple degrees."""
        # Linear, quadratic, and cubic trends
        c_lin = poly_contrast(Formula("time"), name="linear", degree=1)
        c_quad = poly_contrast(Formula("time"), name="quadratic", degree=2)
        c_cubic = poly_contrast(Formula("time"), name="cubic", degree=3)

        cset = contrast_set(c_lin, c_quad, c_cubic)
        results = contrast_weights(cset, term_4_levels)

        # Linear: 1 column
        assert results["linear"].weights.shape == (4, 1)
        # Quadratic: 2 columns
        assert results["quadratic"].weights.shape == (4, 2)
        # Cubic: 3 columns
        assert results["cubic"].weights.shape == (4, 3)

    def test_contrast_subtraction_workflow(self, category_term):
        """Test using subtraction operator in workflow."""
        face = unit_contrast(Formula("category == 'face'"), name="face")
        scene = unit_contrast(Formula("category == 'scene'"), name="scene")
        object_ = unit_contrast(Formula("category == 'object'"), name="object")

        # Create pairwise contrasts using subtraction
        face_vs_scene = face - scene
        face_vs_object = face - object_
        scene_vs_object = scene - object_

        # Test they work
        con1 = contrast_weights(face_vs_scene, category_term)
        con2 = contrast_weights(face_vs_object, category_term)
        con3 = contrast_weights(scene_vs_object, category_term)

        # All should sum to zero
        assert np.abs(np.sum(con1.weights)) < 1e-10
        assert np.abs(np.sum(con2.weights)) < 1e-10
        assert np.abs(np.sum(con3.weights)) < 1e-10

    def test_all_pairwise_comparison_workflow(self, simple_term_3_levels):
        """Test complete pairwise comparison workflow."""
        levels = ["A", "B", "C"]
        cset = pairwise_contrasts(levels, facname="condition")

        results = contrast_weights(cset, simple_term_3_levels)

        # Should have 3 contrasts
        assert len(results) == 3

        # All should be sum-to-zero
        for name, con in results.items():
            assert np.abs(np.sum(con.weights)) < 1e-10
            assert con.weights.shape == (3, 1)


# ============================================================================
# Test ContrastFormulaSpec (Formula-based contrasts)
# ============================================================================

class TestContrastFormulaSpec:
    """Test ContrastFormulaSpec with formula parsing."""

    def test_simple_condition_selection(self, simple_term_3_levels):
        """Test simple condition selection: condition == 'A'"""
        spec = ContrastFormulaSpec(
            name="test_A",
            A=Formula("condition == 'A'")
        )
        con = contrast_weights(spec, simple_term_3_levels)

        assert con.name == "test_A"
        assert con.weights.shape == (3, 1)

        # Only condition A should have weight 1
        condnames = con.condnames
        idx_A = condnames.index('condition.A')
        idx_B = condnames.index('condition.B')
        idx_C = condnames.index('condition.C')

        expected = np.zeros((3, 1))
        expected[idx_A, 0] = 1.0
        np.testing.assert_allclose(con.weights, expected, atol=1e-10)

    def test_difference_formula(self, simple_term_3_levels):
        """Test difference: condition == 'A' - condition == 'B'"""
        spec = ContrastFormulaSpec(
            name="A_minus_B",
            A=Formula("condition == 'A' - condition == 'B'")
        )
        con = contrast_weights(spec, simple_term_3_levels)

        assert con.name == "A_minus_B"
        assert con.weights.shape == (3, 1)

        # Should sum to zero
        assert np.abs(np.sum(con.weights)) < 1e-10

        # A gets +1, B gets -1, C gets 0
        condnames = con.condnames
        idx_A = condnames.index('condition.A')
        idx_B = condnames.index('condition.B')
        idx_C = condnames.index('condition.C')

        expected = np.zeros((3, 1))
        expected[idx_A, 0] = 1.0
        expected[idx_B, 0] = -1.0
        expected[idx_C, 0] = 0.0
        np.testing.assert_allclose(con.weights, expected, atol=1e-10)

    def test_average_formula(self, simple_term_3_levels):
        """Test average: (condition == 'A' + condition == 'B') / 2"""
        spec = ContrastFormulaSpec(
            name="avg_AB",
            A=Formula("(condition == 'A' + condition == 'B') / 2")
        )
        con = contrast_weights(spec, simple_term_3_levels)

        assert con.name == "avg_AB"
        assert con.weights.shape == (3, 1)

        # A and B each get 0.5, C gets 0
        condnames = con.condnames
        idx_A = condnames.index('condition.A')
        idx_B = condnames.index('condition.B')
        idx_C = condnames.index('condition.C')

        expected = np.zeros((3, 1))
        expected[idx_A, 0] = 0.5
        expected[idx_B, 0] = 0.5
        expected[idx_C, 0] = 0.0
        np.testing.assert_allclose(con.weights, expected, atol=1e-10)

    def test_weighted_diff_formula(self, simple_term_3_levels):
        """Test weighted diff: (condition == 'A' + condition == 'B') / 2 - condition == 'C'"""
        spec = ContrastFormulaSpec(
            name="avg_AB_minus_C",
            A=Formula("(condition == 'A' + condition == 'B') / 2 - condition == 'C'")
        )
        con = contrast_weights(spec, simple_term_3_levels)

        assert con.name == "avg_AB_minus_C"
        assert con.weights.shape == (3, 1)

        # Should sum to zero
        assert np.abs(np.sum(con.weights)) < 1e-10

        # A and B each get 0.5, C gets -1
        condnames = con.condnames
        idx_A = condnames.index('condition.A')
        idx_B = condnames.index('condition.B')
        idx_C = condnames.index('condition.C')

        expected = np.zeros((3, 1))
        expected[idx_A, 0] = 0.5
        expected[idx_B, 0] = 0.5
        expected[idx_C, 0] = -1.0
        np.testing.assert_allclose(con.weights, expected, atol=1e-10)

    def test_simple_condition_name(self, simple_term_3_levels):
        """Test simple condition name (no ==): just 'A'"""
        spec = ContrastFormulaSpec(
            name="test_A_simple",
            A=Formula("A")
        )
        con = contrast_weights(spec, simple_term_3_levels)

        assert con.name == "test_A_simple"
        assert con.weights.shape == (3, 1)

        # Should match condition.A
        condnames = con.condnames
        idx_A = condnames.index('condition.A')

        # A gets weight 1, others get 0
        expected = np.zeros((3, 1))
        expected[idx_A, 0] = 1.0
        np.testing.assert_allclose(con.weights, expected, atol=1e-10)

    def test_multiplication_formula(self, simple_term_3_levels):
        """Test multiplication: 2 * condition == 'A'"""
        spec = ContrastFormulaSpec(
            name="two_times_A",
            A=Formula("2 * condition == 'A'")
        )
        con = contrast_weights(spec, simple_term_3_levels)

        assert con.name == "two_times_A"
        assert con.weights.shape == (3, 1)

        # A gets weight 2, others get 0
        condnames = con.condnames
        idx_A = condnames.index('condition.A')

        expected = np.zeros((3, 1))
        expected[idx_A, 0] = 2.0
        np.testing.assert_allclose(con.weights, expected, atol=1e-10)

    def test_complex_formula(self, simple_term_3_levels):
        """Test complex formula: condition == 'A' - (condition == 'B' + condition == 'C') / 2"""
        spec = ContrastFormulaSpec(
            name="A_vs_BC_avg",
            A=Formula("condition == 'A' - (condition == 'B' + condition == 'C') / 2")
        )
        con = contrast_weights(spec, simple_term_3_levels)

        assert con.name == "A_vs_BC_avg"
        assert con.weights.shape == (3, 1)

        # Should sum to zero
        assert np.abs(np.sum(con.weights)) < 1e-10

        # A gets 1, B and C each get -0.5
        condnames = con.condnames
        idx_A = condnames.index('condition.A')
        idx_B = condnames.index('condition.B')
        idx_C = condnames.index('condition.C')

        expected = np.zeros((3, 1))
        expected[idx_A, 0] = 1.0
        expected[idx_B, 0] = -0.5
        expected[idx_C, 0] = -0.5
        np.testing.assert_allclose(con.weights, expected, atol=1e-10)

    def test_contrast_function_wrapper(self, simple_term_3_levels):
        """Test using the contrast() function with formula."""
        spec = contrast(
            Formula("condition == 'A' - condition == 'B'"),
            name="A_vs_B"
        )
        con = contrast_weights(spec, simple_term_3_levels)

        assert con.name == "A_vs_B"
        assert con.weights.shape == (3, 1)

        # Should sum to zero
        assert np.abs(np.sum(con.weights)) < 1e-10

        # A gets +1, B gets -1
        condnames = con.condnames
        idx_A = condnames.index('condition.A')
        idx_B = condnames.index('condition.B')

        assert con.weights[idx_A, 0] == 1.0
        assert con.weights[idx_B, 0] == -1.0


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
