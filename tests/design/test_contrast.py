"""Tests for contrast functionality."""

import numpy as np
import pytest

from fmrimod.contrast import (
    column_contrast,
    contrast,
    contrast_set,
    contrast_weights,
    interaction_contrast,
    one_against_all_contrast,
    oneway_contrast,
    pair_contrast,
    pairwise_contrasts,
    poly_contrast,
    unit_contrast,
)
from fmrimod.contrast.contrast_spec import (
    ColumnContrastSpec,
    ContrastSet,
    ContrastSpec,
    InteractionContrastSpec,
    OnewayContrastSpec,
    PairContrastSpec,
    PolyContrastSpec,
    UnitContrastSpec,
)


class TestContrastSpecs:
    """Test contrast specification creation."""
    
    def test_contrast(self):
        """Test basic contrast creation."""
        # Import Formula from contrast_spec
        from fmrimod.contrast.contrast_spec import Formula
        
        c = contrast(Formula("A - B"), name="A_vs_B")
        
        assert isinstance(c, ContrastSpec)
        assert c.name == "A_vs_B"
        assert c.A.expr == "A - B"
        assert c.B is None
        assert c.where is None
    
    def test_unit_contrast(self):
        """Test unit contrast creation."""
        from fmrimod.contrast.contrast_spec import Formula
        
        c = unit_contrast(Formula("Face"), name="Main_face")
        
        assert isinstance(c, UnitContrastSpec)
        assert c.name == "Main_face"
        assert c.A.expr == "Face"
    
    def test_pair_contrast(self):
        """Test pair contrast creation."""
        from fmrimod.contrast.contrast_spec import Formula
        
        c = pair_contrast(
            Formula("category == 'face'"),
            Formula("category == 'scene'"),
            name="face_vs_scene"
        )
        
        assert isinstance(c, PairContrastSpec)
        assert c.name == "face_vs_scene"
        assert c.A.expr == "category == 'face'"
        assert c.B.expr == "category == 'scene'"
    
    def test_column_contrast(self):
        """Test column contrast creation."""
        c = column_contrast(
            pattern_A="^z_RT$",
            name="Main_RT"
        )
        
        assert isinstance(c, ColumnContrastSpec)
        assert c.name == "Main_RT"
        assert c.pattern_A == "^z_RT$"
        assert c.pattern_B is None
        
        # With B pattern
        c2 = column_contrast(
            pattern_A="^Cond\\.A_z_RT$",
            name="CondA_vs_CondB_RT",
            pattern_B="^Cond\\.B_z_RT$"
        )
        
        assert c2.pattern_B == "^Cond\\.B_z_RT$"
    
    def test_poly_contrast(self):
        """Test polynomial contrast creation."""
        from fmrimod.contrast.contrast_spec import Formula
        
        c = poly_contrast(
            Formula("time"),
            name="linear_time",
            degree=1
        )
        
        assert isinstance(c, PolyContrastSpec)
        assert c.name == "linear_time"
        assert c.degree == 1
        assert c.value_map is None
        
        # With value map
        c2 = poly_contrast(
            Formula("dose"),
            name="dose_cubic",
            degree=3,
            value_map={"low": 0, "med": 2, "high": 5}
        )
        
        assert c2.degree == 3
        assert c2.value_map == {"low": 0, "med": 2, "high": 5}
    
    def test_oneway_contrast(self):
        """Test one-way contrast creation."""
        from fmrimod.contrast.contrast_spec import Formula
        
        c = oneway_contrast(Formula("condition"), name="Main_condition")
        
        assert isinstance(c, OnewayContrastSpec)
        assert c.name == "Main_condition"
        assert c.A.expr == "condition"
    
    def test_interaction_contrast(self):
        """Test interaction contrast creation."""
        from fmrimod.contrast.contrast_spec import Formula
        
        c = interaction_contrast(
            Formula("condition * time"),
            name="condition_by_time"
        )
        
        assert isinstance(c, InteractionContrastSpec)
        assert c.name == "condition_by_time"
        assert c.A.expr == "condition * time"
    
    def test_contrast_set(self):
        """Test contrast set creation."""
        from fmrimod.contrast.contrast_spec import Formula
        
        c1 = contrast(Formula("A - B"), name="A_vs_B")
        c2 = contrast(Formula("B - C"), name="B_vs_C")
        
        cset = contrast_set(c1, c2)
        
        assert isinstance(cset, ContrastSet)
        assert len(cset) == 2
        assert cset[0].name == "A_vs_B"
        assert cset[1].name == "B_vs_C"
    
    def test_pairwise_contrasts(self):
        """Test pairwise contrast generation."""
        levels = ["A", "B", "C"]
        
        cset = pairwise_contrasts(levels, facname="condition")
        
        assert isinstance(cset, ContrastSet)
        assert len(cset) == 3  # 3 choose 2
        
        # Check names
        names = [c.name for c in cset]
        assert "con_A_B" in names
        assert "con_A_C" in names
        assert "con_B_C" in names
        
        # Check that they're pair contrasts
        for c in cset:
            assert isinstance(c, PairContrastSpec)
    
    def test_one_against_all_contrast(self):
        """Test one-against-all contrast generation."""
        levels = ["A", "B", "C"]
        
        cset = one_against_all_contrast(levels, facname="condition")
        
        assert isinstance(cset, ContrastSet)
        assert len(cset) == 3  # One for each level
        
        # Check names
        names = [c.name for c in cset]
        assert "con_A_vs_other" in names
        assert "con_B_vs_other" in names
        assert "con_C_vs_other" in names
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        from fmrimod.contrast.contrast_spec import Formula
        
        # Non-string name
        with pytest.raises(TypeError):
            contrast(Formula("A - B"), name=123)
        
        # Invalid degree
        with pytest.raises(ValueError):
            poly_contrast(Formula("time"), name="test", degree=0)
        
        # Too few levels for pairwise
        with pytest.raises(ValueError):
            pairwise_contrasts(["A"], facname="condition")


class TestContrastWeights:
    """Test contrast weight computation."""
    
    class MockTerm:
        """Mock term for testing."""
        def __init__(self, conditions=None, n_cells=0):
            self._conditions = conditions or []
            self._n_cells = n_cells
            self.varname = "test_term"
            self.nbasis = 1
        
        def conditions(self, drop_empty=False, expand_basis=True):
            return self._conditions
        
        def cells(self):
            # Return mock DataFrame-like object
            import pandas as pd
            if self._n_cells > 0:
                return pd.DataFrame({'col': range(self._n_cells)})
            return pd.DataFrame()
    
    def test_unit_contrast_weights(self):
        """Test unit contrast weight computation."""
        from fmrimod.contrast.contrast_spec import Formula
        
        term = self.MockTerm(conditions=["A", "B", "C"], n_cells=3)
        spec = unit_contrast(Formula("all"), name="test")
        
        con = contrast_weights(spec, term)
        
        assert con.name == "test"
        assert con.weights.shape == (3, 1)
        # Unit contrast should sum to 1
        assert np.isclose(np.sum(con.weights), 1.0)
        assert len(con.condnames) == 3
    
    def test_pair_contrast_weights(self):
        """Test pair contrast weight computation."""
        from fmrimod.contrast.contrast_spec import Formula
        
        term = self.MockTerm(conditions=["A", "B", "C", "D"], n_cells=4)
        spec = pair_contrast(Formula("A"), Formula("B"), name="A_vs_B")
        
        con = contrast_weights(spec, term)
        
        assert con.name == "A_vs_B"
        assert con.weights.shape == (4, 1)
        # Should sum to zero (approximately)
        assert np.abs(np.sum(con.weights)) < 1e-10
    
    def test_column_contrast_weights(self):
        """Test column contrast weight computation."""
        conditions = ["cond1_b1", "cond1_b2", "cond2_b1", "cond2_b2"]
        term = self.MockTerm(conditions=conditions)
        
        # Pattern matching b1
        spec = column_contrast(pattern_A="_b1$", name="basis1")
        con = contrast_weights(spec, term)
        
        assert con.name == "basis1"
        assert con.weights.shape == (4, 1)
        # Should have positive weights for b1 columns
        assert con.weights[0, 0] > 0  # cond1_b1
        assert con.weights[2, 0] > 0  # cond2_b1
        assert con.weights[1, 0] == 0  # cond1_b2
        assert con.weights[3, 0] == 0  # cond2_b2
        
        # Pattern A vs B
        spec2 = column_contrast(
            pattern_A="^cond1_",
            pattern_B="^cond2_",
            name="cond1_vs_cond2"
        )
        con2 = contrast_weights(spec2, term)
        
        assert np.sum(con2.weights) < 1e-10  # Sum to zero
        assert np.all(con2.weights[:2] > 0)  # cond1 positive
        assert np.all(con2.weights[2:] < 0)  # cond2 negative
    
    def test_poly_contrast_weights(self):
        """Test polynomial contrast weight computation."""
        from fmrimod.contrast.contrast_spec import Formula
        
        term = self.MockTerm(
            conditions=["time1", "time2", "time3", "time4"],
            n_cells=4
        )
        spec = poly_contrast(Formula("time"), name="linear", degree=1)
        
        con = contrast_weights(spec, term)
        
        assert con.name == "linear"
        assert con.weights.shape == (4, 1)  # 1 column for degree 1
        
        # Higher degree
        spec2 = poly_contrast(Formula("time"), name="cubic", degree=3)
        con2 = contrast_weights(spec2, term)
        
        assert con2.weights.shape == (4, 3)  # 3 columns for degree 3
    
    def test_oneway_contrast_weights(self):
        """Test one-way contrast weight computation."""
        from fmrimod.contrast.contrast_spec import Formula
        
        term = self.MockTerm(
            conditions=["A", "B", "C"],
            n_cells=3
        )
        spec = oneway_contrast(Formula("factor"), name="main_effect")
        
        con = contrast_weights(spec, term)
        
        assert con.name == "main_effect"
        # One-way with 3 levels should produce 2 contrasts
        assert con.weights.shape == (3, 2)
        # Each column should sum to zero
        for i in range(2):
            assert np.abs(np.sum(con.weights[:, i])) < 1e-10
    
    def test_contrast_set_weights(self):
        """Test weight computation for contrast sets."""
        from fmrimod.contrast.contrast_spec import Formula
        
        term = self.MockTerm(conditions=["A", "B", "C"])
        
        c1 = unit_contrast(Formula("A"), name="test1")
        c2 = unit_contrast(Formula("B"), name="test2")
        cset = contrast_set(c1, c2)
        
        results = contrast_weights(cset, term)
        
        assert isinstance(results, dict)
        assert len(results) == 2
        assert "test1" in results
        assert "test2" in results
        assert results["test1"].weights.shape == (3, 1)
        assert results["test2"].weights.shape == (3, 1)
    
    def test_empty_term(self):
        """Test handling of empty terms."""
        from fmrimod.contrast.contrast_spec import Formula
        
        term = self.MockTerm(conditions=[], n_cells=0)
        spec = unit_contrast(Formula("A"), name="test")
        
        with pytest.warns(UserWarning, match="No conditions found"):
            con = contrast_weights(spec, term)
        
        assert con.weights.shape == (0, 1)
        assert len(con.condnames) == 0
    
    def test_basis_expansion(self):
        """Test contrast weights with basis expansion."""
        from fmrimod.contrast.contrast_spec import Formula

        # Mock term with basis functions
        term = self.MockTerm(conditions=["A", "B"], n_cells=2)
        term.nbasis = 3  # 3 basis functions

        # Override conditions to return expanded names when requested
        def conditions_with_basis(drop_empty=False, expand_basis=True):
            base = ["A", "B"]
            if expand_basis and term.nbasis > 1:
                expanded = []
                for cond in base:
                    for b in range(term.nbasis):
                        expanded.append(f"{cond}_b{b+1}")
                return expanded
            return base

        term.conditions = conditions_with_basis

        spec = pair_contrast(Formula("A"), Formula("B"), name="A_vs_B")
        con = contrast_weights(spec, term)

        # Should have expanded to 6 conditions (2 * 3)
        assert con.weights.shape == (6, 1)
        assert len(con.condnames) == 6
        # Weights should still sum to zero
        assert np.abs(np.sum(con.weights)) < 1e-10


class TestContrastWeightsReal:
    """Test contrast weights with real condition names."""

    class RealMockTerm:
        """Mock term with real condition naming."""
        def __init__(self, conditions=None):
            self._conditions = conditions or []
            self.varname = "test_term"
            self.nbasis = 1

        def conditions(self, drop_empty=False, expand_basis=True):
            return self._conditions

        def cells(self):
            import pandas as pd
            return pd.DataFrame()

    def test_unit_contrast_real_conditions(self):
        """Test unit contrast with real condition names like 'condition.A'."""
        from fmrimod.contrast.contrast_spec import Formula

        # Condition names following naming.level_token format
        term = self.RealMockTerm(conditions=["condition.A", "condition.B", "condition.C"])
        spec = unit_contrast(Formula("condition == 'A'"), name="test_A")

        con = contrast_weights(spec, term)

        assert con.name == "test_A"
        assert con.weights.shape == (3, 1)
        # Only condition.A should have weight 1
        assert con.weights[0, 0] == 1.0
        assert con.weights[1, 0] == 0.0
        assert con.weights[2, 0] == 0.0

    def test_pair_contrast_real_conditions(self):
        """Test pair contrast with real condition names."""
        from fmrimod.contrast.contrast_spec import Formula

        term = self.RealMockTerm(conditions=["category.face", "category.scene", "category.object"])
        spec = pair_contrast(
            Formula("category == 'face'"),
            Formula("category == 'scene'"),
            name="face_vs_scene"
        )

        con = contrast_weights(spec, term)

        assert con.name == "face_vs_scene"
        assert con.weights.shape == (3, 1)
        # face gets +1, scene gets -1, object gets 0
        assert con.weights[0, 0] == 1.0
        assert con.weights[1, 0] == -1.0
        assert con.weights[2, 0] == 0.0
        # Should sum to zero
        assert np.abs(np.sum(con.weights)) < 1e-10

    def test_column_contrast_real_patterns(self):
        """Test column contrast with realistic regex patterns."""
        conditions = ["condition.A_b01", "condition.A_b02", "condition.B_b01", "condition.B_b02"]
        term = self.RealMockTerm(conditions=conditions)

        # Match all b01 columns
        spec = column_contrast(pattern_A="_b01$", name="basis1")
        con = contrast_weights(spec, term)

        assert con.name == "basis1"
        assert con.weights.shape == (4, 1)
        assert con.weights[0, 0] == 0.5  # condition.A_b01
        assert con.weights[1, 0] == 0.0  # condition.A_b02
        assert con.weights[2, 0] == 0.5  # condition.B_b01
        assert con.weights[3, 0] == 0.0  # condition.B_b02

    def test_poly_contrast_real(self):
        """Test polynomial contrast with multiple conditions."""
        from fmrimod.contrast.contrast_spec import Formula

        term = self.RealMockTerm(conditions=["time.1", "time.2", "time.3", "time.4"])
        spec = poly_contrast(Formula("time"), name="linear_time", degree=1)

        con = contrast_weights(spec, term)

        assert con.name == "linear_time"
        assert con.weights.shape == (4, 1)
        # Polynomial contrasts are orthonormal (columns should be orthogonal)
        # Check that the column is normalized
        assert np.abs(np.linalg.norm(con.weights[:, 0]) - 1.0) < 1e-10

        # Test quadratic
        spec2 = poly_contrast(Formula("time"), name="quad_time", degree=2)
        con2 = contrast_weights(spec2, term)

        assert con2.weights.shape == (4, 2)
        # Columns should be orthonormal
        for i in range(2):
            # Each column should be unit length
            assert np.abs(np.linalg.norm(con2.weights[:, i]) - 1.0) < 1e-10
        # Columns should be orthogonal to each other
        dot_product = np.dot(con2.weights[:, 0], con2.weights[:, 1])
        assert np.abs(dot_product) < 1e-10

    def test_oneway_contrast_real(self):
        """Test one-way contrast with real conditions."""
        from fmrimod.contrast.contrast_spec import Formula

        term = self.RealMockTerm(conditions=["cond.A", "cond.B", "cond.C"])
        spec = oneway_contrast(Formula("cond"), name="main_effect")

        con = contrast_weights(spec, term)

        assert con.name == "main_effect"
        # 3 levels -> 2 contrasts (k-1)
        assert con.weights.shape == (3, 2)
        # Each column should sum to zero
        for i in range(2):
            assert np.abs(np.sum(con.weights[:, i])) < 1e-10

    def test_interaction_contrast_2x2(self):
        """Test 2x2 interaction contrast."""
        from fmrimod.contrast.contrast_spec import Formula

        # 2x2 factorial: factor1 x factor2
        term = self.RealMockTerm(conditions=["f1.A_f2.X", "f1.A_f2.Y", "f1.B_f2.X", "f1.B_f2.Y"])
        spec = interaction_contrast(Formula("f1 * f2"), name="interaction")

        con = contrast_weights(spec, term)

        assert con.name == "interaction"
        assert con.weights.shape == (4, 1)
        # 2x2 interaction pattern: [1, -1, -1, 1]
        assert con.weights[0, 0] == 1.0
        assert con.weights[1, 0] == -1.0
        assert con.weights[2, 0] == -1.0
        assert con.weights[3, 0] == 1.0

    def test_pairwise_contrasts_real(self):
        """Test pairwise contrast helper function."""
        levels = ["A", "B", "C"]
        cset = pairwise_contrasts(levels, facname="condition")

        assert len(cset) == 3  # 3 choose 2
        assert all(isinstance(c, PairContrastSpec) for c in cset)

        # Test with a real term
        term = self.RealMockTerm(conditions=["condition.A", "condition.B", "condition.C"])

        for spec in cset:
            con = contrast_weights(spec, term)
            # Each should sum to zero
            assert np.abs(np.sum(con.weights)) < 1e-10

    def test_contrast_subtraction_operator(self):
        """Test ContrastSpec.__sub__ operator."""
        from fmrimod.contrast.contrast_spec import Formula

        c1 = unit_contrast(Formula("condition == 'A'"), name="A")
        c2 = unit_contrast(Formula("condition == 'B'"), name="B")

        # Use subtraction operator
        c_diff = c1 - c2

        assert isinstance(c_diff, PairContrastSpec)
        assert c_diff.name == "A-B"
        assert c_diff.A.expr == "condition == 'A'"
        assert c_diff.B.expr == "condition == 'B'"

        # Test with real term
        term = self.RealMockTerm(conditions=["condition.A", "condition.B", "condition.C"])
        con = contrast_weights(c_diff, term)

        # A gets +1, B gets -1, C gets 0
        assert con.weights[0, 0] == 1.0
        assert con.weights[1, 0] == -1.0
        assert con.weights[2, 0] == 0.0

    def test_contrast_set_computation(self):
        """Test computing weights for an entire contrast set."""
        from fmrimod.contrast.contrast_spec import Formula

        c1 = unit_contrast(Formula("condition == 'A'"), name="test_A")
        c2 = unit_contrast(Formula("condition == 'B'"), name="test_B")
        c3 = pair_contrast(
            Formula("condition == 'A'"),
            Formula("condition == 'B'"),
            name="A_vs_B"
        )

        cset = contrast_set(c1, c2, c3)

        term = self.RealMockTerm(conditions=["condition.A", "condition.B", "condition.C"])
        results = contrast_weights(cset, term)

        assert isinstance(results, dict)
        assert len(results) == 3
        assert "test_A" in results
        assert "test_B" in results
        assert "A_vs_B" in results

        # Check individual contrasts
        assert results["test_A"].weights[0, 0] == 1.0
        assert results["test_B"].weights[1, 0] == 1.0
        assert results["A_vs_B"].weights[0, 0] == 1.0
        assert results["A_vs_B"].weights[1, 0] == -1.0

    def test_poly_contrast_degree_validation(self):
        """Test that polynomial degree is validated."""
        from fmrimod.contrast.contrast_spec import Formula

        # Only 2 conditions - degree 2 should fail
        term = self.RealMockTerm(conditions=["time.1", "time.2"])
        spec = poly_contrast(Formula("time"), name="quad", degree=2)

        with pytest.raises(ValueError, match="too high"):
            contrast_weights(spec, term)

    def test_oneway_single_level(self):
        """Test one-way contrast with single level."""
        from fmrimod.contrast.contrast_spec import Formula

        term = self.RealMockTerm(conditions=["cond.A"])
        spec = oneway_contrast(Formula("cond"), name="main")

        with pytest.warns(UserWarning, match="only 1 level"):
            con = contrast_weights(spec, term)

        # Should return empty contrast matrix (no contrasts for 1 level)
        assert con.weights.shape == (1, 0)


def test_contrast_weights_singledispatch_extension_point():
    """Custom ContrastSpec subclasses can register their own ``contrast_weights``.

    fmrimod exposes two complementary extension seams for external specs:

    * :func:`fmrimod.contrast.contrast_mask` + :func:`contrast_from_mask` ports
      fmridesign's mask/packaging shape and is exercised by
      ``tests/test_contrast/test_contrast_mask_pipeline.py``.
    * :func:`contrast_weights` itself is a :func:`functools.singledispatch`
      generic, so specs that compute final weights directly (no mask/basis
      expansion needed) can register a handler. This test pins that path.
    """
    from fmrimod.contrast.contrast_spec import ContrastSpec
    from fmrimod.contrast.contrast_weights import Contrast

    class _CustomContrastSpec(ContrastSpec):
        def __init__(self, name: str = "custom_id_minus_neg"):
            super().__init__(name=name)

    class _MockTerm:
        def __init__(self, conditions):
            self._conditions = list(conditions)
            self.varname = "mock"
            self.nbasis = 1

        def conditions(self, drop_empty=False, expand_basis=True):
            return list(self._conditions)

        def cells(self):
            import pandas as pd
            return pd.DataFrame()

    term = _MockTerm(["cond.A", "cond.B", "cond.C"])
    spec = _CustomContrastSpec()

    with pytest.raises(NotImplementedError, match="_CustomContrastSpec"):
        contrast_weights(spec, term)

    @contrast_weights.register(_CustomContrastSpec)
    def _custom_handler(x, term, **kwargs):
        condnames = list(term.conditions())
        weights = np.array([[1.0], [0.0], [-1.0]])
        return Contrast(
            term=term,
            name=x.name,
            weights=weights,
            condnames=condnames,
            contrast_spec=x,
        )

    con = contrast_weights(spec, term)
    assert isinstance(con, Contrast)
    assert con.name == "custom_id_minus_neg"
    assert con.weights.shape == (3, 1)
    np.testing.assert_array_equal(con.weights, [[1.0], [0.0], [-1.0]])
    assert con.condnames == ["cond.A", "cond.B", "cond.C"]
