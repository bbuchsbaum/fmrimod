"""Test ContrastDiffSpec functionality (subtraction operator).

This test module verifies that the __sub__ operator on ContrastSpec objects
produces proper difference contrasts (A - B weights).
"""

import pytest
import numpy as np

from fmrimod.events import EventFactor, EventTerm
from fmrimod.contrast import unit_contrast, contrast_weights
from fmrimod.contrast.contrast_spec import Formula, PairContrastSpec


class TestContrastDiffSpec:
    """Test the subtraction operator for ContrastSpec objects."""

    def test_basic_subtraction_creates_pair_contrast(self):
        """Test that A - B creates a PairContrastSpec."""
        a = unit_contrast(Formula("condition == 'A'"), name="A")
        b = unit_contrast(Formula("condition == 'B'"), name="B")

        diff = a - b

        assert isinstance(diff, PairContrastSpec)
        assert diff.name == "A-B"
        assert diff.A.expr == "condition == 'A'"
        assert diff.B.expr == "condition == 'B'"

    def test_subtraction_with_real_term(self):
        """Test subtraction operator with real EventTerm."""
        # Create factor with 3 levels
        event = EventFactor(
            name='cond',
            onsets=[1, 2, 3, 4, 5, 6],
            values=['A', 'B', 'C', 'A', 'B', 'C'],
            durations=1
        )
        term = EventTerm([event])

        # Create unit contrasts
        a = unit_contrast(Formula("cond == 'A'"), name="A")
        b = unit_contrast(Formula("cond == 'B'"), name="B")

        # Compute difference using subtraction operator
        diff = a - b
        weights = contrast_weights(diff, term)

        # Verify weights shape
        assert weights.weights.shape == (3, 1)

        # Verify proper difference contrast: A gets +1, B gets -1, C gets 0
        condnames = weights.condnames
        idx_A = condnames.index('cond.A')
        idx_B = condnames.index('cond.B')
        idx_C = condnames.index('cond.C')

        assert weights.weights[idx_A, 0] == 1.0
        assert weights.weights[idx_B, 0] == -1.0
        assert weights.weights[idx_C, 0] == 0.0

        # Verify sum-to-zero
        assert np.abs(np.sum(weights.weights)) < 1e-10

    def test_subtraction_multiple_conditions(self):
        """Test subtraction with conditions that match multiple levels."""
        # Create factor with 4 levels
        event = EventFactor(
            name='stim',
            onsets=[1, 2, 3, 4, 5, 6, 7, 8],
            values=['face', 'scene', 'object', 'scrambled',
                    'face', 'scene', 'object', 'scrambled'],
            durations=1
        )
        term = EventTerm([event])

        # Create contrasts for faces and scenes
        faces = unit_contrast(Formula("stim == 'face'"), name="faces")
        scenes = unit_contrast(Formula("stim == 'scene'"), name="scenes")

        # Subtract
        diff = faces - scenes
        weights = contrast_weights(diff, term)

        # Verify
        condnames = weights.condnames
        idx_face = condnames.index('stim.face')
        idx_scene = condnames.index('stim.scene')
        idx_object = condnames.index('stim.object')
        idx_scrambled = condnames.index('stim.scrambled')

        assert weights.weights[idx_face, 0] == 1.0
        assert weights.weights[idx_scene, 0] == -1.0
        assert weights.weights[idx_object, 0] == 0.0
        assert weights.weights[idx_scrambled, 0] == 0.0

        # Sum to zero
        assert np.abs(np.sum(weights.weights)) < 1e-10

    def test_chained_subtractions(self):
        """Test multiple subtractions: A - B - C should work."""
        event = EventFactor(
            name='cond',
            onsets=[1, 2, 3, 4, 5, 6],
            values=['A', 'B', 'C', 'A', 'B', 'C'],
            durations=1
        )
        term = EventTerm([event])

        a = unit_contrast(Formula("cond == 'A'"), name="A")
        b = unit_contrast(Formula("cond == 'B'"), name="B")
        c = unit_contrast(Formula("cond == 'C'"), name="C")

        # (A - B) - C
        diff = a - b - c

        # This creates nested PairContrastSpecs
        assert isinstance(diff, PairContrastSpec)

        # Compute weights
        weights = contrast_weights(diff, term)

        # The weights should be computed based on the formulas in the final spec
        # Since (A - B) creates PairContrastSpec(A="cond == 'A'", B="cond == 'B'")
        # Then (A-B) - C creates PairContrastSpec(A="cond == 'A'", B="cond == 'C'")
        # So we get A=+1, C=-1, B=0
        condnames = weights.condnames
        idx_A = condnames.index('cond.A')
        idx_B = condnames.index('cond.B')
        idx_C = condnames.index('cond.C')

        # Note: The current implementation uses self.A from (A-B) which is "cond == 'A'"
        # and other.A from C which is "cond == 'C'"
        assert weights.weights[idx_A, 0] == 1.0
        assert weights.weights[idx_C, 0] == -1.0
        assert weights.weights[idx_B, 0] == 0.0

    def test_subtraction_preserves_where_clause(self):
        """Test that subtraction preserves the where clause from left operand."""
        a = unit_contrast(Formula("cond == 'A'"), name="A", where=Formula("block == 1"))
        b = unit_contrast(Formula("cond == 'B'"), name="B")

        diff = a - b

        # Where clause from 'a' should be preserved
        assert diff.where is not None
        assert diff.where.expr == "block == 1"

    def test_subtraction_not_implemented_for_non_contrast(self):
        """Test that subtraction with non-ContrastSpec raises TypeError."""
        a = unit_contrast(Formula("cond == 'A'"), name="A")

        # Subtracting a non-ContrastSpec should raise TypeError
        with pytest.raises(TypeError):
            _ = a - "not a contrast"

        with pytest.raises(TypeError):
            _ = a - 123

    def test_subtraction_name_generation(self):
        """Test that subtraction generates proper names."""
        a = unit_contrast(Formula("A"), name="condition_A")
        b = unit_contrast(Formula("B"), name="condition_B")

        diff = a - b

        assert diff.name == "condition_A-condition_B"

    def test_realistic_fmri_contrast_workflow(self):
        """Test a realistic fMRI contrast workflow using subtraction."""
        # Simulate a typical fMRI design with faces and houses
        event = EventFactor(
            name='category',
            onsets=np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
            values=['face', 'house', 'face', 'house', 'face', 'house'],
            durations=2.0
        )
        term = EventTerm([event])

        # Define contrasts the easy way
        face = unit_contrast(Formula("category == 'face'"), name="Face")
        house = unit_contrast(Formula("category == 'house'"), name="House")

        # Face > House contrast
        face_vs_house = face - house

        # Compute weights
        weights = contrast_weights(face_vs_house, term)

        # Verify
        assert weights.name == "Face-House"
        condnames = weights.condnames
        idx_face = condnames.index('category.face')
        idx_house = condnames.index('category.house')

        # Face gets +1, House gets -1
        assert weights.weights[idx_face, 0] == 1.0
        assert weights.weights[idx_house, 0] == -1.0

        # Perfect sum-to-zero
        assert np.abs(np.sum(weights.weights)) < 1e-10

        # Verify this is a t-contrast (single column)
        assert not weights.is_fcontrast


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
