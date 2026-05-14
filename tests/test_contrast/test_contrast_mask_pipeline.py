"""fmridesign-style contrast mask extension pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.contrast import contrast_from_mask, contrast_mask
from fmrimod.contrast.contrast_spec import ColumnContrastSpec, ContrastSpec
from fmrimod.contrast.contrast_weights import Contrast


class DummyTerm:
    def __init__(self, nbasis: int = 1) -> None:
        self.nbasis = nbasis
        self._base = ["condition.A", "condition.B"]

    def conditions(
        self,
        drop_empty: bool = False,
        expand_basis: bool = True,
    ) -> list[str]:
        if expand_basis and self.nbasis > 1:
            return [
                f"{name}_b{basis_ix + 1}"
                for name in self._base
                for basis_ix in range(self.nbasis)
            ]
        return list(self._base)


def test_contrast_mask_refuses_unregistered_specs() -> None:
    spec = ContrastSpec(name="custom")

    with pytest.raises(NotImplementedError, match="No contrast_mask method"):
        contrast_mask(spec, DummyTerm())


def test_contrast_from_mask_packages_single_basis_mask() -> None:
    term = DummyTerm(nbasis=1)
    spec = ContrastSpec(name="A_minus_B")
    weights = np.array([[1.0], [-1.0]])

    out = contrast_from_mask(
        {"weights": weights, "condnames": term.conditions(expand_basis=False)},
        spec,
        term,
    )

    assert isinstance(out, Contrast)
    assert out.name == "A_minus_B"
    assert out.condnames == ["condition.A", "condition.B"]
    np.testing.assert_allclose(out.weights, weights)


def test_contrast_from_mask_expands_base_rows_across_basis_columns() -> None:
    term = DummyTerm(nbasis=3)
    spec = ContrastSpec(name="A_minus_B")
    weights = np.array([[1.0], [-1.0]])

    out = contrast_from_mask(
        {"weights": weights, "condnames": term.conditions(expand_basis=False)},
        spec,
        term,
    )

    assert out.condnames == [
        "condition.A_b1",
        "condition.A_b2",
        "condition.A_b3",
        "condition.B_b1",
        "condition.B_b2",
        "condition.B_b3",
    ]
    np.testing.assert_allclose(
        out.weights.ravel(),
        np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]),
    )


def test_contrast_from_mask_leaves_column_targeted_masks_expanded() -> None:
    term = DummyTerm(nbasis=3)
    spec = ColumnContrastSpec(pattern_A="_b2$", pattern_B=None, name="basis_2")
    condnames = term.conditions(expand_basis=True)
    weights = np.array([[0.0], [1.0], [0.0], [0.0], [-1.0], [0.0]])

    out = contrast_from_mask(
        {"weights": weights, "condnames": condnames},
        spec,
        term,
    )

    assert out.condnames == condnames
    np.testing.assert_allclose(out.weights, weights)
