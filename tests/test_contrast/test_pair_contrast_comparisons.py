"""Comparison operators in pair-contrast formulas select complementary levels.

R fmridesign evaluates each side of a ``pair_contrast`` on the term's cells
(``rlang::eval_tidy(f_rhs(A), data = cells)``), so ``~ condition != "A"``
selects every other level and ``one_against_all_contrast`` gives level
``i`` weight 1 and each of the other ``k - 1`` levels ``-1/(k - 1)``
(R ``tests/testthat/test-contrast-set-nesting.R``: ``c(1, -0.5, -0.5)``).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from fmrimod import event_model
from fmrimod.contrast import (
    contrast_weights,
    one_against_all_contrast,
    pair_contrast,
)
from fmrimod.contrast.contrast_spec import Formula
from fmrimod.contrast.contrast_weights import _match_conditions_to_formula
from fmrimod.formula.base import Term


def _model_weights(levels, cset):
    ev = pd.DataFrame(
        {"onset": np.arange(0.0, 10.0 * 2 * len(levels), 10.0), "condition": levels * 2}
    )
    term = Term("condition")
    term.contrast_specs = cset
    em = event_model([term], data=ev, tr=1.0, n_scans=10 * 2 * len(levels))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return contrast_weights(em)["condition"]


@pytest.mark.parametrize("levels", [["A", "B", "C"], ["A", "B", "C", "D"]])
def test_one_against_all_matches_r_weights(levels):
    k = len(levels)
    weights = _model_weights(levels, one_against_all_contrast(levels, "condition"))
    assert list(weights) == [f"con_{lev}_vs_other" for lev in levels]
    for i, lev in enumerate(levels):
        expected = np.full(k, -1.0 / (k - 1))
        expected[i] = 1.0
        np.testing.assert_allclose(
            weights[f"con_{lev}_vs_other"].offset_weights[:, 0], expected
        )


def test_pair_contrast_with_not_equal_formula():
    spec = pair_contrast(
        Formula("condition == 'B'"), Formula("condition != 'B'"), name="B_vs_rest"
    )
    weights = _model_weights(["A", "B", "C", "D"], [spec])
    np.testing.assert_allclose(
        weights["B_vs_rest"].offset_weights[:, 0], [-1 / 3, 1, -1 / 3, -1 / 3]
    )


def test_pair_contrast_with_membership_formula():
    # R: pair_contrast(~ condition %in% c("A", "B"), ~ condition %in% c("C", "D"))
    spec = pair_contrast(
        Formula("condition in ['A', 'B']"),
        Formula("condition not in ['A', 'B']"),
        name="AB_vs_CD",
    )
    weights = _model_weights(["A", "B", "C", "D"], [spec])
    np.testing.assert_allclose(
        weights["AB_vs_CD"].offset_weights[:, 0], [0.5, 0.5, -0.5, -0.5]
    )


def test_matcher_not_equal_is_complement_within_the_factor():
    condnames = ["condition.A", "condition.B", "condition.C", "rt"]
    mask = _match_conditions_to_formula(condnames, "condition != 'A'")
    assert mask.tolist() == [False, True, True, False]
    nested = ["cond.A_task.x", "cond.A_task.y", "cond.B_task.x", "cond.B_task.y"]
    assert _match_conditions_to_formula(nested, "cond != 'A'").tolist() == [
        False, False, True, True,
    ]
