"""Parity tests for factorial contrast generators."""

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.contrast import (
    generate_interaction_contrast,
    generate_main_effect_contrast,
)


def _design_4x3() -> pd.DataFrame:
    levels_a = ["1", "2", "3", "4"]
    levels_b = ["a", "b", "c"]
    cells = [(a, b) for a in levels_a for b in levels_b]
    des = pd.DataFrame(cells, columns=["A", "B"])
    des["A"] = pd.Categorical(des["A"], categories=levels_a)
    des["B"] = pd.Categorical(des["B"], categories=levels_b)
    return des


def _diff_block(n_levels: int) -> np.ndarray:
    return -np.diff(np.eye(n_levels), axis=0).T


def test_main_effect_single_factor_matches_r_difference_coding():
    des = pd.DataFrame({"A": pd.Categorical(["1", "2", "3", "4"])})
    out = generate_main_effect_contrast(des, "A")
    expected = _diff_block(4)
    assert out.shape == (4, 3)
    assert np.array_equal(out, expected)


def test_main_effects_in_two_factor_design_use_kronecker_blocks():
    des = _design_4x3()
    out_a = generate_main_effect_contrast(des, "A")
    out_b = generate_main_effect_contrast(des, "B")

    expected_a = np.kron(_diff_block(4), np.ones((3, 1)))
    expected_b = np.kron(np.ones((4, 1)), _diff_block(3))

    assert out_a.shape == (12, 3)
    assert out_b.shape == (12, 2)
    assert np.array_equal(out_a, expected_a)
    assert np.array_equal(out_b, expected_b)


def test_interaction_contrast_matches_kronecker_of_difference_blocks():
    des = _design_4x3()
    out = generate_interaction_contrast(des, ["A", "B"])
    expected = np.kron(_diff_block(4), _diff_block(3))
    assert out.shape == (12, 6)
    assert np.array_equal(out, expected)


def test_interaction_single_factor_matches_main_effect():
    des = _design_4x3()
    out_inter = generate_interaction_contrast(des, "A")
    out_main = generate_main_effect_contrast(des, "A")
    assert np.array_equal(out_inter, out_main)


def test_invalid_inputs_raise():
    des = _design_4x3()
    with pytest.raises(ValueError, match="exactly one factor"):
        generate_main_effect_contrast(des, ["A", "B"])
    with pytest.raises(ValueError, match="not found"):
        generate_interaction_contrast(des, ["A", "missing"])


def test_top_level_exports_work():
    des = _design_4x3()
    out_a = fm.generate_main_effect_contrast(des, "A")
    out_i = fm.generate_interaction_contrast(des, ["A", "B"])
    assert out_a.shape == (12, 3)
    assert out_i.shape == (12, 6)
