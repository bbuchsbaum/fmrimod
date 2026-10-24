"""Factorial contrast generators: row r is the cell in row r of ``des``.

Port of the first three blocks of fmrireg ``test-review-C-gen-contrast.R``
(review C3, 91bb402). ``expand.grid`` order lists the first factor fastest;
the generators must follow whatever row order ``des`` has.
"""

import numpy as np
import pandas as pd
import pytest

from fmrimod.contrast import (
    generate_interaction_contrast,
    generate_main_effect_contrast,
)


def _expand_grid(**levels: list[str]) -> pd.DataFrame:
    """R ``expand.grid`` of categorical factors (first factor fastest)."""
    names = list(levels)
    grids = np.meshgrid(*[np.arange(len(v)) for v in levels.values()], indexing="ij")
    codes = [g.ravel(order="F") for g in grids]
    return pd.DataFrame(
        {
            name: pd.Categorical.from_codes(code, categories=levels[name])
            for name, code in zip(names, codes, strict=True)
        }
    )


def _cell_means(des, a, b, ab):
    ti = des["Time"].cat.codes.to_numpy()
    ci = des["Cond"].cat.codes.to_numpy()
    return a[ti] + b[ci] + ab[ti, ci]


def _time_by_cond() -> pd.DataFrame:
    return _expand_grid(Time=["1", "2", "3", "4"], Cond=["face", "scene"])


def test_expand_grid_helper_lists_first_factor_fastest():
    des = _time_by_cond()
    assert list(des["Time"]) == ["1", "2", "3", "4"] * 2
    assert list(des["Cond"]) == ["face"] * 4 + ["scene"] * 4


def test_main_effect_contrast_isolates_its_factor():
    des = _time_by_cond()
    a = np.array([0.0, 1.0, 3.0, 7.0])
    b = np.array([0.0, 10.0])
    mu = _cell_means(des, a, b, np.zeros((4, 2)))

    M = generate_main_effect_contrast(des, "Time")
    np.testing.assert_allclose(M.T @ mu / 2, -np.diff(a))
    mu_cond_only = _cell_means(des, np.zeros(4), b, np.zeros((4, 2)))
    np.testing.assert_allclose(M.T @ mu_cond_only, np.zeros(3), atol=1e-12)

    MC = generate_main_effect_contrast(des, "Cond")
    np.testing.assert_allclose(MC.T @ mu / 4, -np.diff(b))
    mu_time_only = _cell_means(des, a, np.zeros(2), np.zeros((4, 2)))
    np.testing.assert_allclose(MC.T @ mu_time_only, np.zeros(1), atol=1e-12)


def test_interaction_contrast_is_difference_of_differences():
    des = _time_by_cond()
    a = np.array([0.0, 1.0, 3.0, 7.0])
    b = np.array([0.0, 10.0])
    inter = generate_interaction_contrast(des, ["Time", "Cond"])
    np.testing.assert_allclose(
        inter.T @ _cell_means(des, a, b, np.zeros((4, 2))), np.zeros(3), atol=1e-12
    )

    ab = np.zeros((4, 2))
    ab[2, 1] = 5.0
    d = ab[:, 0] - ab[:, 1]
    np.testing.assert_allclose(inter.T @ _cell_means(des, a, b, ab), -np.diff(d))


def test_contrast_rows_follow_a_permuted_design():
    des = _expand_grid(Time=["1", "2", "3"], Cond=["x", "y"])
    perm = np.array([4, 1, 6, 2, 5, 3]) - 1
    des_p = des.iloc[perm].reset_index(drop=True)

    M = generate_main_effect_contrast(des, "Time")
    Mp = generate_main_effect_contrast(des_p, "Time")
    np.testing.assert_array_equal(Mp, M[perm])

    inter = generate_interaction_contrast(des, ["Time", "Cond"])
    inter_p = generate_interaction_contrast(des_p, ["Time", "Cond"])
    np.testing.assert_array_equal(inter_p, inter[perm])


@pytest.mark.parametrize(
    "rows",
    [
        [0, 1, 2, 3, 4],  # a cell missing
        [0, 1, 2, 3, 4, 4],  # a cell listed twice, another missing
    ],
)
def test_incomplete_factorial_design_raises(rows):
    des = _expand_grid(Time=["1", "2", "3"], Cond=["x", "y"]).iloc[rows]
    with pytest.raises(ValueError, match="every combination"):
        generate_main_effect_contrast(des, "Time")
