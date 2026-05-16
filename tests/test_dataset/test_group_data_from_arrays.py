"""P2: group_data_from_arrays — typed array boundary -> frozen GroupData.

bd-01KRFMD3JFKMH2ETGPRRAWAFME criterion (c): the array constructor
produces a *frozen* GroupData (not a hand-assembled dict). It delegates
to the existing typed group_data_from_csv so there is one validation
path; these pin numeric equivalence to that path (proving delegation,
not a reimplementation that could drift), the frozen-type contract, and
the validation guards. A stub returning a dict, or a parallel
validation copy, fails these.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from fmrimod.dataset import group_data_from_arrays, group_data_from_csv
from fmrimod.dataset.group_data import GroupData
from fmrimod.stats import fmri_meta

_BETA = np.array([0.42, -0.11, 0.63, 0.20, -0.30, 0.55, 0.08, 0.37])
_SE = np.array([0.15, 0.22, 0.18, 0.25, 0.20, 0.17, 0.24, 0.19])
_SUBJECTS = [f"s{i + 1}" for i in range(8)]


def test_arrays_path_is_a_frozen_group_data_not_a_dict() -> None:
    gd = group_data_from_arrays(_BETA, se=_SE)
    assert isinstance(gd, GroupData)
    assert dataclasses.is_dataclass(gd)
    with pytest.raises(dataclasses.FrozenInstanceError):
        gd.format = "tampered"  # type: ignore[misc]


def test_arrays_path_matches_explicit_csv_construction() -> None:
    from_arrays = group_data_from_arrays(_BETA, se=_SE, subjects=_SUBJECTS)
    frame = pd.DataFrame({"subject": _SUBJECTS, "beta": _BETA, "se": _SE})
    from_csv = group_data_from_csv(
        frame, effect_cols={"beta": "beta", "se": "se"}, subject_col="subject"
    )

    a = fmri_meta(from_arrays, method="dl")
    c = fmri_meta(from_csv, method="dl")
    np.testing.assert_allclose(a.coefficients, c.coefficients, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a.se, c.se, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a.tau2, c.tau2, atol=1e-12)


def test_var_path_equivalent_to_se_squared() -> None:
    by_se = fmri_meta(group_data_from_arrays(_BETA, se=_SE), method="dl")
    by_var = fmri_meta(group_data_from_arrays(_BETA, var=_SE**2), method="dl")
    np.testing.assert_allclose(
        by_se.coefficients, by_var.coefficients, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(by_se.tau2, by_var.tau2, atol=1e-12)


def test_subjects_default_is_generated_and_length_checked() -> None:
    gd = group_data_from_arrays(_BETA, se=_SE)
    assert gd.subjects == _SUBJECTS  # auto s1..sN
    with pytest.raises(ValueError, match="'subjects' length must match"):
        group_data_from_arrays(_BETA, se=_SE, subjects=["only", "two"])


def test_covariates_pass_through_supports_meta_regression() -> None:
    age = np.array([20.0, 33.0, 41.0, 29.0, 55.0, 48.0, 37.0, 26.0])
    gd = group_data_from_arrays(
        _BETA, se=_SE, covariates=pd.DataFrame({"age": age})
    )
    out = fmri_meta(gd, formula="~ 1 + age", method="dl")
    assert out.predictor_names == ["Intercept", "age"]
    assert np.all(np.isfinite(out.coefficients))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "exactly one of se= or var="),
        ({"se": _SE, "var": _SE**2}, "exactly one of se= or var="),
        ({"se": np.array([0.1, 0.2])}, "same shape as 'beta'"),
    ],
)
def test_effect_uncertainty_guards(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        group_data_from_arrays(_BETA, **kwargs)


def test_beta_must_be_one_dimensional() -> None:
    with pytest.raises(ValueError, match="must be a 1-D array"):
        group_data_from_arrays(np.zeros((4, 2)), se=np.ones((4, 2)))
