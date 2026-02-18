"""Tests for parity-oriented fmri_meta implementation."""

import numpy as np
import pandas as pd
import pytest

import fmrimod
from fmrimod.dataset import group_data_from_csv, group_data_from_h5
from fmrimod.stats import fmri_meta


def _make_csv_group_data() -> object:
    df = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s3", "s4"],
            "beta": [0.20, 0.10, 0.30, 0.25],
            "se": [0.10, 0.20, 0.10, 0.10],
            "roi": ["r1", "r1", "r1", "r1"],
        }
    )
    return group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="roi",
    )


def test_fmri_meta_fixed_effects_intercept_matches_inverse_variance_mean():
    gd = _make_csv_group_data()
    result = fmri_meta(gd, formula="~ 1", method="fe")

    y = np.array([0.20, 0.10, 0.30, 0.25], dtype=np.float64)
    se = np.array([0.10, 0.20, 0.10, 0.10], dtype=np.float64)
    w = 1.0 / (se ** 2)
    expected = np.sum(w * y) / np.sum(w)

    assert result.coefficients.shape == (1, 1)
    np.testing.assert_allclose(result.coefficients[0, 0], expected, atol=1e-12)
    assert result.predictor_names == ["Intercept"]
    assert result.feature_names == ["roi=r1"]


def test_fmri_meta_random_effects_methods_run_without_alias_warning():
    gd = _make_csv_group_data()
    pm = fmri_meta(gd, formula="~ 1", method="pm")
    reml = fmri_meta(gd, formula="~ 1", method="reml")
    assert pm.tau2.shape == (1,)
    assert reml.tau2.shape == (1,)
    assert pm.tau2[0] >= 0.0
    assert reml.tau2[0] >= 0.0


def test_fmri_meta_pm_and_reml_are_distinct_from_dl_on_heterogeneous_data():
    df = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "beta": [-1.2, 0.1, 2.3, 0.0, 1.4, -0.7],
            "se": [np.sqrt(0.2), np.sqrt(0.3), np.sqrt(0.15), np.sqrt(0.25), np.sqrt(0.18), np.sqrt(0.22)],
            "roi": ["r1", "r1", "r1", "r1", "r1", "r1"],
        }
    )
    gd = group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="roi",
    )
    dl = fmri_meta(gd, formula="~ 1", method="dl")
    pm = fmri_meta(gd, formula="~ 1", method="pm")
    reml = fmri_meta(gd, formula="~ 1", method="reml")
    assert pm.tau2[0] >= 0.0
    assert reml.tau2[0] >= 0.0
    assert not np.isclose(pm.tau2[0], dl.tau2[0])
    assert not np.isclose(reml.tau2[0], dl.tau2[0])


def test_fmri_meta_with_covariate_formula_on_csv_data():
    df = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s3", "s4"],
            "beta": [0.2, 0.25, 0.3, 0.35],
            "se": [0.1, 0.1, 0.1, 0.1],
        }
    )
    cov = pd.DataFrame({"age": [20.0, 30.0, 40.0, 50.0]})
    gd = group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        covariates=cov,
    )
    result = fmri_meta(gd, formula="~ 1 + age", method="fe")
    assert result.coefficients.shape == (1, 2)
    assert result.predictor_names == ["Intercept", "age"]


def test_fmri_meta_top_level_wrapper_matches_stats_module():
    gd = _make_csv_group_data()
    a = fmri_meta(gd, method="fe")
    b = fmrimod.fmri_meta(gd, method="fe")
    np.testing.assert_allclose(a.coefficients, b.coefficients, atol=1e-12)
    np.testing.assert_allclose(a.se, b.se, atol=1e-12)


def test_fmri_meta_rejects_non_csv_group_data_in_this_slice(tmp_path):
    h5_path = tmp_path / "sub-01.h5"
    h5_path.touch()
    gd = group_data_from_h5([h5_path], subjects=["sub-01"], validate=True)
    with pytest.raises(NotImplementedError, match="format='csv' only"):
        fmri_meta(gd)
