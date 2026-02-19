"""Computational stress tests for fmri_meta."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmrimod.dataset import group_data_from_csv
from fmrimod.stats import fmri_meta


def _group_data_from_df(
    df: pd.DataFrame,
    *,
    subjects: list[str],
    covariates: pd.DataFrame | None = None,
) -> object:
    return group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="roi",
        subjects=subjects,
        covariates=covariates,
    )


def test_fe_equal_weights_matches_numpy_lstsq_coefficients():
    subjects = [f"s{i+1}" for i in range(6)]
    age = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float64)
    covariates = pd.DataFrame({"age": age})

    rows: list[dict[str, object]] = []
    for roi, intercept, slope in [("r1", 1.0, 0.05), ("r2", -2.0, 0.10)]:
        for s, a in zip(subjects, age):
            rows.append(
                {
                    "subject": s,
                    "roi": roi,
                    "beta": intercept + slope * a,
                    "se": 0.3,
                }
            )
    df = pd.DataFrame(rows)
    gd = _group_data_from_df(df, subjects=subjects, covariates=covariates)

    out = fmri_meta(gd, formula="~ 1 + age", method="fe", weights="equal")
    X = np.column_stack([np.ones_like(age), age])
    expected = []
    for roi in ["r1", "r2"]:
        y = df.loc[df["roi"] == roi, "beta"].to_numpy(dtype=np.float64)
        beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
        expected.append(beta_hat)
    expected_arr = np.vstack(expected)

    np.testing.assert_allclose(out.coefficients, expected_arr, atol=1e-12)


def test_fe_scaling_metamorphic_scales_coef_and_se_but_not_p_or_z():
    subjects = [f"s{i+1}" for i in range(5)]
    df = pd.DataFrame(
        {
            "subject": subjects,
            "roi": ["r1"] * len(subjects),
            "beta": [0.2, -0.1, 0.4, 0.1, 0.3],
            "se": [0.05, 0.20, 0.10, 0.15, 0.08],
        }
    )

    gd = _group_data_from_df(df, subjects=subjects)
    base = fmri_meta(gd, method="fe")

    scale = 37.0
    df_scaled = df.copy()
    df_scaled["beta"] *= scale
    df_scaled["se"] *= scale
    gd_scaled = _group_data_from_df(df_scaled, subjects=subjects)
    got = fmri_meta(gd_scaled, method="fe")

    np.testing.assert_allclose(got.coefficients, base.coefficients * scale, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got.se, base.se * scale, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got.z, base.z, atol=1e-12)
    np.testing.assert_allclose(got.p, base.p, atol=1e-12)


def test_custom_weights_vector_matches_replicated_matrix():
    subjects = [f"s{i+1}" for i in range(5)]
    rows: list[dict[str, object]] = []
    for roi, a, b in [("r1", 1.0, 0.1), ("r2", -2.0, 0.2)]:
        for i, s in enumerate(subjects):
            rows.append(
                {
                    "subject": s,
                    "roi": roi,
                    "beta": a + b * i,
                    "se": 0.2 + 0.05 * i,
                }
            )
    df = pd.DataFrame(rows)
    gd = _group_data_from_df(df, subjects=subjects)

    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    out_vec = fmri_meta(gd, method="fe", weights="custom", weights_custom=w)
    out_mat = fmri_meta(
        gd,
        method="fe",
        weights="custom",
        weights_custom=np.tile(w[:, np.newaxis], (1, 2)),
    )

    np.testing.assert_allclose(out_vec.coefficients, out_mat.coefficients, atol=1e-12)
    np.testing.assert_allclose(out_vec.se, out_mat.se, atol=1e-12)
    np.testing.assert_allclose(out_vec.p, out_mat.p, atol=1e-12)


@pytest.mark.parametrize(
    "weights_custom",
    [
        np.array([1.0, 0.0, 1.0, 1.0]),
        np.array([1.0, -1.0, 1.0, 1.0]),
        np.array([1.0, np.nan, 1.0, 1.0]),
    ],
)
def test_custom_weights_reject_non_positive_or_non_finite(weights_custom: np.ndarray):
    subjects = ["s1", "s2", "s3", "s4"]
    df = pd.DataFrame(
        {
            "subject": subjects,
            "roi": ["r1"] * 4,
            "beta": [0.1, 0.2, 0.3, 0.4],
            "se": [0.2, 0.2, 0.2, 0.2],
        }
    )
    gd = _group_data_from_df(df, subjects=subjects)

    with pytest.raises(ValueError, match="Weights must be finite and > 0"):
        fmri_meta(gd, method="fe", weights="custom", weights_custom=weights_custom)


def test_extreme_variance_scales_are_finite_and_weighted_toward_low_variance_subject():
    subjects = [f"s{i+1}" for i in range(6)]
    df = pd.DataFrame(
        {
            "subject": subjects,
            "roi": ["r1"] * len(subjects),
            "beta": [3.0, -10.0, 20.0, 15.0, -8.0, 7.0],
            "se": [1e-6, 1e3, 1e3, 1e3, 1e3, 1e3],
        }
    )
    gd = _group_data_from_df(df, subjects=subjects)
    out = fmri_meta(gd, method="fe", weights="ivw")

    assert np.all(np.isfinite(out.coefficients))
    assert np.all(np.isfinite(out.se))
    assert np.all(np.isfinite(out.z))
    assert np.all(np.isfinite(out.p))
    assert abs(out.coefficients[0, 0] - 3.0) < 1e-6


@pytest.mark.parametrize("method", ["dl", "pm", "reml"])
def test_random_effects_tau2_is_near_zero_for_homogeneous_effects(method: str):
    subjects = [f"s{i+1}" for i in range(8)]
    rows: list[dict[str, object]] = []
    for roi in ["r1", "r2"]:
        for i, s in enumerate(subjects):
            rows.append(
                {
                    "subject": s,
                    "roi": roi,
                    "beta": 2.0,
                    "se": 0.1 + 0.02 * i,
                }
            )
    df = pd.DataFrame(rows)
    gd = _group_data_from_df(df, subjects=subjects)
    out = fmri_meta(gd, method=method)

    assert np.all(out.tau2 >= 0.0)
    assert np.max(out.tau2) < 1e-4
    np.testing.assert_allclose(out.coefficients[:, 0], 2.0, atol=1e-8)


def test_random_effects_with_non_intercept_formula_raises_not_implemented():
    subjects = [f"s{i+1}" for i in range(6)]
    age = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float64)
    covariates = pd.DataFrame({"age": age})
    df = pd.DataFrame(
        {
            "subject": subjects,
            "roi": ["r1"] * len(subjects),
            "beta": 1.0 + 0.1 * age,
            "se": np.full(len(subjects), 0.2),
        }
    )
    gd = _group_data_from_df(df, subjects=subjects, covariates=covariates)

    with pytest.raises(NotImplementedError, match="intercept-only"):
        fmri_meta(gd, formula="~ 1 + age", method="pm")


def test_rank_deficient_design_warns_and_returns_finite_outputs():
    subjects = [f"s{i+1}" for i in range(6)]
    age = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float64)
    covariates = pd.DataFrame({"age": age, "age_dup": 2.0 * age})
    df = pd.DataFrame(
        {
            "subject": subjects,
            "roi": ["r1"] * len(subjects),
            "beta": 1.0 + 0.02 * age,
            "se": np.full(len(subjects), 0.3),
        }
    )
    gd = _group_data_from_df(df, subjects=subjects, covariates=covariates)

    with pytest.warns(UserWarning, match="rank-deficient"):
        out = fmri_meta(gd, formula="~ 1 + age + age_dup", method="fe", weights="equal")

    assert np.all(np.isfinite(out.coefficients))
    assert np.all(np.isfinite(out.se))
    assert np.all(np.isfinite(out.z))
    assert np.all(np.isfinite(out.p))
