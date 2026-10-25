"""REG-B-09: missing data and rank deficiency in the group facades.

fmrireg ebf57cb (B7) drops subjects with non-finite outcomes per feature
(classic t-test, meta kernel ``fit_one``) instead of failing the whole
analysis, and 3ed5e8d makes a rank-deficient group design stop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from fmrimod.dataset import group_data_from_csv
from fmrimod.glm.compat import fmri_ols_fit
from fmrimod.stats import fmri_meta, fmri_ttest


def _frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subjects = [f"s{i:02d}" for i in range(12)]
    rows = []
    for roi in ("r1", "r2", "r3"):
        for s in subjects:
            rows.append(
                {
                    "subject": s,
                    "roi": roi,
                    "beta": rng.normal(0.4, 1.0),
                    "var": rng.uniform(0.05, 0.3),
                }
            )
    df = pd.DataFrame(rows)
    # One missing effect in r2, a NaN and an Inf in r3, a bad variance in r1.
    df.loc[(df.roi == "r2") & (df.subject == "s03"), "beta"] = np.nan
    df.loc[(df.roi == "r3") & (df.subject == "s01"), "beta"] = np.nan
    df.loc[(df.roi == "r3") & (df.subject == "s08"), "beta"] = np.inf
    df.loc[(df.roi == "r1") & (df.subject == "s05"), "var"] = np.nan
    return df


def _gd(df: pd.DataFrame, effect_cols: dict[str, str]):
    return group_data_from_csv(
        df, effect_cols=effect_cols, subject_col="subject", roi_col="roi"
    )


def test_classic_ttest_uses_complete_cases_per_feature():
    df = _frame()
    result = fmri_ttest(_gd(df, {"beta": "beta", "var": "var"}), engine="classic")

    for i, roi in enumerate(("r1", "r2", "r3")):
        y = df.loc[df.roi == roi, "beta"].to_numpy()
        y = y[np.isfinite(y)]
        ref = sp_stats.ttest_1samp(y, popmean=0.0)
        np.testing.assert_allclose(result.estimate[i], y.mean(), rtol=1e-12)
        np.testing.assert_allclose(
            result.se[i], y.std(ddof=1) / np.sqrt(y.size), rtol=1e-12
        )
        np.testing.assert_allclose(result.statistic[i], ref.statistic, rtol=1e-12)
        np.testing.assert_allclose(result.p[i], ref.pvalue, rtol=1e-12)
    np.testing.assert_array_equal(result.n_obs, [12, 11, 10])


@pytest.mark.parametrize("method", ["fe", "dl", "pm"])
def test_meta_drops_invalid_subjects_per_feature(method):
    df = _frame()
    fit = fmri_meta(_gd(df, {"beta": "beta", "var": "var"}), method=method)

    for i, roi in enumerate(("r1", "r2", "r3")):
        block = df[df.roi == roi]
        keep = np.isfinite(block.beta) & np.isfinite(block["var"]) & (block["var"] > 0)
        ref = fmri_meta(
            _gd(df[df.subject.isin(block.subject[keep])], {"beta": "beta", "var": "var"}),
            method=method,
        )
        np.testing.assert_allclose(fit.coefficients[i], ref.coefficients[i], rtol=1e-10)
        np.testing.assert_allclose(fit.se[i], ref.se[i], rtol=1e-10)
        np.testing.assert_allclose(fit.tau2[i], ref.tau2[i], rtol=1e-10, atol=1e-14)
    np.testing.assert_array_equal(fit.n_obs, [11, 11, 10])


def test_meta_reference_for_reduced_rows_is_inverse_variance_mean():
    """The complete-case answer is the FE estimate on the kept rows."""
    df = _frame()
    fit = fmri_meta(_gd(df, {"beta": "beta", "var": "var"}), method="fe")
    block = df[df.roi == "r3"]
    keep = np.isfinite(block.beta)
    w = 1.0 / block["var"][keep].to_numpy()
    y = block.beta[keep].to_numpy()
    np.testing.assert_allclose(fit.coefficients[2, 0], np.sum(w * y) / np.sum(w), rtol=1e-12)
    np.testing.assert_allclose(fit.se[2, 0], np.sqrt(1.0 / np.sum(w)), rtol=1e-12)


def test_unestimable_feature_is_nan_not_an_error():
    df = _frame()
    df.loc[(df.roi == "r2") & (df.subject != "s00"), "beta"] = np.nan
    result = fmri_ttest(_gd(df, {"beta": "beta", "var": "var"}), engine="classic")
    assert np.isnan(result.statistic[1]) and np.isnan(result.p[1])
    assert np.all(np.isfinite(result.statistic[[0, 2]]))
    assert result.n_obs[1] == 1


def test_fmri_ols_fit_rejects_rank_deficient_design():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(12, 3))
    X = np.ones((12, 2))
    with pytest.raises(ValueError, match="rank deficient"):
        fmri_ols_fit(Y, X)

    age = rng.normal(size=12)
    X = np.column_stack([np.ones(12), age, 2.0 * age])
    with pytest.raises(ValueError, match="full column rank; got rank 2 for 3"):
        fmri_ols_fit(Y, X)

    fit = fmri_ols_fit(Y, np.column_stack([np.ones(12), age]))
    assert fit["df"] == 10
