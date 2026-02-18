"""Tests for parity-oriented fmri_ttest wrapper."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

import fmrimod
from fmrimod.dataset import group_data_from_csv, group_data_from_h5
from fmrimod.stats import fmri_ttest


def _csv_group_data_with_se() -> object:
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


def test_fmri_ttest_auto_selects_meta_when_se_available():
    gd = _csv_group_data_with_se()
    result = fmri_ttest(gd, engine="auto", method="fe")
    assert result.engine == "meta"
    assert result.estimate.shape == (1,)
    assert result.p.shape == (1,)


def test_fmri_ttest_classic_matches_scipy_one_sample_t():
    gd = _csv_group_data_with_se()
    result = fmri_ttest(gd, engine="classic")

    y = np.array([0.20, 0.10, 0.30, 0.25], dtype=np.float64)
    stat, pval = sp_stats.ttest_1samp(y, popmean=0.0)
    np.testing.assert_allclose(result.statistic[0], stat, atol=1e-12)
    np.testing.assert_allclose(result.p[0], pval, atol=1e-12)
    assert result.engine == "classic"


def test_fmri_ttest_auto_selects_meta_with_var_when_no_se():
    df = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s3", "s4"],
            "beta": [0.20, 0.10, 0.30, 0.25],
            "var": [0.01, 0.04, 0.01, 0.01],
            "roi": ["r1", "r1", "r1", "r1"],
        }
    )
    gd = group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "var": "var"},
        subject_col="subject",
        roi_col="roi",
    )
    result = fmri_ttest(gd, engine="auto", method="fe")
    assert result.engine == "meta"
    assert result.estimate.shape == (1,)
    assert result.p.shape == (1,)


def test_fmri_ttest_top_level_wrapper_matches_stats_module():
    gd = _csv_group_data_with_se()
    a = fmri_ttest(gd, engine="meta", method="fe")
    b = fmrimod.fmri_ttest(gd, engine="meta", method="fe")
    np.testing.assert_allclose(a.estimate, b.estimate, atol=1e-12)
    np.testing.assert_allclose(a.p, b.p, atol=1e-12)


def test_fmri_ttest_rejects_non_csv_group_data_in_this_slice(tmp_path):
    h5_path = tmp_path / "sub-01.h5"
    h5_path.touch()
    gd = group_data_from_h5([h5_path], subjects=["sub-01"], validate=True)
    with pytest.raises(NotImplementedError, match="format='csv' only"):
        fmri_ttest(gd, engine="meta")
