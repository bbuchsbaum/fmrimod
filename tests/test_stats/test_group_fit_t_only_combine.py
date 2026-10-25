"""Combining t-only subject data through ``group_fit`` (inventory v2, REG-B-05/GDS-02).

fmrireg combines t+df data with per-subject p from ``pt(|t|, df)`` and
Stouffer z_i = -qnorm(pt(-|t|, df)) * sign(t) (``R/metafuns.R``
``combine_t_statistics`` after ``aecf0ac``; fmrigds ``R/derive-stats.R``).
df = 8 is small enough that treating t as z fails both checks.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from fmrimod.dataset import group_data_from_csv
from fmrimod.stats.inference import group_fit
from fmrimod.stats.interfaces import GroupFitRequest

DF = 8


def _t_only_data():
    rng = np.random.default_rng(5)
    t = rng.standard_t(DF, size=12) + 0.8
    frame = pd.DataFrame({"subject": [f"s{i}" for i in range(t.size)], "t": t, "df": DF})
    return group_data_from_csv(frame, effect_cols={"t": "t", "df": "df"}), t


def test_fisher_uses_t_distribution_p_values():
    gd, t = _t_only_data()
    out = group_fit(GroupFitRequest(data=gd, model="meta", combine="fisher"))

    chi2 = -2.0 * np.sum(np.log(2.0 * sp_stats.t.cdf(-np.abs(t), DF)))
    np.testing.assert_allclose(out.p.ravel()[0], sp_stats.chi2.sf(chi2, 2 * t.size), rtol=1e-10)


def test_stouffer_derives_z_from_t_and_df():
    gd, t = _t_only_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = group_fit(GroupFitRequest(data=gd, model="meta", combine="stouffer"))

    z_i = -sp_stats.norm.ppf(sp_stats.t.cdf(-np.abs(t), DF)) * np.sign(t)
    np.testing.assert_allclose(out.statistic.ravel()[0], z_i.sum() / np.sqrt(t.size), rtol=1e-10)


def test_stouffer_with_default_ivw_weights_warns_that_equal_weights_are_used():
    gd, _ = _t_only_data()
    with pytest.warns(UserWarning, match="equal weights"):
        group_fit(GroupFitRequest(data=gd, model="meta", combine="stouffer"))
