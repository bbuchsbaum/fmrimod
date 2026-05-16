"""DataFrame entry as a typed input boundary for fmri_meta / fmri_ttest.

bd-01KRFMD3JFKMH2ETGPRRAWAFME P1: a DataFrame may be passed directly,
but it is *validated and converted to a frozen GroupData at entry* via
the existing typed ``group_data_from_csv`` (its ``effect_cols`` schema),
never consumed as a string-keyed table at use time (the recorded
MISSION audit's criterion b). These pin:

* the DataFrame path is numerically identical to the explicit
  GroupData path (it really is the same typed object downstream — not
  a separate string-keyed code path);
* the schema is mandatory at construction (a DataFrame without
  ``effect_cols`` raises, not a silent escape hatch);
* the GroupData path is unchanged (``effect_cols`` is rejected there);
* a non-GroupData/non-DataFrame still raises ``TypeError``.

A stub that ignored the conversion, or consumed the DataFrame as a
bare dict, fails the numeric-equivalence and frozen-type assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmrimod.dataset import group_data_from_csv
from fmrimod.dataset.group_data import GroupData
from fmrimod.stats import fmri_meta, fmri_ttest

_EFFECT_COLS = {"beta": "beta", "se": "se"}


def _frame() -> pd.DataFrame:
    """Single-feature effect-size frame.

    Minimal P1 surfaces only ``effect_cols``; ``group_data_from_csv``'s
    ``roi_col``/``contrast_col`` keep their defaults, so the DataFrame
    path supports a single feature. Multi-ROI/contrast DataFrames still
    use the explicit GroupData path until those pass-throughs land
    (deferred per the bead plan).
    """
    subjects = [f"s{i + 1}" for i in range(8)]
    rng = np.random.default_rng(20260516)
    return pd.DataFrame(
        {
            "subject": subjects,
            "beta": [float(rng.normal(0.4, 0.2)) + 0.05 * i for i in range(8)],
            "se": [0.15 + 0.02 * i for i in range(8)],
        }
    )


def test_fmri_meta_dataframe_path_matches_explicit_group_data() -> None:
    df = _frame()
    gd = group_data_from_csv(df, effect_cols=_EFFECT_COLS)

    from_df = fmri_meta(df, method="dl", effect_cols=_EFFECT_COLS)
    from_gd = fmri_meta(gd, method="dl")

    np.testing.assert_allclose(
        from_df.coefficients, from_gd.coefficients, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(from_df.se, from_gd.se, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(from_df.tau2, from_gd.tau2, atol=1e-12)
    assert isinstance(from_gd, type(from_df))  # same typed result object


def test_fmri_ttest_dataframe_path_matches_explicit_group_data() -> None:
    df = _frame()
    gd = group_data_from_csv(df, effect_cols=_EFFECT_COLS)

    from_df = fmri_ttest(df, effect_cols=_EFFECT_COLS)
    from_gd = fmri_ttest(gd)

    np.testing.assert_allclose(
        from_df.statistic, from_gd.statistic, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(from_df.p, from_gd.p, rtol=1e-12, atol=1e-12)


def test_dataframe_requires_effect_cols_schema() -> None:
    df = _frame()
    with pytest.raises(ValueError, match="requires effect_cols"):
        fmri_meta(df, method="dl")
    with pytest.raises(ValueError, match="requires effect_cols"):
        fmri_ttest(df)


def test_group_data_path_rejects_effect_cols() -> None:
    gd = group_data_from_csv(_frame(), effect_cols=_EFFECT_COLS)
    with pytest.raises(ValueError, match="only applies when 'data' is a DataFrame"):
        fmri_meta(gd, effect_cols=_EFFECT_COLS)
    with pytest.raises(ValueError, match="only applies when 'data' is a DataFrame"):
        fmri_ttest(gd, effect_cols=_EFFECT_COLS)


def test_group_data_path_is_unchanged() -> None:
    # The existing GroupData path needs no effect_cols and still yields
    # finite results (the broad no-regression guard is the existing
    # meta/ttest suites; this pins the entry contract specifically).
    gd = group_data_from_csv(_frame(), effect_cols=_EFFECT_COLS)
    meta = fmri_meta(gd, method="dl")
    tt = fmri_ttest(gd)
    assert np.all(np.isfinite(meta.coefficients))
    assert np.all(np.isfinite(meta.tau2))
    assert np.all(np.isfinite(tt.p))


def test_non_group_data_non_dataframe_raises_typeerror() -> None:
    for bad in ([1, 2, 3], {"beta": 1.0}, np.zeros((3, 2))):
        with pytest.raises(TypeError, match="GroupData or a pandas DataFrame"):
            fmri_meta(bad, effect_cols=_EFFECT_COLS)  # type: ignore[arg-type]


def test_coerced_dataframe_is_a_frozen_group_data_not_a_raw_dict() -> None:
    """The typed-boundary contract: DataFrame -> frozen GroupData."""
    from fmrimod.stats.meta import _coerce_group_data

    coerced = _coerce_group_data(_frame(), _EFFECT_COLS)
    assert isinstance(coerced, GroupData)
    assert _coerce_group_data(coerced, None) is coerced
