"""Parity checks between Python and fmrigds backends.

These tests auto-skip when R/fmrigds is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fmrimod.dataset import group_data_from_csv
from fmrimod.stats import GroupFitRequest, fmrigds_backend_available, group_fit


def _candidate_fmrigds_source() -> str | None:
    env = os.environ.get("FMRIGDS_SOURCE_DIR", "").strip()
    if env and Path(env).exists():
        return env
    candidate = Path.home() / "code" / "fmrigds"
    if candidate.exists():
        return str(candidate)
    return None


def _skip_if_no_fmrigds() -> dict[str, str]:
    source = _candidate_fmrigds_source()
    ok, reason = fmrigds_backend_available(fmrigds_source=source)
    if not ok:
        pytest.skip(f"fmrigds unavailable: {reason}")
    opts: dict[str, str] = {}
    if source is not None:
        opts["fmrigds_source"] = source
    return opts


def _make_csv_group_data():
    df = pd.DataFrame(
        {
            "subject": ["s1", "s2", "s3", "s4", "s5"],
            "beta": [0.20, 0.10, 0.30, 0.25, 0.15],
            "se": [0.10, 0.20, 0.10, 0.10, 0.15],
            "roi": ["r1", "r1", "r1", "r1", "r1"],
        }
    )
    return group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="roi",
    )


@pytest.mark.cross_test
def test_group_fit_fmrigds_matches_python_for_fixed_effects():
    backend_opts = _skip_if_no_fmrigds()
    gd = _make_csv_group_data()

    py = group_fit(
        GroupFitRequest(
            data=gd,
            model="meta",
            effects="fixed",
            backend="python",
        )
    )
    rg = group_fit(
        GroupFitRequest(
            data=gd,
            model="meta",
            effects="fixed",
            backend="fmrigds",
            backend_options=backend_opts,
        )
    )

    np.testing.assert_allclose(py.estimate, rg.estimate, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(py.se, rg.se, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(py.p, rg.p, rtol=1e-6, atol=1e-8)


@pytest.mark.cross_test
def test_group_fit_fmrigds_matches_python_for_ttest_fixed():
    backend_opts = _skip_if_no_fmrigds()
    gd = _make_csv_group_data()

    py = group_fit(
        GroupFitRequest(
            data=gd,
            model="ttest",
            effects="fixed",
            backend="python",
        )
    )
    rg = group_fit(
        GroupFitRequest(
            data=gd,
            model="ttest",
            effects="fixed",
            backend="fmrigds",
            backend_options=backend_opts,
        )
    )

    np.testing.assert_allclose(py.estimate, rg.estimate, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(py.se, rg.se, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(py.p, rg.p, rtol=1e-6, atol=1e-8)


def test_group_fit_fmrigds_rejects_pm_until_backend_supports_it():
    gd = _make_csv_group_data()
    with pytest.raises(NotImplementedError, match="method='fe' or 'dl'"):
        group_fit(
            GroupFitRequest(
                data=gd,
                model="meta",
                effects="random",
                tau2="pm",
                backend="fmrigds",
            )
        )
