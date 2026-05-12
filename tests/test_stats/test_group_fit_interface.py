"""Tests for canonical group_fit interface and parity scaffolding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmrimod.dataset import group_data_from_csv
from fmrimod.stats import (
    GroupFitRequest,
    available_second_level_backends,
    fmri_meta,
    fmri_ttest,
    group_fit,
)


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


def test_group_fit_meta_matches_fmri_meta_for_fixed_effects():
    gd = _make_csv_group_data()
    request = GroupFitRequest(data=gd, model="meta", effects="fixed")
    got = group_fit(request)
    ref = fmri_meta(gd, formula="~ 1", method="fe")

    np.testing.assert_allclose(got.estimate, ref.coefficients, atol=1e-12)
    np.testing.assert_allclose(got.se, ref.se, atol=1e-12)
    np.testing.assert_allclose(got.statistic, ref.z, atol=1e-12)
    np.testing.assert_allclose(got.p, ref.p, atol=1e-12)
    np.testing.assert_allclose(got.tau2, ref.tau2, atol=1e-12)
    assert got.method == "fe"
    assert got.backend == "python"


def test_group_fit_ttest_matches_fmri_ttest_auto_engine():
    gd = _make_csv_group_data()
    request = GroupFitRequest(data=gd, model="ttest", effects="fixed")
    got = group_fit(request)
    ref = fmri_ttest(gd, engine="auto", method="fe")

    np.testing.assert_allclose(got.estimate[:, 0], ref.estimate, atol=1e-12)
    np.testing.assert_allclose(got.se[:, 0], ref.se, atol=1e-12)
    np.testing.assert_allclose(got.statistic[:, 0], ref.statistic, atol=1e-12)
    np.testing.assert_allclose(got.p[:, 0], ref.p, atol=1e-12)
    assert got.method == "fe"
    assert got.backend == "python"


def test_group_fit_normalizes_aliases_for_weights_and_correction():
    gd = _make_csv_group_data()
    request = GroupFitRequest(
        data=gd,
        model="meta",
        method="fixed",
        weights="1/var",
        correction="fdr:bh",
    )
    out = group_fit(request)
    assert out.method == "fe"
    assert out.metadata["correction"] == "bh"
    assert out.q is not None
    np.testing.assert_equal(out.q.shape, out.p.shape)
    assert np.all(out.q >= out.p - 1e-12)


def test_group_fit_spatial_requires_group_ids():
    gd = _make_csv_group_data()
    request = GroupFitRequest(data=gd, model="meta", effects="fixed", correction="spatial")
    with pytest.raises(ValueError, match="group_ids"):
        group_fit(request)


def test_group_fit_fmrigds_backend_rejects_unimplemented_mode_explicitly():
    gd = _make_csv_group_data()
    request = GroupFitRequest(
        data=gd,
        model="meta",
        effects="random",
        tau2="pm",
        backend="fmrigds",
    )
    with pytest.raises(NotImplementedError, match="method='fe' or 'dl'"):
        group_fit(request)


def test_available_backends_surface_contains_expected_names():
    backends = available_second_level_backends()
    assert set(backends) == {"auto", "python", "fmrigds"}
