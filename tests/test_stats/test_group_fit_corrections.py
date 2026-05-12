"""Computation-focused tests for group_fit correction behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmrimod.dataset import group_data_from_csv
from fmrimod.stats import GroupFitRequest, group_fit


def _make_multi_feature_group_data():
    subjects = ["s1", "s2", "s3", "s4"]
    rows: list[dict[str, object]] = []
    # r1: near-null signal (high p)
    # r2: strong effect (very low p)
    # r3: moderate effect (intermediate p)
    beta_by_roi = {
        "r1": [0.01, 0.00, 0.02, -0.01],
        "r2": [0.50, 0.60, 0.55, 0.52],
        "r3": [0.20, 0.15, 0.25, 0.22],
    }
    for roi, betas in beta_by_roi.items():
        for subject, beta in zip(subjects, betas):
            rows.append(
                {
                    "subject": subject,
                    "roi": roi,
                    "beta": beta,
                    "se": 0.1,
                }
            )
    df = pd.DataFrame(rows)
    return group_data_from_csv(
        df,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="roi",
        subjects=subjects,
    )


def test_group_fit_by_is_more_conservative_than_bh():
    gd = _make_multi_feature_group_data()
    bh = group_fit(GroupFitRequest(data=gd, model="meta", effects="fixed", correction="bh"))
    by = group_fit(GroupFitRequest(data=gd, model="meta", effects="fixed", correction="by"))

    assert bh.q is not None
    assert by.q is not None
    assert bh.q.shape == bh.p.shape
    assert by.q.shape == by.p.shape
    assert np.all(bh.q >= bh.p - 1e-12)
    assert np.all(by.q >= by.p - 1e-12)
    assert np.all(by.q >= bh.q - 1e-12)


def test_group_fit_spatial_is_invariant_to_group_label_encoding():
    gd = _make_multi_feature_group_data()
    # Same partition with different raw labels should produce identical q-values
    # because spatial_fdr compresses labels to dense IDs.
    group_ids_a = np.array([1, 1, 2], dtype=np.intp)
    group_ids_b = np.array([10, 10, 99], dtype=np.intp)

    out_a = group_fit(
        GroupFitRequest(
            data=gd,
            model="meta",
            effects="fixed",
            correction="spatial",
            group_ids=group_ids_a,
        )
    )
    out_b = group_fit(
        GroupFitRequest(
            data=gd,
            model="meta",
            effects="fixed",
            correction="spatial",
            group_ids=group_ids_b,
        )
    )

    assert out_a.q is not None
    assert out_b.q is not None
    assert out_a.q.shape == out_a.p.shape
    assert np.all((out_a.q >= 0.0) & (out_a.q <= 1.0))
    np.testing.assert_allclose(out_a.q, out_b.q, atol=1e-12, rtol=1e-12)
    assert out_a.metadata["correction"] == "spatial"


def test_group_fit_spatial_rejects_group_length_mismatch():
    gd = _make_multi_feature_group_data()
    with pytest.raises(ValueError, match="group_ids length"):
        group_fit(
            GroupFitRequest(
                data=gd,
                model="meta",
                effects="fixed",
                correction="spatial",
                group_ids=[1, 2],  # should be length 3
            )
        )
