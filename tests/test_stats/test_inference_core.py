"""Core numerical invariants for stats inference helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmrimod.dataset import group_data_from_csv
from fmrimod.stats import GroupFitRequest, fdr_correction, group_fit, p_to_z, z_to_p


def _make_multi_feature_group_data():
    subjects = ["s1", "s2", "s3", "s4"]
    rows: list[dict[str, object]] = []
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


def test_p_to_z_and_z_to_p_are_inverse_for_two_sided_mode():
    p = np.array([1e-12, 1e-6, 1e-3, 0.02, 0.2, 0.8], dtype=np.float64)
    z = p_to_z(p, two_sided=True)
    p_back = z_to_p(z, two_sided=True)
    np.testing.assert_allclose(p_back, p, rtol=1e-10, atol=1e-12)


def test_p_to_z_is_monotone_decreasing_in_p():
    p = np.array([1e-8, 1e-4, 1e-2, 1e-1], dtype=np.float64)
    z = p_to_z(p, two_sided=True)
    assert np.all(np.diff(z) < 0)


def test_fdr_correction_is_permutation_invariant():
    p = np.array([0.3, 0.01, 0.2, 0.6, 0.001, 0.05], dtype=np.float64)
    perm = np.array([3, 0, 5, 2, 1, 4], dtype=np.intp)
    inv_perm = np.argsort(perm)

    reject_ref, q_ref = fdr_correction(p, alpha=0.05, method="bh")
    reject_perm, q_perm = fdr_correction(p[perm], alpha=0.05, method="bh")

    np.testing.assert_array_equal(reject_ref, reject_perm[inv_perm])
    np.testing.assert_allclose(q_ref, q_perm[inv_perm], atol=1e-12)


def test_by_adjusted_qvalues_are_never_smaller_than_bh():
    p = np.array([0.001, 0.01, 0.02, 0.2, 0.8], dtype=np.float64)
    _rbh, q_bh = fdr_correction(p, alpha=0.05, method="bh")
    _rby, q_by = fdr_correction(p, alpha=0.05, method="by")
    assert np.all(q_by >= q_bh - 1e-12)


def test_group_fit_spatial_rejects_non_integer_group_ids():
    gd = _make_multi_feature_group_data()
    with pytest.raises(ValueError, match="integer labels"):
        group_fit(
            GroupFitRequest(
                data=gd,
                model="meta",
                effects="fixed",
                correction="spatial",
                group_ids=np.array([0.0, 1.5, 2.0], dtype=np.float64),
            )
        )
