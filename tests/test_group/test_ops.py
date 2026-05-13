"""Tests for eager native group-analysis operations."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.group import (
    AdapterContractError,
    SampleLabelSpace,
    UnsupportedGroupFeatureError,
    derive,
    group_dataset,
    mask,
    posthoc,
    reduce,
    subset,
    write_out,
)


def _dataset():
    return group_dataset(
        {
            "beta": np.array(
                [
                    [[1.0], [2.0]],
                    [[3.0], [4.0]],
                    [[5.0], [6.0]],
                ]
            ),
            "se": np.ones((3, 2, 1)),
        },
        space=SampleLabelSpace(["r1", "r2", "r3"]),
        subjects=["s1", "s2"],
        contrasts=["c1"],
    )


def test_subset_restricts_axes_and_metadata() -> None:
    ds = _dataset()
    out = subset(ds, sample=[0, 2], subject=[1], contrast=[0])

    assert out.shape == (2, 1, 1)
    assert out.space.labels == ("r1", "r3")
    assert out.subjects == ("s2",)
    assert out.contrasts == ("c1",)
    np.testing.assert_allclose(out.assay("beta")[:, 0, 0], np.array([2.0, 6.0]))


def test_mask_aliases_sample_subset() -> None:
    ds = _dataset()
    out = mask(ds, np.array([True, False, True]))

    assert out.space.labels == ("r1", "r3")
    np.testing.assert_allclose(out.assay("beta")[:, :, 0], np.array([[1.0, 2.0], [5.0, 6.0]]))


def test_subset_rejects_bad_axis_indices() -> None:
    ds = _dataset()
    with pytest.raises(AdapterContractError, match="out of range"):
        subset(ds, sample=[3])
    with pytest.raises(AdapterContractError, match="logical index length"):
        subset(ds, subject=np.array([True, False, True]))


def test_derive_adds_variance_t_z_and_p() -> None:
    ds = _dataset()
    out = derive(ds, ["var", "t", "z", "p"])

    np.testing.assert_allclose(out.assay("var"), np.ones((3, 2, 1)))
    np.testing.assert_allclose(out.assay("t"), ds.assay("beta"))
    np.testing.assert_allclose(out.assay("z"), ds.assay("beta"))
    assert np.all(out.assay("p") < 1.0)


def test_derive_can_compute_se_from_var() -> None:
    ds = group_dataset(
        {"beta": np.ones((1, 1, 1)), "var": np.array([[[4.0]]])},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1"],
        contrasts=["c1"],
    )

    out = derive(ds, ["se", "z"])

    np.testing.assert_allclose(out.assay("se"), np.array([[[2.0]]]))
    np.testing.assert_allclose(out.assay("z"), np.array([[[0.5]]]))


def test_derive_rejects_unknown_target() -> None:
    with pytest.raises(UnsupportedGroupFeatureError, match="not supported"):
        derive(_dataset(), "cohens_d")


def test_posthoc_adds_q_assay_columnwise() -> None:
    ds = group_dataset(
        {"p": np.array([[[0.01]], [[0.02]], [[0.50]]])},
        space=SampleLabelSpace(["r1", "r2", "r3"]),
        subjects=["s1"],
        contrasts=["c1"],
    )

    out = posthoc(ds, "fdr:bh")

    assert "q" in out.assays
    np.testing.assert_allclose(out.assay("q")[:, 0, 0], np.array([0.03, 0.03, 0.5]))


def test_reduce_and_write_out_are_explicit_phase_gaps() -> None:
    ds = _dataset()
    with pytest.raises(UnsupportedGroupFeatureError, match="R fmrigds oracle"):
        reduce(ds, method="lmm:ri")
    with pytest.raises(UnsupportedGroupFeatureError, match="format='h5'"):
        write_out(ds, "out.csv", format="csv")
