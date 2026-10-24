"""Tests for eager native group-analysis operations."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.group import (
    AdapterContractError,
    GroupProgressEvent,
    SampleLabelSpace,
    UnsupportedGroupFeatureError,
    derive,
    group_dataset,
    mask,
    posthoc,
    posthoc_registry,
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


def _t_df_dataset(t: list[float], df: float):
    t_arr = np.asarray(t, dtype=np.float64).reshape(1, -1, 1)
    return group_dataset(
        {"t": t_arr, "df": np.full_like(t_arr, df)},
        space=SampleLabelSpace(["a"]),
        subjects=[f"s{i + 1}" for i in range(t_arr.shape[1])],
        contrasts=["c1"],
    )


def test_derive_z_and_p_from_t_use_the_t_distribution() -> None:
    # fmrigds 7c7ecae, R/derive-stats.R and tests/testthat/test-review-numeric.R:120-147.
    # df = 5 and t = 12 (df = 100) rule out treating t as z.
    from scipy import stats as sp_stats

    out = derive(_t_df_dataset([12.0, 1.0, 1.0, -12.0, 60.0], 100.0), ["z", "p"])
    np.testing.assert_allclose(
        out.assay("z")[0, :4, 0],
        [9.4227301, 0.9950249, 0.9950249, -9.4227301],
        rtol=1e-7,
    )
    assert np.isfinite(out.assay("z")[0, 4, 0])
    np.testing.assert_allclose(
        out.assay("p")[0, :, 0],
        2.0 * sp_stats.t.sf([12.0, 1.0, 1.0, 12.0, 60.0], 100.0),
        rtol=1e-12,
    )

    p_df5 = derive(_t_df_dataset([2.5], 5.0), "p").assay("p")
    np.testing.assert_allclose(p_df5[0, 0, 0], 0.0544900, rtol=1e-5)
    assert "z" not in derive(_t_df_dataset([2.5], 5.0), "p").assays


def test_stouffer_on_z_derived_from_t_keeps_far_tail_subjects() -> None:
    # fmrigds test-review-numeric.R:135: sum(z_i) / sqrt(3) with z_i from t, df = 100.
    ds = derive(_t_df_dataset([12.0, 1.0, 1.0], 100.0), "z")
    out = reduce(ds, method="combine:stouffer")
    np.testing.assert_allclose(out.assay("z_g")[0, 0, 0], 6.589172, rtol=1e-6)


def test_derive_p_from_t_without_df_is_refused() -> None:
    ds = group_dataset(
        {"t": np.array([[[2.0]]])},
        space=SampleLabelSpace(["a"]),
        subjects=["s1"],
        contrasts=["c1"],
    )
    with pytest.raises(AdapterContractError, match="df"):
        derive(ds, "p")


def test_derive_signed_z_from_p_needs_beta() -> None:
    p = np.array([[[1e-20], [0.05]]])
    ds = group_dataset(
        {"p": p, "beta": np.array([[[-1.0], [2.0]]])},
        space=SampleLabelSpace(["a"]),
        subjects=["s1", "s2"],
        contrasts=["c1"],
    )
    from scipy import stats as sp_stats

    z = derive(ds, "z_from_p").assay("z")
    np.testing.assert_allclose(
        z[0, :, 0], [-sp_stats.norm.isf(5e-21), sp_stats.norm.isf(0.025)], rtol=1e-12
    )

    no_beta = group_dataset(
        {"p": p}, space=SampleLabelSpace(["a"]), subjects=["s1", "s2"], contrasts=["c1"]
    )
    with pytest.raises(AdapterContractError, match="beta"):
        derive(no_beta, "z_from_p")


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


def test_posthoc_dispatches_registered_alias() -> None:
    assert posthoc_registry.is_registered("fdr:by")
    ds = group_dataset(
        {"p": np.array([[[0.01]], [[0.02]], [[0.50]]])},
        space=SampleLabelSpace(["r1", "r2", "r3"]),
        subjects=["s1"],
        contrasts=["c1"],
    )

    out = posthoc(ds, "fdr:by")

    assert out.metadata["posthoc"] == "fdr:by"
    assert "q" in out.assays
    assert np.all(np.isfinite(out.assay("q")))


def test_reduce_and_write_out_are_explicit_phase_gaps() -> None:
    ds = _dataset()
    with pytest.raises(UnsupportedGroupFeatureError, match="fmrigds-r"):
        reduce(ds, method="lmm:ri")
    with pytest.raises(UnsupportedGroupFeatureError, match="format='h5'"):
        write_out(ds, "out.csv", format="csv")


def test_reduce_emits_progress_events() -> None:
    events: list[GroupProgressEvent] = []
    out = reduce(_dataset(), method="meta:fe", progress=events.append)

    assert out.metadata["reduce_method"] == "meta:fe"
    assert [event.stage for event in events] == ["start", "done"]
    assert [event.method for event in events] == ["meta:fe", "meta:fe"]
    assert events[0].completed == 0
    assert events[1].completed == 1
    assert events[1].total == 1


def test_reduce_emits_progress_error_event() -> None:
    events: list[GroupProgressEvent] = []
    with pytest.raises(UnsupportedGroupFeatureError, match="fmrigds-r"):
        reduce(_dataset(), method="lmm:ri", progress=events.append)

    assert [event.stage for event in events] == ["start", "error"]
    assert events[1].method == "lmm:ri"
    assert events[1].metadata["error_type"] == "UnsupportedGroupFeatureError"
