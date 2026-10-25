"""Red checks for native examine_group."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmrimod.design.event_model import EventModel
from fmrimod.group import (
    SampleLabelSpace,
    examination_control,
    examine_group,
    group_dataset,
    meta_fe,
    reduce,
)


def _r_style_fixture():
    n_sample = 40
    n_subject = 10
    signal = np.sin(np.linspace(0, 2 * np.pi, n_sample))
    beta = np.empty((n_sample, n_subject, 1), dtype=np.float64)
    for i in range(8):
        beta[:, i, 0] = signal + 0.03 * np.cos(np.arange(n_sample) + i + 1)
    beta[:, 8, 0] = signal + 3 * np.sin(np.arange(n_sample) * 1.7)
    beta[:, 9, 0] = -signal
    var = np.full_like(beta, 0.04)
    var[:, 7, 0] = 0.001
    var[:, 8, 0] = 9.0
    return group_dataset(
        {"beta": beta, "var": var},
        space=SampleLabelSpace([f"feature-{i}" for i in range(1, n_sample + 1)]),
        subjects=[f"s{i}" for i in range(1, n_subject + 1)],
        contrasts=["task"],
    )


def _two_group_outlier_fixture():
    rng = np.random.default_rng(0)
    n_sample = 12
    n_subject = 8
    groups = ["control"] * 4 + ["patient"] * 4
    beta = rng.normal(0.2, 0.05, size=(n_sample, n_subject, 1))
    beta[:, 4:, 0] += 0.4
    beta[:, 7, 0] = -8.0  # planted patient outlier
    var = np.full_like(beta, 0.05)
    col = pd.DataFrame({"group": pd.Categorical(groups)})
    return group_dataset(
        {"beta": beta, "var": var},
        space=SampleLabelSpace([f"feature-{i}" for i in range(1, n_sample + 1)]),
        subjects=[f"sub-{i:02d}" for i in range(1, n_subject + 1)],
        contrasts=["task"],
        col_data=col,
    )


def test_examine_group_does_not_rewrite_source() -> None:
    ds = _r_style_fixture()
    before = np.array(ds.assay("beta"), copy=True)
    exam = examine_group(ds, method="meta:fe")
    np.testing.assert_array_equal(ds.assay("beta"), before)
    np.testing.assert_array_equal(exam.dataset.assay("beta"), before)
    assert exam.method == "meta:fe"
    assert exam.group_maps.subjects == ("group",)


def test_examination_control_accepts_typed_nested_mappings() -> None:
    control = examination_control(
        geometry={"rank": 4, "stability_replicates": 3},
        review={
            "surprise": {"energy_threshold": 1.5},
            "influence": {"max_abs_threshold": 1.25},
            "quality": {
                "coverage_fraction": {"direction": "low", "threshold": 0.9}
            },
            "min_stability": 0.8,
        },
        tolerance={"rank": 1e-7},
    )

    assert control.geometry.rank == 4
    assert control.geometry.stability_replicates == 3
    assert control.surprise.energy_threshold == 1.5
    assert control.influence.max_abs_threshold == 1.25
    assert control.quality["coverage_fraction"].threshold == 0.9
    assert control.min_stability == 0.8
    assert control.tolerance.rank == 1e-7


def test_examination_control_rejects_unknown_nested_fields() -> None:
    with pytest.raises(TypeError, match="unknown geometry field"):
        examination_control(geometry={"rank": 4, "rnak": 5})


def test_r_fixture_flags_surprise_and_influence() -> None:
    ds = _r_style_fixture()
    exam = examine_group(
        ds,
        method="meta:fe",
        control=examination_control(block_size=7),
    )
    sub = exam.subject_data.set_index("subject")
    assert sub.loc["s10", "review_status"] == "review"
    assert sub.loc["s10", "review_source"] == "surprise"
    assert "negative map gain" in str(sub.loc["s10", "review_reason"])
    assert sub.loc["s8", "review_status"] == "review"
    assert sub.loc["s8", "review_source"] == "influence"
    assert sub.loc["s9", "review_status"] == "none"
    energy = exam.estimand_data.set_index("subject")["influence_energy"]
    assert energy.loc["s9"] < energy.loc["s8"]


def test_homogeneous_cohort_has_no_review_labels() -> None:
    beta = np.ones((60, 12, 1))
    ds = group_dataset(
        {"beta": beta, "var": np.full_like(beta, 0.1)},
        space=SampleLabelSpace([f"v{i}" for i in range(1, 61)]),
        subjects=[f"s{i}" for i in range(1, 13)],
        contrasts=["task"],
    )
    exam = examine_group(ds, method="meta:fe")
    assert (exam.subject_data["review_status"] == "none").all()
    assert exam.subject_data["review_priority"].notna().all()


def test_poor_coverage_is_quality_not_surprise() -> None:
    beta = np.ones((30, 10, 1))
    beta[15:, 0, 0] = np.nan
    ds = group_dataset(
        {"beta": beta, "var": np.full_like(beta, 0.1)},
        space=SampleLabelSpace([f"v{i}" for i in range(1, 31)]),
        subjects=[f"s{i}" for i in range(1, 11)],
        contrasts=["task"],
    )
    exam = examine_group(ds, method="meta:fe")
    row = exam.subject_data.set_index("subject").loc["s1"]
    assert row["coverage_fraction"] == 0.5
    assert row["review_status"] == "review"
    assert row["review_source"] == "quality"
    assert "coverage_fraction" in str(row["review_reason"])


def test_two_group_outlier_ranks_first_and_source_reduce_unchanged() -> None:
    ds = _two_group_outlier_fixture()
    reduced = reduce(ds, method="meta:fe_reg", formula="~ group")
    before = np.array(ds.assay("beta"), copy=True)
    exam = examine_group(
        ds,
        method="meta:fe_reg",
        formula="~ group",
        estimands=["group[T.patient]"],
    )
    np.testing.assert_array_equal(ds.assay("beta"), before)
    assert reduced.assay("coef:group[T.patient]").shape[1] == 1
    ranked = exam.subject_data.sort_values("review_priority", ascending=False)
    assert ranked.iloc[0]["subject"] == "sub-08"
    assert ranked.iloc[0]["review_status"] == "review"


def test_re_refit_does_not_reorder_review_queue() -> None:
    ds = _r_style_fixture()
    exam = examine_group(ds, method="meta:re")
    screening = exam.estimand_data[exam.estimand_data["mode"] == "tau2_fixed_full"]
    refit = exam.estimand_data[exam.estimand_data["mode"] == "tau2_refit_exact"]
    assert not screening.empty
    assert not refit.empty
    order = list(
        exam.subject_data.sort_values("review_priority", ascending=False)["subject"]
    )
    # Refit rows are appended after the queue is assigned.
    assert (
        list(
            exam.subject_data.sort_values("review_priority", ascending=False)["subject"]
        )
        == order
    )


def test_zero_event_docs_name_r_placeholders() -> None:
    assert "placeholder" in (EventModel.__doc__ or "")
    assert "fmridesign_zero_events" in (EventModel.__doc__ or "")


def test_reduced_result_is_rejected() -> None:
    ds = _r_style_fixture()
    out = meta_fe(ds)
    try:
        examine_group(out, method="meta:fe")
    except Exception as exc:
        assert "subject-level" in str(exc)
    else:
        raise AssertionError("expected AdapterContractError for reduced input")
