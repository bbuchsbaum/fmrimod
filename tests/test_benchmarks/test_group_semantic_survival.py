"""Checks for the group-facing semantic-survival flagship demo."""

from __future__ import annotations

import json

import pytest

from benchmarks.parity.tier_group_semantic_survival import workflow

pytest.importorskip("nilearn")


def test_group_semantic_survival_carries_typed_intent_to_group() -> None:
    result = workflow.run_demo(n_subjects=4, max_voxels=8)

    assert result.group.contrasts == ("conditions_omnibus",)
    assert result.group.metadata["semantic_survival"] is True
    assert result.group.contrast_data is not None
    contrast_row = result.group.contrast_data.iloc[0]
    assert contrast_row["intent_kind"] == "omnibus"
    assert contrast_row["intent_term"] == "trial_type"
    assert contrast_row["intent_levels"] == "condition_a,condition_b"
    assert contrast_row["statistic_family"] == "F"
    assert "condition_a" in contrast_row["touched_columns"]
    assert "condition_b" in contrast_row["touched_columns"]

    assert result.group.assay("beta").shape == (8, 4, 1)
    assert result.group_result.assay("t_coef:Intercept").shape == (8, 1, 1)
    assert all(
        explanation["intent"]["term"] == "trial_type"
        for explanation in result.explanations
    )


def test_group_semantic_survival_report_is_regenerable(tmp_path) -> None:
    workflow.main(["--out-dir", str(tmp_path)])

    report = json.loads((tmp_path / "semantic_survival_report.json").read_text())
    assert report["status"] == "pass"
    assert report["caveats"] == []
    assert report["checks"]["typed_intent_kind"] == "omnibus"
    assert report["checks"]["typed_intent_term"] == "trial_type"
    assert report["checks"]["contrast_name_survives"] == ["conditions_omnibus"]
    assert "t_coef:Intercept" in report["checks"]["group_result_assays"]
