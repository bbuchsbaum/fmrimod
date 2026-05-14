"""Contract tests for the BIDS parametric group follow-through scenario."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from benchmarks.parity.tier_e_bids_parametric_group_followthrough import workflow

ROOT = Path(__file__).resolve().parents[2]


def test_bids_parametric_group_followthrough_is_full_ux_path() -> None:
    report = workflow.run_benchmark(n_subjects=7, n_voxels=8)

    assert report["schema_version"] == "bids-parametric-group-followthrough/v1"
    assert report["status"] == "pass"
    assert report["caveats"] == []
    assert report["ergonomics"]["ux_status"] == "full"
    assert report["ergonomics"]["raw_vector_user_code"] is False
    assert report["checks"] == {
        "bids_main_semantic": True,
        "parametric_slope_semantic": True,
        "group_behavior_positive": True,
        "group_behavior_detected": True,
    }
    assert {
        item["main_kind"] for item in report["subject_summary"]
    } == {"semantic_linear_contrast"}
    assert {item["slope_kind"] for item in report["subject_summary"]} == {
        "semantic_contrast"
    }
    assert report["group"]["behavior_coef_median"] > 0.2
    assert report["group"]["behavior_t_median"] > 5.0


def test_bids_parametric_group_followthrough_callsite_has_no_raw_vectors() -> None:
    source = inspect.getsource(workflow.fit_subject)
    assert 'condition("word", term="trial_type:rt_z")' in source
    assert '"pseudoword"' in source
    assert "np.zeros" not in source
    assert "fit.contrast(" in source
    assert "slope_spec" in source

    group_source = inspect.getsource(workflow.group_followthrough)
    assert 'group_model("behavior")' in group_source
    assert "formula=" not in group_source


def test_bids_parametric_group_followthrough_manifest_row() -> None:
    manifest = json.loads(
        (ROOT / "benchmarks/parity/proof_artifacts.json").read_text()
    )
    row = next(
        item
        for item in manifest["artifacts"]
        if item["benchmark_id"] == "tier_e_bids_parametric_group_followthrough"
    )

    assert row["evidence_level"] == "workflow_parity"
    assert row["public_seam"] is True
    assert row["ux_status"] == "full"
    assert row["ux_blockers"] == []
    assert row["claim_axes"]["typed_contrast_authoring"] == "full"
    assert "condition(term='trial_type:rt_z')" in row["fmrimod_path"]
    assert "GroupLinearModel" in row["typed_objects"]


def test_bids_parametric_group_followthrough_main_writes_report(tmp_path) -> None:
    workflow.main(["--out-dir", str(tmp_path), "--n-subjects", "7", "--n-voxels", "8"])

    report_path = tmp_path / "followthrough_report.json"
    markdown_path = tmp_path / "REPORT.md"
    report = json.loads(report_path.read_text())
    assert report["status"] == "pass"
    assert report["ergonomics"]["ux_status"] == "full"
    assert "BIDS Parametric Group Follow-through" in markdown_path.read_text()
