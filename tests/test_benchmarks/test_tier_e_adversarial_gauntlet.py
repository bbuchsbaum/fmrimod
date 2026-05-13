"""Contract tests for the Tier E adversarial parity gauntlet."""

from __future__ import annotations

import json

import pytest

from benchmarks.parity.tier_e_adversarial_gauntlet import workflow

pytest.importorskip("nilearn")


def test_gauntlet_records_recovery_and_failure_boundaries() -> None:
    report = workflow.run_gauntlet(max_voxels=12)

    assert report["schema_version"] == "adversarial-gauntlet/v1"
    assert report["status"] == "pass"
    cases = {case["case_id"]: case for case in report["cases"]}

    survivable = cases["survivable_rank_deficiency"]
    assert survivable["status"] == "pass"
    assert survivable["design_rank"] < survivable["design_shape"][1]
    assert survivable["comparisons"]["max_abs_effect_delta"] < 1e-8
    assert survivable["comparisons"]["stat_pearson"] > 0.999999
    assert survivable["comparisons"]["max_abs_stat_delta"] > 0.0
    assert survivable["fmrimod"]["ill_conditioned"] is True
    assert survivable["fmrimod"]["aliased_columns"]
    assert survivable["nilearn"]["aliased_columns"] == []
    assert "diagnostics" in survivable["verdict"]

    wide = cases["zero_residual_dof_wide_design"]
    assert wide["status"] == "boundary_observed"
    assert wide["design_rank"] == wide["design_shape"][0]
    assert wide["fmrimod"]["df_residual"] == 0.0
    assert wide["nilearn"]["df_residual"] == 0.0
    assert wide["fmrimod"]["undefined_t_policy"] == "zero_filled_t_with_nan_se"
    assert wide["nilearn"]["undefined_t_policy"] == "nan_t"
    assert wide["fmrimod"]["finite_stat_fraction"] == 1.0
    assert wide["nilearn"]["finite_stat_fraction"] == 0.0
    assert wide["fmrimod"]["nan_se_fraction"] == 1.0
    assert wide["nilearn"]["warning_messages"]
    assert "split boundary" in wide["verdict"]


def test_gauntlet_main_writes_report(tmp_path) -> None:
    workflow.main(["--out-dir", str(tmp_path), "--max-voxels", "8"])

    report_path = tmp_path / "adversarial_gauntlet_report.json"
    markdown_path = tmp_path / "REPORT.md"
    report = json.loads(report_path.read_text())
    assert report["status"] == "pass"
    assert {case["case_id"] for case in report["cases"]} == {
        "survivable_rank_deficiency",
        "zero_residual_dof_wide_design",
    }
    assert "zero_residual_dof_wide_design" in markdown_path.read_text()
