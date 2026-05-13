"""Contract tests for the Tier E parametric-centering stress benchmark."""

from __future__ import annotations

import json

import pytest

from benchmarks.parity.tier_e_parametric_centering import workflow

pytest.importorskip("nilearn")


def test_parametric_centering_stress_records_parity_and_pain_point() -> None:
    report = workflow.run_benchmark(max_voxels=12)

    assert report["schema_version"] == "parametric-centering-stress/v1"
    assert report["status"] == "pass"
    assert report["rt_means"]["A_minus_B"] > 0.6
    assert report["pain_points"]["observed"] is True
    assert report["pain_points"]["main_effect_shift_median"] > 0.10
    assert "main effect changed" in report["pain_points"]["verdict"]

    cases = {case["case_id"]: case for case in report["cases"]}
    assert set(cases) == {
        "within_condition_centering",
        "global_centering_with_imbalanced_rt",
    }

    for case in cases.values():
        assert case["status"] == "pass"
        assert case["design_rank"] == case["design_shape"][1]
        assert case["comparisons"]["main_effect_delta"] < 1e-7
        assert case["comparisons"]["main_stat_delta"] < 1e-5
        assert case["comparisons"]["slope_effect_delta"] < 1e-7
        assert case["comparisons"]["slope_stat_delta"] < 1e-5
        assert case["fmrimod"]["finite_main_stat_fraction"] == 1.0
        assert case["nilearn"]["finite_main_stat_fraction"] == 1.0
        assert "agree on the realised design" in case["verdict"]

    within = cases["within_condition_centering"]
    global_case = cases["global_centering_with_imbalanced_rt"]
    assert (
        within["fmrimod"]["main_effect_median"]
        > global_case["fmrimod"]["main_effect_median"]
    )


def test_parametric_centering_main_writes_report(tmp_path) -> None:
    workflow.main(["--out-dir", str(tmp_path), "--max-voxels", "8"])

    report_path = tmp_path / "centering_stress_report.json"
    markdown_path = tmp_path / "REPORT.md"
    report = json.loads(report_path.read_text())
    assert report["status"] == "pass"
    assert report["pain_points"]["main_effect_shift_median"] > 0.10
    assert "Parametric Centering Stress" in markdown_path.read_text()
