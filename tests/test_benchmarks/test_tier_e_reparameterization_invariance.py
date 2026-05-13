"""Contract tests for the Tier E reparameterization-invariance canary."""

from __future__ import annotations

import json

import pytest

from benchmarks.parity.tier_e_reparameterization_invariance import workflow

pytest.importorskip("nilearn")


def test_reparameterization_canary_records_invariance_and_boundary() -> None:
    report = workflow.run_benchmark(max_voxels=12)

    assert report["schema_version"] == "reparameterization-invariance/v1"
    assert report["status"] == "pass"
    cases = {case["case_id"]: case for case in report["cases"]}

    moderate = cases["moderate_scale_reparameterization"]
    assert moderate["status"] == "pass"
    assert moderate["fmrimod"]["base_rank"] == moderate["fmrimod"]["transformed_rank"]
    assert moderate["nilearn"]["base_rank"] == moderate["nilearn"]["transformed_rank"]
    assert moderate["comparisons"]["fmrimod_effect_delta"] < 1e-8
    assert moderate["comparisons"]["fmrimod_stat_delta"] < 1e-5
    assert moderate["comparisons"]["nilearn_effect_delta"] < 1e-8
    assert moderate["comparisons"]["nilearn_stat_delta"] < 1e-5
    assert "preserve the hypothesis" in moderate["verdict"]

    extreme = cases["extreme_scale_rank_boundary"]
    assert extreme["status"] == "boundary_observed"
    assert (
        extreme["fmrimod"]["transformed_rank"]
        < extreme["fmrimod"]["base_rank"]
    )
    assert (
        extreme["nilearn"]["transformed_rank"]
        < extreme["nilearn"]["base_rank"]
    )
    assert extreme["comparisons"]["fmrimod_effect_delta"] > 1e-3
    assert extreme["comparisons"]["nilearn_effect_delta"] > 1e-3
    assert extreme["comparisons"]["scaled_cross_engine_effect_delta"] < 1e-8
    assert "numerical rank" in extreme["verdict"]


def test_reparameterization_canary_main_writes_report(tmp_path) -> None:
    workflow.main(["--out-dir", str(tmp_path), "--max-voxels", "8"])

    report_path = tmp_path / "reparameterization_invariance_report.json"
    markdown_path = tmp_path / "REPORT.md"
    report = json.loads(report_path.read_text())
    assert report["status"] == "pass"
    assert {case["case_id"] for case in report["cases"]} == {
        "moderate_scale_reparameterization",
        "extreme_scale_rank_boundary",
    }
    assert "Reparameterization Invariance" in markdown_path.read_text()
