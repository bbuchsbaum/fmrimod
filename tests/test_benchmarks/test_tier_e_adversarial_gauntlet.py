"""Contract tests for the Tier E adversarial parity gauntlet."""

from __future__ import annotations

import json

import numpy as np
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
    assert survivable["comparisons"]["stat_scale_ratio_median"] == pytest.approx(
        survivable["comparisons"]["expected_stat_scale_ratio_from_dof"]
    )
    assert (
        survivable["comparisons"]["fmrimod_dispersion_denominator_rank_df"]
        == pytest.approx(89.0)
    )
    assert (
        survivable["comparisons"]["nilearn_dispersion_denominator_column_df"]
        == pytest.approx(88.0)
    )
    assert survivable["comparisons"]["contrast_covariance_factor_delta"] < 1e-10
    assert survivable["fmrimod"]["ill_conditioned"] is True
    assert survivable["fmrimod"]["aliased_columns"]
    assert survivable["nilearn"]["aliased_columns"] == []
    assert "not covariance pseudoinverse choice" in survivable["verdict"]

    wide = cases["zero_residual_dof_wide_design"]
    assert wide["status"] == "pass"
    assert wide["design_rank"] == wide["design_shape"][0]
    assert wide["fmrimod"]["df_residual"] == 0.0
    assert wide["nilearn"]["df_residual"] == 0.0
    assert wide["fmrimod"]["undefined_t_policy"] == "nan_t"
    assert wide["nilearn"]["undefined_t_policy"] == "nan_t"
    assert wide["fmrimod"]["finite_stat_fraction"] == 0.0
    assert wide["nilearn"]["finite_stat_fraction"] == 0.0
    assert wide["fmrimod"]["nan_se_fraction"] == 1.0
    assert wide["nilearn"]["warning_messages"]
    assert "both engines expose undefined" in wide["verdict"]


def test_rank_deficient_tstat_drift_is_dof_convention() -> None:
    """Pin the algebra behind the Tier E rank-deficient t-stat scale delta."""

    inputs = workflow._make_survivable_inputs(max_voxels=12, seed=20260513)
    fmrimod_probe, _, fmrimod_stat = workflow.fmrimod_probe(inputs)
    nilearn_probe, _, nilearn_stat = workflow.nilearn_probe(inputs)

    assert fmrimod_probe.df_residual == nilearn_probe.df_residual == 89.0
    assert inputs.design.shape == (96, 8)

    finite = (
        np.isfinite(fmrimod_stat)
        & np.isfinite(nilearn_stat)
        & (np.abs(nilearn_stat) > np.finfo(np.float64).eps)
    )
    ratio = fmrimod_stat[finite] / nilearn_stat[finite]

    # Nilearn reports residual DoF as n-rank, but its OLS dispersion path
    # divides RSS by n-p. fmrimod uses n-rank in both places. That is the
    # entire t-stat scale delta for this rank-deficient but estimable contrast.
    expected = np.sqrt((96 - 7) / (96 - 8))
    assert np.min(ratio) == pytest.approx(expected)
    assert np.max(ratio) == pytest.approx(expected)

    diagnostics = workflow._rank_deficient_stat_scale_diagnostics(
        inputs,
        fmrimod_probe,
    )
    assert diagnostics["contrast_covariance_factor_delta"] < 1e-10


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
