"""Contract tests for the Tier E semantic contrast-alignment stress benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.parity.tier_e_semantic_contrast_alignment import workflow

pytest.importorskip("nilearn")

ROOT = Path(__file__).resolve().parents[2]


def test_semantic_contrast_alignment_uses_existing_matrix_surface() -> None:
    assert not hasattr(workflow, "_MatrixDataset")
    assert not hasattr(workflow, "_NamedMatrixModel")
    report = workflow.run_benchmark(max_voxels=4)
    assert report["ergonomics"]["status"] == "matrix-first partial"
    assert (
        "fit_glm_from_matrix(named DataFrame)"
        in report["ergonomics"]["fmrimod_path"]
    )
    assert (
        "condition('gain', term='trial_type')"
        in report["ergonomics"]["authored_term_prototype"]
    )
    assert "not yet the flagship" in report["ergonomics"]["limitation"]
    assert "matched within tolerance" in report["win_ladder"][
        "numerical_oracle_status"
    ]
    assert "fails visibly" in report["win_ladder"]["positional_trap_status"]
    assert (
        "authored_term_prototype_matrix_first"
        in report["win_ladder"]["ergonomic_win_status"]
    )


def test_semantic_contrast_manifest_is_matrix_first_not_full_public_seam() -> None:
    manifest = json.loads(
        (ROOT / "benchmarks/parity/proof_artifacts.json").read_text()
    )
    row = next(
        item
        for item in manifest["artifacts"]
        if item["benchmark_id"] == "tier_e_semantic_contrast_alignment"
    )

    assert row["evidence_level"] == "workflow_parity"
    assert row["public_seam"] is False
    assert "matrix-first partial" in row["fmrimod_expresses_better"]
    assert "fit_glm_from_matrix" in row["fmrimod_path"]
    assert row["replacement_target"]["owner_bead"] == (
        "bd-01KRJ3J97HYXK9DV0EKV0HVMYX"
    )
    assert "fmri_dataset -> fmri_lm" in row["replacement_target"]["blocking_api_gap"]


def test_semantic_contrast_alignment_records_invariance_and_pain_point() -> None:
    report = workflow.run_benchmark(max_voxels=12)

    assert report["schema_version"] == "semantic-contrast-alignment/v1"
    assert report["status"] == "pass"
    assert (
        report["design_column_orders"]["canonical"]
        != report["design_column_orders"]["permuted"]
    )
    assert report["invariance"]["fmrimod_effect_delta"] < 1e-8
    assert report["invariance"]["fmrimod_stat_delta"] < 1e-5
    assert report["invariance"]["fmrimod_authored_effect_delta"] < 1e-8
    assert report["invariance"]["fmrimod_authored_stat_delta"] < 1e-5
    assert report["pain_points"]["observed"] is True
    assert report["pain_points"]["nilearn_positional_effect_median_abs_delta"] > 10.0
    assert "targets different columns" in report["pain_points"]["verdict"]

    cases = {case["case_id"]: case for case in report["cases"]}
    assert set(cases) == {"canonical_order", "permuted_order"}
    for case in cases.values():
        assert case["status"] == "pass"
        assert case["design_rank"] == case["design_shape"][1]
        assert case["fmrimod"]["finite_effect_fraction"] == 1.0
        assert case["fmrimod"]["finite_stat_fraction"] == 1.0
        assert set(case["fmrimod"]["touched_columns"]) == {"gain", "loss"}
        assert set(case["fmrimod_authored"]["touched_columns"]) == {"gain", "loss"}
        assert case["comparisons"]["aligned_effect_delta"] < 1e-8
        assert case["comparisons"]["aligned_stat_delta"] < 1e-5
        assert case["comparisons"]["authored_effect_delta"] < 1e-8
        assert case["comparisons"]["authored_stat_delta"] < 1e-5
        assert "authored condition contrast" in case["verdict"]

    canonical = cases["canonical_order"]
    permuted = cases["permuted_order"]
    assert canonical["comparisons"]["positional_effect_median_abs_delta"] < 1e-8
    assert permuted["nilearn_positional_reuse"]["touched_columns"] != [
        "gain",
        "loss",
    ]
    assert permuted["comparisons"]["positional_stat_median_abs_delta"] > 10.0


def test_semantic_contrast_alignment_main_writes_report(tmp_path) -> None:
    workflow.main(["--out-dir", str(tmp_path), "--max-voxels", "8"])

    report_path = tmp_path / "semantic_contrast_alignment_report.json"
    markdown_path = tmp_path / "REPORT.md"
    report = json.loads(report_path.read_text())
    assert report["status"] == "pass"
    assert report["pain_points"]["observed"] is True
    assert "Semantic Contrast Alignment" in markdown_path.read_text()
