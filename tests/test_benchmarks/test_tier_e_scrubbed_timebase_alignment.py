"""Contract tests for the Tier E scrubbed-timebase alignment benchmark."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from benchmarks.parity.tier_e_scrubbed_timebase_alignment import workflow

pytest.importorskip("nilearn")

ROOT = Path(__file__).resolve().parents[2]


def test_scrubbed_timebase_records_alignment_and_compaction_trap() -> None:
    report = workflow.run_benchmark(max_voxels=12)

    assert report["schema_version"] == "scrubbed-timebase-alignment/v1"
    assert report["status"] == "pass"
    assert report["alignment_contract"]["censor_policy"] == (
        "row_subset_original_timebase"
    )
    assert report["alignment_contract"]["original_timebase_rows"] > (
        report["alignment_contract"]["kept_rows"]
    )
    assert report["fmrimod"]["status"] == "ok"
    assert report["nilearn_aligned"]["status"] == "ok"
    assert report["nilearn_compressed_timebase"]["status"] == "ok"
    assert report["fmrimod"]["touched_columns"] == [
        "trial_type_trial_type.A",
        "trial_type_trial_type.B",
    ]
    assert report["comparisons"]["aligned_effect_max_abs_delta"] < 1e-8
    assert report["comparisons"]["aligned_stat_max_abs_delta"] < 1e-6
    assert report["comparisons"]["compressed_timebase_effect_median_abs_delta"] > 0.15
    assert report["comparisons"]["compressed_timebase_stat_median_abs_delta"] > 0.10
    assert report["pain_point"]["observed"] is True
    assert report["pain_point"]["max_event_onset_shift_seconds"] >= 70.0
    assert "right row count" in report["pain_point"]["verdict"]


def test_scrubbed_timebase_uses_public_fmrimod_seam() -> None:
    source = inspect.getsource(workflow.fmrimod_pipeline)

    assert "fm.fmri_dataset" in source
    assert "censor=inputs.censor" in source
    assert "fm.fmri_lm" in source
    assert "inputs.semantic_contrast" in source


def test_scrubbed_timebase_manifest_is_public_workflow_parity() -> None:
    manifest = json.loads((ROOT / "benchmarks/parity/proof_artifacts.json").read_text())
    row = next(
        item
        for item in manifest["artifacts"]
        if item["benchmark_id"] == "tier_e_scrubbed_timebase_alignment"
    )

    assert row["evidence_level"] == "workflow_parity"
    assert row["public_seam"] is True
    assert "replacement_target" not in row
    assert "fmri_dataset(censor=...)" in row["fmrimod_path"]
    assert "fmrimod.contrast.SemanticContrast" in row["typed_objects"]
    assert "compressed-timebase" in row["fmrimod_expresses_better"]
    assert row["receipt"]["status"] == "regenerable"


def test_scrubbed_timebase_main_writes_report(tmp_path: Path) -> None:
    workflow.main(["--out-dir", str(tmp_path), "--max-voxels", "8"])

    report_path = tmp_path / "scrubbed_timebase_alignment_report.json"
    markdown_path = tmp_path / "REPORT.md"
    report = json.loads(report_path.read_text())
    assert report["status"] == "pass"
    assert "Scrubbed Timebase Alignment" in markdown_path.read_text()
