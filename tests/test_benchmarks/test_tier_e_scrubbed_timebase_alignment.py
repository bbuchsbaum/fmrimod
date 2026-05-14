"""Contract tests for the Tier E scrubbed-timebase alignment canary."""

from __future__ import annotations

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
    assert report["n_scans_original"] > report["n_scans_kept"]
    assert report["fmrimod"]["status"] == "ok"
    assert report["nilearn_aligned"]["status"] == "ok"
    assert report["compacted_timebase"]["status"] == "ok"
    assert report["fmrimod"]["touched_columns"] == ["task"]
    assert report["comparisons"]["aligned_effect_delta"] < 1e-8
    assert report["comparisons"]["aligned_stat_delta"] < 1e-5
    assert (
        report["comparisons"]["compacted_timebase_effect_median_abs_delta"] > 0.15
    )
    assert report["comparisons"]["compacted_timebase_stat_median_abs_delta"] > 5.0
    assert report["pain_point"]["observed"] is True
    assert "right row count" in report["pain_point"]["verdict"]


def test_scrubbed_timebase_manifest_is_canary_not_public_seam() -> None:
    manifest = json.loads(
        (ROOT / "benchmarks/parity/proof_artifacts.json").read_text()
    )
    row = next(
        item
        for item in manifest["artifacts"]
        if item["benchmark_id"] == "tier_e_scrubbed_timebase_alignment"
    )

    assert row["evidence_level"] == "numerical_canary"
    assert row["public_seam"] is False
    assert "scrubbed rows" in row["fmrimod_expresses_better"]
    assert "scrubbed timebase" in row["replacement_target"]["description"]
    assert row["replacement_target"]["owner_bead"] == (
        "bd-01KRJ36AJD1XQEVM23356BSXV1"
    )


def test_scrubbed_timebase_main_writes_report(tmp_path) -> None:
    workflow.main(["--out-dir", str(tmp_path), "--max-voxels", "8"])

    report_path = tmp_path / "scrubbed_timebase_alignment_report.json"
    markdown_path = tmp_path / "REPORT.md"
    report = json.loads(report_path.read_text())
    assert report["status"] == "pass"
    assert "Scrubbed Timebase Alignment" in markdown_path.read_text()
