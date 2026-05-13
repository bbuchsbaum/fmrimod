"""Tests for parity proof-artifact manifest row tooling."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.parity_artifact import insert_row, validate_row

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "parity_artifact.py"


def _manifest() -> dict:
    return {
        "schema_version": "parity-proof-artifacts/v1",
        "artifacts": [
            {
                "benchmark_id": "existing_case",
                "evidence_level": "numerical_canary",
                "public_seam": False,
                "workflow_path": "benchmarks/parity/existing_case/workflow.py",
                "report_path": "benchmarks/parity/existing_case/reports/report.json",
                "reference_path": "existing::reference",
                "fmrimod_path": "existing::fmrimod",
                "caveats": [],
                "replacement_target": "public workflow",
            }
        ],
    }


def _row() -> dict:
    return {
        "benchmark_id": "new_case",
        "evidence_level": "numerical_canary",
        "public_seam": False,
        "workflow_path": "benchmarks/parity/new_case/workflow.py",
        "report_path": "benchmarks/parity/new_case/reports/report.json",
        "reference_path": "new_case::reference",
        "fmrimod_path": "new_case::fmrimod",
        "caveats": [],
        "replacement_target": "promote to public seam",
    }


def test_insert_row_validates_and_leaves_input_manifest_unchanged() -> None:
    manifest = _manifest()
    row = _row()

    validate_row(manifest, row)
    updated = insert_row(manifest, row)

    assert [item["benchmark_id"] for item in manifest["artifacts"]] == [
        "existing_case"
    ]
    assert [item["benchmark_id"] for item in updated["artifacts"]] == [
        "existing_case",
        "new_case",
    ]


def test_insert_row_rejects_duplicate_benchmark_id() -> None:
    manifest = _manifest()
    duplicate = _row() | {"benchmark_id": "existing_case"}

    try:
        validate_row(manifest, duplicate)
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("duplicate benchmark_id should be rejected")


def test_cli_writes_candidate_manifest_to_output_path(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    row_path = tmp_path / "row.json"
    output_path = tmp_path / "candidate.json"
    manifest_path.write_text(json.dumps(_manifest()))
    row_path.write_text(json.dumps(_row()))

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "insert-row",
            "--manifest",
            str(manifest_path),
            "--row",
            str(row_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        check=True,
    )

    assert "new_case" not in manifest_path.read_text()
    candidate = json.loads(output_path.read_text())
    assert candidate["artifacts"][-1]["benchmark_id"] == "new_case"
