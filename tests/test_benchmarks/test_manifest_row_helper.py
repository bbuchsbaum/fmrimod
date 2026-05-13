"""Tests for ``scripts/manifest_row.py`` — the proof-artifact insertion helper.

Uses temp manifests so adding/validating rows here does not race the
canonical ``proof_artifacts.json`` while parallel turns hold edits to it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts"

sys.path.insert(0, str(SCRIPT_DIR))
try:
    from manifest_row import add_row, validate_row
finally:
    sys.path.pop(0)


def _good_canary_row() -> dict:
    return {
        "benchmark_id": "tier_x_synthetic",
        "evidence_level": "numerical_canary",
        "public_seam": False,
        "workflow_path": "benchmarks/parity/tier_x_synthetic/workflow.py",
        "report_path": "benchmarks/parity/tier_x_synthetic/reports/parity_report.json",
        "reference_path": "benchmarks/parity/tier_x_synthetic/workflow.py::reference",
        "fmrimod_path": "benchmarks/parity/tier_x_synthetic/workflow.py::fmrimod",
        "caveats": [],
        "replacement_target": "tier_x_public_seam",
    }


def _good_public_seam_row() -> dict:
    return {
        "benchmark_id": "tier_x_public",
        "evidence_level": "workflow_parity",
        "public_seam": True,
        "workflow_path": "benchmarks/parity/tier_x_public/workflow.py",
        "report_path": "benchmarks/parity/tier_x_public/reports/parity_report.json",
        "reference_path": "benchmarks/parity/tier_x_public/workflow.py::nilearn",
        "fmrimod_path": "benchmarks/parity/tier_x_public/workflow.py::fmrimod",
        "caveats": [],
        "typed_objects": ["fmrimod.glm.FmriLm"],
        "fmrimod_expresses_better": "Carries typed contrast intent.",
        "timings": {"status": "not_recorded"},
        "receipt": {
            "status": "regenerable",
            "command": ["python", "-m", "benchmarks.parity.tier_x_public.workflow"],
            "checks": ["report_path", "status", "caveats"],
        },
    }


def _empty_manifest() -> dict:
    return {"schema_version": "parity-proof-artifacts/v1", "artifacts": []}


def test_validate_good_canary_row_passes() -> None:
    assert validate_row(_good_canary_row()) == []


def test_validate_good_public_seam_row_passes() -> None:
    assert validate_row(_good_public_seam_row()) == []


def test_validate_reports_missing_required_fields() -> None:
    row = _good_canary_row()
    del row["benchmark_id"]
    del row["caveats"]
    errors = validate_row(row)
    assert any("benchmark_id" in e for e in errors)
    assert any("caveats" in e for e in errors)


def test_validate_rejects_unknown_evidence_level() -> None:
    row = _good_canary_row()
    row["evidence_level"] = "made_up"
    assert any("evidence_level" in e for e in validate_row(row))


def test_validate_public_seam_requires_typed_objects_and_better() -> None:
    row = _good_public_seam_row()
    del row["typed_objects"]
    del row["fmrimod_expresses_better"]
    errors = validate_row(row)
    assert any("typed_objects" in e for e in errors)
    assert any("fmrimod_expresses_better" in e for e in errors)


def test_validate_public_seam_requires_recorded_or_not_recorded_timings() -> None:
    row = _good_public_seam_row()
    row["timings"] = {"status": "kinda_recorded"}
    assert any("timings.status" in e for e in validate_row(row))


def test_validate_flagship_requires_receipt() -> None:
    row = _good_public_seam_row()
    row["evidence_level"] = "flagship_workflow"
    del row["receipt"]
    assert any("receipt" in e for e in validate_row(row))


def test_validate_regenerable_receipt_requires_command_list() -> None:
    row = _good_public_seam_row()
    row["receipt"]["command"] = "python -m foo"  # string, not list
    assert any("command" in e for e in validate_row(row))


def test_validate_regenerable_receipt_requires_canonical_checks() -> None:
    row = _good_public_seam_row()
    row["receipt"]["checks"] = ["report_path"]  # missing status, caveats
    errors = validate_row(row)
    assert any("status" in e for e in errors)
    assert any("caveats" in e for e in errors)


def test_validate_static_snapshot_receipt_requires_metadata() -> None:
    row = _good_public_seam_row()
    row["receipt"] = {"status": "static_snapshot"}
    errors = validate_row(row)
    assert any("reason" in e for e in errors)
    assert any("date" in e for e in errors)
    assert any("owner" in e for e in errors)


def test_validate_static_snapshot_receipt_must_not_carry_command() -> None:
    row = _good_public_seam_row()
    row["receipt"] = {
        "status": "static_snapshot",
        "reason": "external fixture",
        "date": "2026-05-13",
        "owner": "bd-01TEST",
        "command": ["python", "-m", "should_not_be_here"],
    }
    assert any("command" in e for e in validate_row(row))


def test_validate_canary_requires_replacement_target() -> None:
    row = _good_canary_row()
    del row["replacement_target"]
    assert any("replacement_target" in e for e in validate_row(row))


def test_validate_canary_must_not_be_public_seam() -> None:
    row = _good_canary_row()
    row["public_seam"] = True
    # also needs the public-seam fields when public_seam is true; we expect
    # at least one error specifically about the canary contradiction
    assert any("numerical_canary" in e and "public_seam" in e for e in validate_row(row))


def test_validate_detects_duplicate_benchmark_id() -> None:
    manifest = _empty_manifest()
    row = _good_canary_row()
    manifest["artifacts"].append(row)
    duplicate = _good_canary_row()
    errors = validate_row(duplicate, manifest=manifest)
    assert any("already in manifest" in e for e in errors)


def test_add_row_inserts_sorted_and_writes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    initial = _empty_manifest()
    initial["artifacts"].append({**_good_canary_row(), "benchmark_id": "tier_z_other"})
    manifest_path.write_text(json.dumps(initial, indent=2) + "\n")

    new_row = _good_canary_row()  # benchmark_id="tier_x_synthetic"
    merged = add_row(manifest_path, new_row)
    on_disk = json.loads(manifest_path.read_text())

    assert merged == on_disk
    ids = [a["benchmark_id"] for a in on_disk["artifacts"]]
    assert ids == sorted(ids), f"artifacts not sorted: {ids}"
    assert "tier_x_synthetic" in ids
    assert "tier_z_other" in ids


def test_add_row_dry_run_does_not_write(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    original = _empty_manifest()
    manifest_path.write_text(json.dumps(original, indent=2) + "\n")

    add_row(manifest_path, _good_canary_row(), write=False)

    on_disk = json.loads(manifest_path.read_text())
    assert on_disk == original, "dry-run should not write the manifest"


def test_add_row_refuses_invalid_row(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_empty_manifest(), indent=2) + "\n")

    bad = _good_canary_row()
    del bad["benchmark_id"]
    with pytest.raises(ValueError, match="benchmark_id"):
        add_row(manifest_path, bad)


def test_add_row_refuses_duplicate(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = _empty_manifest()
    manifest["artifacts"].append(_good_canary_row())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    with pytest.raises(ValueError, match="already in manifest"):
        add_row(manifest_path, _good_canary_row())


def test_helper_validates_canonical_manifest_rows() -> None:
    """Every existing artifact in the canonical manifest passes the helper.

    This is the integration check: if the helper's per-row validation
    diverges from the rules the test suite enforces, the canonical
    manifest stops being a valid example. Catching that here prevents
    the helper from drifting silently.
    """
    canonical = REPO_ROOT / "benchmarks" / "parity" / "proof_artifacts.json"
    if not canonical.exists():
        pytest.skip("canonical manifest not present")
    payload = json.loads(canonical.read_text())
    for row in payload.get("artifacts", []):
        errors = validate_row(row)
        assert not errors, (
            f"{row.get('benchmark_id')!r} fails the helper's validation:\n  - "
            + "\n  - ".join(errors)
        )
