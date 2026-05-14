"""Tests for the 1.0 release proof-bundle receipt."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from benchmarks.parity import release_1_0_bundle as bundle


def test_release_manifest_maps_each_flagship_family_to_a_real_artifact() -> None:
    manifest = json.loads(bundle.RELEASE_MANIFEST.read_text())
    artifacts = json.loads(bundle.PROOF_ARTIFACTS.read_text())
    artifact_ids = {row["benchmark_id"] for row in artifacts["artifacts"]}

    assert manifest["schema_version"] == "release_1_0_manifest/v1"
    assert manifest["receipt_command"] == [
        "python",
        "-m",
        "benchmarks.parity.release_1_0_bundle",
    ]
    families = manifest["flagship_families"]
    assert {row["family"] for row in families} == {
        "SPM auditory / first-level modeling",
        "FIAC/localizer fixed effects",
        "FitLins/BIDS Stats Model translation",
        "second-level/group inference",
        "single-trial/LSS or trialwise estimation",
        "typed proof scorecard / underdog showcase",
    }
    for row in families:
        assert bundle.REQUIRED_MAPPING_KEYS <= set(row)
        assert row["benchmark_id"] in artifact_ids
        assert row["required_public_path"]
        assert row["current_status"]
        assert row["owner_bead"].startswith("bd-")


def test_release_receipt_is_currently_blocked_by_named_red_checks() -> None:
    receipt = bundle.build_receipt()

    assert receipt["schema_version"] == "release_1_0_receipt/v1"
    assert receipt["release_status"] == "blocked"
    blockers = "\n".join(receipt["blockers"])
    assert "tier_a_fiac: public_seam is not true" in blockers
    assert "tier_a_fiac: numerical_canary cannot be flagship proof" in blockers
    assert "tier_b_fitlins_bids: public_seam is not true" in blockers
    assert "tier_d_lss_trialwise: numerical_canary cannot be flagship proof" in blockers
    assert "api spine fmri_dataset: still review_pending" in blockers


def test_release_receipt_carries_gate_files_and_api_spine_evidence() -> None:
    receipt = bundle.build_receipt()

    gate_paths = {row["path"] for row in receipt["evidence_gates"]}
    assert set(bundle.GATE_FILES) == gate_paths
    assert all(row["exists"] for row in receipt["evidence_gates"])

    spine = {row["name"]: row for row in receipt["api_spine"]}
    assert set(bundle.API_SPINE_NAMES) == set(spine)
    assert spine["fmri_dataset"]["tier"] == "spine"
    assert spine["fmri_dataset"]["compatibility_status"] == "review_pending"


def test_release_receipt_writes_canonical_json(tmp_path: Path) -> None:
    output = tmp_path / "release_receipt.json"

    payload = bundle.write_receipt(output)

    assert output.exists()
    on_disk = json.loads(output.read_text())
    assert on_disk == payload
    assert output.read_text().endswith("\n")


def test_release_bundle_module_command_writes_requested_output(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.parity.release_1_0_bundle",
            "--output",
            str(output),
        ],
        cwd=bundle.ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text())
    assert payload["release_status"] == "blocked"
    assert json.loads(result.stdout)["source"] == payload["source"]


def test_release_bundle_strict_mode_exits_nonzero_while_blocked(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.parity.release_1_0_bundle",
            "--output",
            str(output),
            "--strict",
        ],
        cwd=bundle.ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert output.exists()
    assert json.loads(output.read_text())["release_status"] == "blocked"
