"""Schema tests for parity proof-artifact classification."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks" / "parity" / "proof_artifacts.json"
CAVEATS = ROOT / "docs" / "contracts" / "CAVEATS.md"

EVIDENCE_LEVELS = {
    "numerical_canary",
    "workflow_parity",
    "flagship_workflow",
}
PUBLIC_SEAM_FIELDS = {
    "timings",
    "typed_objects",
    "fmrimod_expresses_better",
}


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _walk_caveat_ids(value: object) -> set[str]:
    caveat_ids: set[str] = set()
    if isinstance(value, dict):
        caveat_id = value.get("caveat_id")
        if isinstance(caveat_id, str) and caveat_id:
            caveat_ids.add(caveat_id)
        for child in value.values():
            caveat_ids.update(_walk_caveat_ids(child))
    elif isinstance(value, list):
        for child in value:
            caveat_ids.update(_walk_caveat_ids(child))
    return caveat_ids


def _report_caveat_ids() -> set[str]:
    caveat_ids: set[str] = set()
    for path in (ROOT / "benchmarks" / "parity").glob("**/*.json"):
        if not ({"reports", "reports_public"} & set(path.parts)):
            continue
        caveat_ids.update(_walk_caveat_ids(json.loads(path.read_text())))
    return caveat_ids


def _caveat_rows() -> dict[str, str]:
    rows: dict[str, str] = {}
    row_pattern = re.compile(
        r"^\| `(?P<caveat>[^`]+)` \| .* \| `(?P<owner>bd-[^`]+)` \|"
    )
    for line in CAVEATS.read_text().splitlines():
        match = row_pattern.match(line)
        if match:
            rows[match.group("caveat")] = match.group("owner")
    return rows


def test_all_parity_workflow_dirs_are_classified() -> None:
    payload = _load_manifest()
    assert payload["schema_version"] == "parity-proof-artifacts/v1"

    workflow_dirs = {
        path.name
        for path in (ROOT / "benchmarks" / "parity").iterdir()
        if path.is_dir() and (path / "workflow.py").exists()
    }
    artifact_ids = {item["benchmark_id"] for item in payload["artifacts"]}

    assert workflow_dirs <= artifact_ids
    assert len(artifact_ids) == len(payload["artifacts"])


def test_proof_artifacts_have_required_paths_and_evidence_levels() -> None:
    for item in _load_manifest()["artifacts"]:
        assert item["evidence_level"] in EVIDENCE_LEVELS
        assert isinstance(item["public_seam"], bool)
        assert (ROOT / item["workflow_path"]).exists()
        assert (ROOT / item["report_path"]).exists()
        assert item["reference_path"]
        assert item["fmrimod_path"]
        assert isinstance(item["caveats"], list)


def test_public_seam_artifacts_carry_strict_report_obligations() -> None:
    for item in _load_manifest()["artifacts"]:
        if not item["public_seam"]:
            continue

        missing = PUBLIC_SEAM_FIELDS - set(item)
        assert not missing, f"{item['benchmark_id']} missing {sorted(missing)}"
        assert item["typed_objects"]
        assert item["fmrimod_expresses_better"]
        assert item["timings"]["status"] in {"recorded", "not_recorded"}


def test_public_or_flagship_artifacts_declare_receipt_status() -> None:
    for item in _load_manifest()["artifacts"]:
        if not item["public_seam"] and item["evidence_level"] != "flagship_workflow":
            continue

        receipt = item.get("receipt")
        assert isinstance(receipt, dict), (
            f"{item['benchmark_id']} should declare receipt metadata"
        )
        assert "expected_report_path" not in receipt

        if receipt["status"] == "regenerable":
            command = receipt.get("command")
            assert isinstance(command, list)
            assert all(isinstance(arg, str) for arg in command)
            assert "report_path" in receipt["checks"]
            assert "status" in receipt["checks"]
            assert "caveats" in receipt["checks"]
        elif receipt["status"] == "static_snapshot":
            assert receipt["reason"]
            assert receipt["date"]
            assert receipt["owner"]
            assert "command" not in receipt
        else:
            raise AssertionError(
                f"{item['benchmark_id']} has unknown receipt status "
                f"{receipt['status']!r}"
            )


def test_canaries_name_the_public_workflow_that_should_replace_them() -> None:
    for item in _load_manifest()["artifacts"]:
        if item["evidence_level"] == "numerical_canary":
            assert not item["public_seam"]
            assert item["replacement_target"]


def test_caveats_index_matches_generated_report_caveat_ids() -> None:
    assert set(_caveat_rows()) == _report_caveat_ids()


def test_caveats_index_owners_are_live_mote_items() -> None:
    if shutil.which("mote") is None:
        pytest.skip("mote CLI not available")

    for caveat_id, owner in _caveat_rows().items():
        result = subprocess.run(
            ["mote", "show", owner],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "status:   open" in result.stdout, (
            f"{caveat_id} owner {owner} should be a live work item"
        )
