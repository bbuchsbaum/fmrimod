"""Schema tests for parity proof-artifact classification."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks" / "parity" / "proof_artifacts.json"

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


def test_canaries_name_the_public_workflow_that_should_replace_them() -> None:
    for item in _load_manifest()["artifacts"]:
        if item["evidence_level"] == "numerical_canary":
            assert not item["public_seam"]
            assert item["replacement_target"]
