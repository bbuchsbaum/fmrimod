"""Schema tests for parity proof-artifact classification."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from benchmarks.parity.caveats_contract import (
    caveat_index_rows,
    live_caveat_ids,
)

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
NUMERIC_TIMING_FIELDS = ("seconds", "seconds_total", "wall_seconds")
PRIVATE_KERNEL_NAMES = (
    "fit_glm_from_suffstats",
    "fit_glm_from_matrix",
    "contrast_t",
    "contrast_f",
    "fast_lm_matrix",
    "fast_preproject",
)


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _load_report(path: str) -> dict | None:
    report_path = ROOT / path
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text())


def _has_numeric_timing_payload(timings: object) -> bool:
    if not isinstance(timings, dict):
        return False
    if any(
        isinstance(timings.get(field), (int, float))
        and not isinstance(timings.get(field), bool)
        for field in NUMERIC_TIMING_FIELDS
    ):
        return True
    stages = timings.get("stages")
    if not isinstance(stages, dict) or not stages:
        return False
    return all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in stages.values()
    )


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
        # Static-snapshot receipts commit to a checked-in report file as
        # their proof of having been run. Regenerable receipts are verified
        # by ``test_parity_receipts`` (which actually runs the receipt
        # command); their report files may be absent on a fresh checkout
        # before that test renders them. Canaries without a receipt only
        # carry an aspirational ``report_path`` because their reports are
        # gitignored under ``*/reports/`` and produced on-demand.
        receipt = item.get("receipt")
        assert isinstance(item["report_path"], str) and item["report_path"]
        if isinstance(receipt, dict) and receipt.get("status") == "static_snapshot":
            assert (ROOT / item["report_path"]).exists(), (
                f"{item['benchmark_id']} declares a static_snapshot receipt "
                f"but its report is missing"
            )
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


def test_multirun_concat_public_row_does_not_claim_private_kernel_path() -> None:
    rows = {
        item["benchmark_id"]: item
        for item in _load_manifest()["artifacts"]
        if item["benchmark_id"].startswith("tier_a_multirun_concat")
    }

    canary = rows["tier_a_multirun_concat"]
    assert canary["evidence_level"] == "numerical_canary"
    assert canary["public_seam"] is False
    assert "fast_lm_matrix" in canary["fmrimod_path"]

    public = rows["tier_a_multirun_concat_public_seam"]
    assert public["public_seam"] is True
    assert public["realized_design_source"] == "fmrimod"
    assert not any(name in public["fmrimod_path"] for name in PRIVATE_KERNEL_NAMES)


def test_flagship_workflows_are_complete_public_proof_receipts() -> None:
    """Flagship rows must be real proof receipts, not aspirational labels."""
    for item in _load_manifest()["artifacts"]:
        if item["evidence_level"] != "flagship_workflow":
            continue

        bid = item["benchmark_id"]
        assert item["public_seam"] is True, f"{bid} must be a public seam"
        assert item["typed_objects"], f"{bid} must name typed-object receipts"
        assert item["fmrimod_expresses_better"], (
            f"{bid} must explain the typed fmrimod advantage"
        )
        assert not item.get("replacement_target"), (
            f"{bid} is already flagship_workflow and must not carry a "
            f"replacement_target"
        )

        timings = item.get("timings")
        assert isinstance(timings, dict), f"{bid} must carry timings metadata"
        assert timings.get("status") == "recorded", (
            f"{bid} must have recorded timings, not {timings.get('status')!r}"
        )
        assert _has_numeric_timing_payload(timings), (
            f"{bid} timings must include numeric seconds or numeric stages"
        )

        report_path = ROOT / item["report_path"]
        if report_path.exists():
            report = json.loads(report_path.read_text())
            scorecard = report.get("proof_scorecard")
            if scorecard is None:
                continue
            assert isinstance(scorecard, dict), f"{bid} must render a scorecard"
            assert "low_level_canaries" not in scorecard, (
                f"{bid} is a flagship receipt and must not carry "
                "low_level_canaries in proof_scorecard"
            )
            public_rows = set(scorecard.get("public_rows") or ())
            if public_rows:
                report_rows = report.get("rows")
                assert isinstance(report_rows, list), f"{bid} must render rows"
                assert {row["case_id"] for row in report_rows} <= public_rows
                non_public = [
                    row["case_id"]
                    for row in report_rows
                    if "public-seam" not in row.get("capability", "")
                ]
                assert not non_public, (
                    f"{bid} flagship report includes non-public rows: "
                    f"{non_public}"
                )


def test_flagship_reports_render_complete_fit_provenance() -> None:
    """Flagship receipts must expose complete FitProvenance, not just fits."""
    for item in _load_manifest()["artifacts"]:
        if item["evidence_level"] != "flagship_workflow":
            continue

        bid = item["benchmark_id"]
        report = _load_report(item["report_path"])
        if report is None:
            continue
        provenance = report.get("fit_provenance")
        assert isinstance(provenance, dict), (
            f"{bid} must render a fit_provenance block"
        )
        assert provenance.get("schema_version") == "FitProvenance/v1"
        assert provenance.get("seed_status") in {"randomized", "not_randomized"}
        assert provenance.get("ar_status") == "carried"
        assert provenance.get("mask_status") == "carried"
        assert provenance.get("design_source_status") == "carried"
        assert provenance.get("completeness_errors") == []


def test_canaries_name_the_public_workflow_that_should_replace_them() -> None:
    for item in _load_manifest()["artifacts"]:
        if item["evidence_level"] == "numerical_canary":
            assert not item["public_seam"]
            target = item["replacement_target"]
            assert isinstance(target, dict)
            assert target["description"]
            assert target["owner_bead"].startswith("bd-")
            assert target["blocking_api_gap"]


def test_caveats_index_matches_generated_report_caveat_ids() -> None:
    # A caveat is "live" if a manifest artifact declares it, a generated
    # report emits it, or it is explicitly classed no-report by design
    # (manifest ``no_report_caveats`` — a documented, owned caveat whose
    # typed path raises so no workflow can honestly emit it). The index
    # must match that set exactly: an un-classed extra row still signals
    # stale documentation; a missing row signals an undocumented caveat.
    assert set(caveat_index_rows(ROOT)) == live_caveat_ids(ROOT)


_LIVE_OWNER_STATUSES = frozenset({"open", "doing", "blocked"})


def _owner_status_is_live(show_stdout: str) -> bool:
    """True if a ``mote show`` owner is a live (not closed) work item.

    The caveats index requires every row to have a *live* owning work
    item. ``open``/``doing``/``blocked`` are all live (tracked, not
    done); only ``closed`` means the caveat outlived its owner and
    should have been retired. Accepting only ``open`` spuriously fails
    the moment someone claims the bead (``-> doing``) to actually work
    the exit criterion (bd-01KRRQNE0K4BJBZG0W07P53ET8).
    """
    for line in show_stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("status:"):
            status = stripped.split(":", 1)[1].strip()
            return status in _LIVE_OWNER_STATUSES
    return False


def test_owner_status_is_live_accepts_active_states_rejects_closed() -> None:
    for status in ("open", "doing", "blocked"):
        assert _owner_status_is_live(f"id:       bd-x\nstatus:   {status}\n"), (
            f"{status!r} owner must count as a live work item"
        )
    # The real stale-caveat defect must still be caught (cheap-pass
    # disqualifier: a fix that no longer fails a closed owner is wrong).
    assert not _owner_status_is_live("id:       bd-x\nstatus:   closed\n")
    # Defensive: a missing/garbled status is not live.
    assert not _owner_status_is_live("id: bd-x\n(no status line)\n")
    assert not _owner_status_is_live("")


def test_caveats_index_owners_are_live_mote_items() -> None:
    if shutil.which("mote") is None:
        pytest.skip("mote CLI not available")

    for caveat_id, owner in caveat_index_rows(ROOT).items():
        result = subprocess.run(
            ["mote", "show", owner],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert _owner_status_is_live(result.stdout), (
            f"{caveat_id} owner {owner} is not a live work item "
            f"(expected open/doing/blocked); a closed owner means the "
            f"caveat should have been retired"
        )
