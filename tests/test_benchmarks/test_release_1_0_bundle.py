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
    # Tier A FIAC retired its public_seam, numerical_canary, and
    # private-kernel blockers in bd-01KRKACNDMRB6WYYQGE050SJB3: the
    # flagship now fits per-run through
    # ``fmri_dataset -> RealizedDesign(source='nilearn') -> fmri_lm``,
    # pools via ``combine_runs``, and resolves both 2x2 main effects via
    # typed ``column_contrast`` patterns. None of those three Tier A
    # FIAC strings should appear in the blocker list any more.
    fiac_blockers = [
        blocker
        for blocker in receipt["blockers"]
        if blocker.startswith("tier_a_fiac:")
        and (
            "public_seam is not true" in blocker
            or "numerical_canary cannot be flagship proof" in blocker
            or "private kernel path" in blocker
        )
    ]
    assert fiac_blockers == [], (
        "FIAC public-seam blockers must be retired; got " + repr(fiac_blockers)
    )
    tier_b_blockers = [
        blocker
        for blocker in receipt["blockers"]
        if blocker.startswith("tier_b_fitlins_bids:")
    ]
    assert tier_b_blockers == []
    # Tier D LSS trialwise retired its numerical_canary blocker in
    # bd-01KRKAEDBE9BV5HZ5SY0A3ZAA5: the release row now points at the
    # public-seam ``estimate_single_trial_from_dataset`` showcase row;
    # the matrix oracle survives under id ``tier_d_lss_trialwise_oracle``.
    assert (
        "tier_d_lss_trialwise: numerical_canary cannot be flagship proof"
        not in blockers
    )
    assert "api spine " not in blockers


def test_release_receipt_tier_b_is_public_typed_bids_seam() -> None:
    receipt = bundle.build_receipt()

    rows = {row["benchmark_id"]: row for row in receipt["flagship_families"]}
    tier_b = rows["tier_b_fitlins_bids"]
    assert tier_b["current_status"] == "ready"
    assert tier_b["public_seam"] is True
    assert tier_b["receipt"]["status"] == "regenerable"
    assert tier_b["timings"]["status"] == "recorded"
    assert tier_b["hardware_tag"] == "Darwin-arm64-arm"
    assert (
        "fmri_dataset -> fmri_lm -> typed BIDS contrasts"
        in tier_b["fmrimod_path"]
    )
    assert "fmrimod.bids.StatsModelTranslation" in tier_b["typed_objects"]
    assert "fmrimod.bids.StatsModelContrast" in tier_b["typed_objects"]


def test_release_receipt_reports_private_kernel_evidence_by_row() -> None:
    receipt = bundle.build_receipt()

    rows = {row["benchmark_id"]: row for row in receipt["flagship_families"]}
    fiac = rows["tier_a_fiac"]["private_kernel_evidence"]
    # FIAC retired the suffstats kernel path in
    # bd-01KRKACNDMRB6WYYQGE050SJB3 — the flagship now fits through
    # ``fmri_dataset -> RealizedDesign -> fmri_lm`` and combines runs
    # via ``combine_runs``. ``row_uses_private_kernel`` must therefore
    # be False; the only remaining ``fit_glm_from_suffstats`` mention
    # in the file would be a regression we want this test to catch.
    assert fiac["row_uses_private_kernel"] is False, (
        "tier_a_fiac unexpectedly re-introduced a private kernel: "
        f"{fiac['row_private_kernels']}"
    )
    assert fiac["row_private_kernels"] == []
    assert "fit_glm_from_suffstats" not in fiac["imported_private_kernels"], (
        "tier_a_fiac re-imported fit_glm_from_suffstats; the public-seam "
        "port should not depend on that private kernel."
    )
    assert not any(
        blocker.startswith("tier_a_fiac: private kernel path")
        for blocker in rows["tier_a_fiac"]["blockers"]
    )

    showcase = rows["tier_d_showcase"]["private_kernel_evidence"]
    assert "fast_lm_matrix" in showcase["imported_private_kernels"]
    assert "fast_preproject" in showcase["imported_private_kernels"]
    assert showcase["expected_private_kernel_rows"] == ["tier_d_sketched_glm"]
    assert showcase["row_uses_private_kernel"] is False
    assert not any(
        blocker.startswith("tier_d_showcase: private kernel path")
        for blocker in rows["tier_d_showcase"]["blockers"]
    )


def test_release_receipt_carries_gate_files_and_api_spine_evidence() -> None:
    receipt = bundle.build_receipt()

    gate_paths = {row["path"] for row in receipt["evidence_gates"]}
    assert set(bundle.GATE_FILES) == gate_paths
    assert all(row["exists"] for row in receipt["evidence_gates"])

    spine = {row["name"]: row for row in receipt["api_spine"]}
    assert set(bundle.API_SPINE_NAMES) == set(spine)
    assert all(row["classified"] for row in spine.values())
    assert spine["fmri_dataset"]["tier"] == "spine"
    assert spine["fmri_lm"]["tier"] == "spine_review"
    assert {row["compatibility_status"] for row in spine.values()} == {"spine"}
    assert {row["used_by_public_seam_artifact"] for row in spine.values()} == {
        "release_1_0_api_spine"
    }


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
