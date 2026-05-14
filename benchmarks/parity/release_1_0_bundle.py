"""Build the fmrimod 1.0 release proof-bundle receipt."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
PARITY_ROOT = ROOT / "benchmarks" / "parity"
PROOF_ARTIFACTS = PARITY_ROOT / "proof_artifacts.json"
RELEASE_MANIFEST = PARITY_ROOT / "release_1_0_manifest.json"
DEFAULT_OUTPUT = PARITY_ROOT / "release_1_0" / "release_receipt.json"
CAVEATS = ROOT / "docs" / "contracts" / "CAVEATS.md"
API_INVENTORY = ROOT / "docs" / "contracts" / "api_inventory_v1.json"

SCHEMA_VERSION = "release_1_0_receipt/v1"

GATE_FILES = (
    "tests/test_flagship_seam_smoke.py",
    "tests/test_public_api/test_api_inventory.py",
    "tests/test_benchmarks/test_proof_artifact_timing_gate.py",
    "tests/test_benchmarks/test_proof_artifact_hardware_tag_gate.py",
    "tests/test_benchmarks/test_tolerance_audit.py",
    "tests/test_benchmarks/test_report_caveats_schema.py",
)

API_SPINE_NAMES = (
    "fmri_dataset",
    "fmri_lm",
    "Spec",
    "Term",
    "event_model",
    "baseline_model",
    "fmri_meta",
    "fmri_ttest",
    "group_data_from_fmrilm",
    "estimate_single_trial",
    "estimate_single_trial_from_dataset",
)

PRIVATE_KERNEL_NAMES = (
    "fit_glm_from_suffstats",
    "fit_glm_from_matrix",
    "contrast_t",
    "contrast_f",
    "fast_lm_matrix",
    "fast_preproject",
)

REQUIRED_MAPPING_KEYS = frozenset(
    {
        "family",
        "benchmark_id",
        "required_public_path",
        "current_status",
        "owner_bead",
    }
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _load_proof_artifacts() -> dict[str, dict[str, object]]:
    payload = _load_json(PROOF_ARTIFACTS)
    artifacts = cast(list[dict[str, object]], payload["artifacts"])
    return {cast(str, item["benchmark_id"]): item for item in artifacts}


def _load_release_manifest(path: Path = RELEASE_MANIFEST) -> dict[str, object]:
    payload = _load_json(path)
    if payload.get("schema_version") != "release_1_0_manifest/v1":
        raise ValueError("release manifest must use schema release_1_0_manifest/v1")
    return payload


def _caveat_rows() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    row_pattern = re.compile(
        r"^\| `(?P<caveat>[^`]+)` \| (?P<first>.*?) \| (?P<tiers>.*?) "
        r"\| `(?P<owner>bd-[^`]+)` \| (?P<exit>.*?) \|$"
    )
    for line in CAVEATS.read_text().splitlines():
        match = row_pattern.match(line)
        if match:
            rows[match.group("caveat")] = {
                "owner": match.group("owner"),
                "exit_criterion": match.group("exit").strip(),
            }
    return rows


def _api_spine_rows() -> list[dict[str, object]]:
    inventory = _load_json(API_INVENTORY)
    inventory_rows = cast(list[dict[str, object]], inventory["rows"])
    rows_by_name = {row["name"]: row for row in inventory_rows}
    rows: list[dict[str, object]] = []
    for name in API_SPINE_NAMES:
        row = rows_by_name.get(name)
        if row is None:
            rows.append(
                {
                    "name": name,
                    "present": False,
                    "tier": None,
                    "used_by_public_seam_artifact": None,
                    "compatibility_status": None,
                    "classified": False,
                }
            )
            continue
        used = row.get("used_by_public_seam_artifact")
        compat = row.get("compatibility_status")
        rows.append(
            {
                "name": name,
                "present": True,
                "tier": row.get("tier"),
                "used_by_public_seam_artifact": used,
                "compatibility_status": compat,
                "classified": used != "review_pending"
                and compat != "review_pending",
            }
        )
    return rows


def _has_numeric_timing(timings: object) -> bool:
    if not isinstance(timings, dict):
        return False
    for field in ("seconds", "seconds_total", "wall_seconds"):
        value = timings.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
    stages = timings.get("stages")
    return isinstance(stages, dict) and bool(stages) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in stages.values()
    )


def _artifact_workflow_path(artifact: dict[str, object]) -> Path | None:
    workflow_path = artifact.get("workflow_path")
    if not isinstance(workflow_path, str) or not workflow_path:
        return None
    return ROOT / workflow_path


def _private_kernel_imports(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    tree = ast.parse(path.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.rsplit(".", maxsplit=1)[-1]
                if name in PRIVATE_KERNEL_NAMES:
                    imported.add(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in PRIVATE_KERNEL_NAMES:
                    imported.add(alias.name)
    return sorted(imported)


def _expected_private_kernel_rows(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name)
            and target.id == "EXPECTED_PRIVATE_KERNEL_ROWS"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, tuple):
            raise ValueError("EXPECTED_PRIVATE_KERNEL_ROWS must be a tuple")
        if not all(isinstance(item, str) for item in value):
            raise ValueError("EXPECTED_PRIVATE_KERNEL_ROWS must contain strings")
        return sorted(value)
    return []


def _private_kernel_names_in_text(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return sorted(name for name in PRIVATE_KERNEL_NAMES if name in value)


def _private_kernel_evidence(artifact: dict[str, object]) -> dict[str, object]:
    workflow_path = _artifact_workflow_path(artifact)
    benchmark_id = cast(str, artifact["benchmark_id"])
    fmrimod_path_names = _private_kernel_names_in_text(artifact.get("fmrimod_path"))
    expected_rows = _expected_private_kernel_rows(workflow_path)
    expected_for_row = benchmark_id in expected_rows
    row_private_names = sorted(set(fmrimod_path_names))
    return {
        "imported_private_kernels": _private_kernel_imports(workflow_path),
        "expected_private_kernel_rows": expected_rows,
        "row_expected_private_kernel": expected_for_row,
        "row_private_kernels": row_private_names,
        "row_uses_private_kernel": expected_for_row or bool(row_private_names),
    }


def _mapping_blockers(
    mapping: dict[str, object],
    artifact: dict[str, object],
    private_kernel_evidence: dict[str, object],
) -> list[str]:
    blockers: list[str] = []
    benchmark_id = cast(str, mapping["benchmark_id"])
    if artifact.get("public_seam") is not True:
        blockers.append(f"{benchmark_id}: public_seam is not true")
    if artifact.get("evidence_level") == "numerical_canary":
        blockers.append(f"{benchmark_id}: numerical_canary cannot be flagship proof")
    required_path = cast(str, mapping["required_public_path"])
    fmrimod_path = cast(str, artifact.get("fmrimod_path", ""))
    if required_path not in fmrimod_path:
        blockers.append(
            f"{benchmark_id}: required public path {required_path!r} "
            "not in fmrimod_path"
        )
    timings = artifact.get("timings")
    if not (isinstance(timings, dict) and timings.get("status") == "recorded"):
        blockers.append(f"{benchmark_id}: timings are not recorded")
    elif not _has_numeric_timing(timings):
        blockers.append(f"{benchmark_id}: timings lack numeric payload")
    hardware_tag = artifact.get("hardware_tag")
    if not isinstance(hardware_tag, str) or not hardware_tag:
        blockers.append(f"{benchmark_id}: hardware_tag is missing")
    receipt = artifact.get("receipt")
    if not isinstance(receipt, dict):
        blockers.append(f"{benchmark_id}: receipt metadata is missing")
    elif receipt.get("status") != "regenerable":
        blockers.append(f"{benchmark_id}: receipt is not regenerable")
    if private_kernel_evidence["row_uses_private_kernel"]:
        names = private_kernel_evidence["row_private_kernels"]
        if private_kernel_evidence["row_expected_private_kernel"] and not names:
            names = ["declared expected private-kernel row"]
        blockers.append(
            f"{benchmark_id}: private kernel path cannot be flagship proof "
            f"({', '.join(cast(list[str], names))})"
        )
    return blockers


def _validate_mapping(mapping: dict[str, object]) -> list[str]:
    blockers: list[str] = []
    missing = REQUIRED_MAPPING_KEYS - set(mapping)
    if missing:
        blockers.append(f"release mapping missing columns: {sorted(missing)}")
    for key in sorted(REQUIRED_MAPPING_KEYS & set(mapping)):
        value = mapping[key]
        if not isinstance(value, str) or not value:
            blockers.append(
                f"release mapping for {mapping.get('family', '<unknown>')} "
                f"has empty {key}"
            )
    return blockers


def build_receipt() -> dict[str, object]:
    """Build the deterministic release receipt payload."""
    release_manifest = _load_release_manifest()
    artifacts = _load_proof_artifacts()
    caveat_index = _caveat_rows()
    blockers: list[str] = []
    rows: list[dict[str, object]] = []

    mappings = cast(list[dict[str, object]], release_manifest["flagship_families"])
    for mapping in mappings:
        row_blockers = _validate_mapping(mapping)
        artifact = artifacts.get(mapping.get("benchmark_id"))
        if artifact is None:
            row_blockers.append(
                f"{mapping.get('benchmark_id')}: benchmark_id not in proof_artifacts"
            )
            rows.append(
                {**mapping, "artifact_present": False, "blockers": row_blockers}
            )
            blockers.extend(row_blockers)
            continue

        private_kernel_evidence = _private_kernel_evidence(artifact)
        row_blockers.extend(
            _mapping_blockers(mapping, artifact, private_kernel_evidence)
        )
        caveats: list[dict[str, object]] = []
        for caveat_id in cast(list[str], artifact.get("caveats", [])):
            caveat_row = caveat_index.get(caveat_id)
            if caveat_row is None:
                row_blockers.append(
                    f"{artifact['benchmark_id']}: caveat {caveat_id!r} "
                    "missing from CAVEATS.md"
                )
                caveats.append({"caveat_id": caveat_id, "indexed": False})
            else:
                caveats.append(
                    {
                        "caveat_id": caveat_id,
                        "indexed": True,
                        "owner": caveat_row["owner"],
                        "exit_criterion": caveat_row["exit_criterion"],
                    }
                )

        rows.append(
            {
                **mapping,
                "artifact_present": True,
                "evidence_level": artifact.get("evidence_level"),
                "public_seam": artifact.get("public_seam"),
                "workflow_path": artifact.get("workflow_path"),
                "report_path": artifact.get("report_path"),
                "reference_path": artifact.get("reference_path"),
                "fmrimod_path": artifact.get("fmrimod_path"),
                "typed_objects": artifact.get("typed_objects", []),
                "caveats": caveats,
                "timings": artifact.get("timings"),
                "hardware_tag": artifact.get("hardware_tag"),
                "receipt": artifact.get("receipt"),
                "private_kernel_evidence": private_kernel_evidence,
                "blockers": row_blockers,
            }
        )
        blockers.extend(row_blockers)

    api_spine = _api_spine_rows()
    for row in api_spine:
        if not row["present"]:
            blockers.append(f"api spine {row['name']}: missing from inventory")
        elif not row["classified"]:
            blockers.append(f"api spine {row['name']}: still review_pending")

    gate_rows = []
    for rel_path in GATE_FILES:
        exists = (ROOT / rel_path).exists()
        gate_rows.append({"path": rel_path, "exists": exists})
        if not exists:
            blockers.append(f"gate file missing: {rel_path}")

    return {
        "schema_version": SCHEMA_VERSION,
        "release_status": "blocked" if blockers else "pass",
        "source": {
            "proof_artifacts": str(PROOF_ARTIFACTS.relative_to(ROOT)),
            "release_manifest": str(RELEASE_MANIFEST.relative_to(ROOT)),
            "caveats": str(CAVEATS.relative_to(ROOT)),
            "api_inventory": str(API_INVENTORY.relative_to(ROOT)),
        },
        "board_source": release_manifest["board_source"],
        "owner_bead": release_manifest["owner_bead"],
        "receipt_command": release_manifest["receipt_command"],
        "flagship_families": rows,
        "api_spine": api_spine,
        "evidence_gates": gate_rows,
        "blockers": blockers,
    }


def write_receipt(path: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Build and write the release receipt to *path*."""
    payload = build_receipt()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="receipt output path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when the release receipt is blocked",
    )
    args = parser.parse_args(argv)
    payload = write_receipt(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict and payload["release_status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
