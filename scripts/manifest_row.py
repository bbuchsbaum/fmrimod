"""Validate and insert one parity-artifact row into ``proof_artifacts.json``.

The proof-artifact manifest is a shared coordination hotspot: every new
benchmark row must satisfy the same per-row schema enforced by
``tests/test_benchmarks/test_parity_proof_artifacts.py``. Editing the
manifest by hand to insert one row routinely scoops unrelated edits
made by parallel agent turns. This helper centralises the per-row
validation and the insert/sort step so a new row can land without
rewriting the rest of the file.

Usage::

    python scripts/manifest_row.py validate path/to/row.json
    python scripts/manifest_row.py add      path/to/row.json
    python scripts/manifest_row.py add      path/to/row.json --dry-run

The validator is a strict subset of the runtime test rules — passing
``validate`` is necessary but not sufficient for the test suite (file-
existence and live-mote checks remain there). Failing ``validate`` is
sufficient to know the row would break the suite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "benchmarks" / "parity" / "proof_artifacts.json"

EVIDENCE_LEVELS = {"numerical_canary", "workflow_parity", "flagship_workflow"}
PUBLIC_SEAM_FIELDS = ("timings", "typed_objects", "fmrimod_expresses_better")
REQUIRED_FIELDS = (
    "benchmark_id",
    "evidence_level",
    "public_seam",
    "workflow_path",
    "report_path",
    "reference_path",
    "fmrimod_path",
    "caveats",
)


def validate_row(
    row: Dict[str, Any],
    *,
    manifest: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return per-row validation errors against the manifest schema.

    An empty list means the row passes the schema. ``manifest`` is
    optional; when provided, this also checks that ``benchmark_id``
    does not duplicate an existing artifact id.
    """
    errors: List[str] = []

    for field in REQUIRED_FIELDS:
        if field not in row:
            errors.append(f"missing required field: {field!r}")

    if "benchmark_id" in row and not isinstance(row["benchmark_id"], str):
        errors.append("benchmark_id must be a string")
    if "evidence_level" in row and row["evidence_level"] not in EVIDENCE_LEVELS:
        errors.append(
            f"evidence_level {row['evidence_level']!r} not in {sorted(EVIDENCE_LEVELS)}"
        )
    if "public_seam" in row and not isinstance(row["public_seam"], bool):
        errors.append("public_seam must be a bool")
    if "caveats" in row and not isinstance(row["caveats"], list):
        errors.append("caveats must be a list")
    for path_field in ("workflow_path", "report_path", "reference_path", "fmrimod_path"):
        if path_field in row and not (
            isinstance(row[path_field], str) and row[path_field]
        ):
            errors.append(f"{path_field} must be a non-empty string")

    if row.get("public_seam") is True:
        for field in PUBLIC_SEAM_FIELDS:
            if field not in row:
                errors.append(f"public_seam=true requires field: {field!r}")
        timings = row.get("timings", {})
        if isinstance(timings, dict) and timings.get("status") not in {"recorded", "not_recorded"}:
            errors.append(
                "public_seam=true requires timings.status in {'recorded', 'not_recorded'}"
            )
        if "typed_objects" in row and not row["typed_objects"]:
            errors.append("public_seam=true requires non-empty typed_objects")
        if "fmrimod_expresses_better" in row and not row["fmrimod_expresses_better"]:
            errors.append("public_seam=true requires non-empty fmrimod_expresses_better")

    needs_receipt = (
        row.get("public_seam") is True
        or row.get("evidence_level") == "flagship_workflow"
    )
    if needs_receipt:
        receipt = row.get("receipt")
        if not isinstance(receipt, dict):
            errors.append("public_seam or flagship_workflow row requires a 'receipt' dict")
        else:
            if "expected_report_path" in receipt:
                errors.append(
                    "receipt must not carry legacy 'expected_report_path' field"
                )
            status = receipt.get("status")
            if status == "regenerable":
                command = receipt.get("command")
                if not (isinstance(command, list) and all(isinstance(a, str) for a in command)):
                    errors.append("regenerable receipt requires command: list[str]")
                checks = receipt.get("checks", [])
                for required_check in ("report_path", "status", "caveats"):
                    if required_check not in checks:
                        errors.append(
                            f"regenerable receipt.checks must include {required_check!r}"
                        )
            elif status == "static_snapshot":
                for field in ("reason", "date", "owner"):
                    if field not in receipt:
                        errors.append(f"static_snapshot receipt requires {field!r}")
                if "command" in receipt:
                    errors.append("static_snapshot receipt must not carry 'command'")
            else:
                errors.append(
                    f"unknown receipt.status {status!r}; "
                    f"expected 'regenerable' or 'static_snapshot'"
                )

    if row.get("evidence_level") == "numerical_canary":
        if row.get("public_seam") is True:
            errors.append("numerical_canary rows must have public_seam=false")
        if not row.get("replacement_target"):
            errors.append(
                "numerical_canary rows require a 'replacement_target' field "
                "naming the public workflow that should replace them"
            )

    if manifest is not None and "benchmark_id" in row:
        existing = {a.get("benchmark_id") for a in manifest.get("artifacts", [])}
        if row["benchmark_id"] in existing:
            errors.append(
                f"benchmark_id {row['benchmark_id']!r} already in manifest "
                f"(use a different id or update the existing row in place)"
            )

    return errors


def add_row(
    manifest_path: Path,
    row: Dict[str, Any],
    *,
    write: bool = True,
) -> Dict[str, Any]:
    """Insert ``row`` into the manifest at ``manifest_path``.

    Validates the row against the existing manifest, inserts it sorted
    by ``benchmark_id``, and (unless ``write=False``) writes the merged
    manifest back. Returns the merged manifest payload.
    """
    manifest = json.loads(manifest_path.read_text())
    errors = validate_row(row, manifest=manifest)
    if errors:
        raise ValueError(
            "row validation failed:\n  - " + "\n  - ".join(errors)
        )
    artifacts = list(manifest.get("artifacts", []))
    artifacts.append(row)
    artifacts.sort(key=lambda r: r.get("benchmark_id", ""))
    manifest["artifacts"] = artifacts
    if write:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _load_row(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"row file not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"row file must contain a single JSON object, got {type(payload).__name__}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_validate = sub.add_parser("validate", help="Validate a row JSON file against the manifest schema.")
    p_validate.add_argument("row", type=Path, help="Path to a single-row JSON file.")
    p_validate.add_argument(
        "--against",
        type=Path,
        default=MANIFEST_PATH,
        help="Manifest to check uniqueness against (default: proof_artifacts.json).",
    )

    p_add = sub.add_parser("add", help="Insert a validated row into the manifest.")
    p_add.add_argument("row", type=Path, help="Path to a single-row JSON file.")
    p_add.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Manifest to insert into (default: proof_artifacts.json).",
    )
    p_add.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the merged manifest without writing.",
    )

    args = parser.parse_args()

    try:
        row = _load_row(args.row)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.cmd == "validate":
        try:
            manifest = json.loads(args.against.read_text()) if args.against.exists() else None
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error reading manifest: {exc}", file=sys.stderr)
            return 2
        errors = validate_row(row, manifest=manifest)
        if errors:
            print("invalid:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 1
        print(f"valid: {row.get('benchmark_id', '<no id>')}")
        return 0

    if args.cmd == "add":
        try:
            merged = add_row(args.manifest, row, write=not args.dry_run)
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.dry_run:
            print(json.dumps(merged, indent=2))
        else:
            print(
                f"inserted: {row['benchmark_id']} "
                f"({len(merged['artifacts'])} artifacts now in {args.manifest.name})"
            )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
