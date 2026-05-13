#!/usr/bin/env python3
"""Validate and insert one parity proof-artifact manifest row.

The helper keeps benchmark-row work reviewable: draft the new row in its own
JSON file, validate it against the current manifest, then write a candidate
manifest to a temp output path before touching `proof_artifacts.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "benchmark_id",
    "evidence_level",
    "public_seam",
    "workflow_path",
    "report_path",
    "reference_path",
    "fmrimod_path",
    "caveats",
}
EVIDENCE_LEVELS = {"numerical_canary", "workflow_parity", "flagship_workflow"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _artifact_ids(manifest: dict[str, Any]) -> set[str]:
    return {str(item["benchmark_id"]) for item in manifest.get("artifacts", [])}


def validate_row(manifest: dict[str, Any], row: dict[str, Any]) -> None:
    """Validate one candidate artifact row against an existing manifest."""

    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        raise ValueError(f"artifact row missing required fields: {', '.join(missing)}")
    benchmark_id = row["benchmark_id"]
    if not isinstance(benchmark_id, str) or not benchmark_id:
        raise ValueError("benchmark_id must be a nonempty string")
    if benchmark_id in _artifact_ids(manifest):
        raise ValueError(f"benchmark_id already exists: {benchmark_id}")
    if row["evidence_level"] not in EVIDENCE_LEVELS:
        raise ValueError(
            f"evidence_level must be one of {sorted(EVIDENCE_LEVELS)}, "
            f"got {row['evidence_level']!r}"
        )
    if not isinstance(row["public_seam"], bool):
        raise ValueError("public_seam must be a boolean")
    if not isinstance(row["caveats"], list):
        raise ValueError("caveats must be a list")
    if row["evidence_level"] == "numerical_canary" and not row.get(
        "replacement_target"
    ):
        raise ValueError("numerical_canary rows require replacement_target")


def insert_row(manifest: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Return a manifest copy with one validated row appended."""

    validate_row(manifest, row)
    out = dict(manifest)
    out["artifacts"] = [*manifest.get("artifacts", []), row]
    return out


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-row")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--row", type=Path, required=True)

    insert = subparsers.add_parser("insert-row")
    insert.add_argument("--manifest", type=Path, required=True)
    insert.add_argument("--row", type=Path, required=True)
    insert.add_argument(
        "--output",
        type=Path,
        help="Output manifest path. Defaults to in-place manifest rewrite.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = _load_json(args.manifest)
    row = _load_json(args.row)
    if args.command == "validate-row":
        validate_row(manifest, row)
        return 0
    if args.command == "insert-row":
        out = insert_row(manifest, row)
        _write_manifest(args.output or args.manifest, out)
        return 0
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
