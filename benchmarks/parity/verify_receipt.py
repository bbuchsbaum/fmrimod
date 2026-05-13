"""Regenerate and verify parity proof receipts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks" / "parity" / "proof_artifacts.json"


def _load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text())


def _artifact(benchmark_id: str) -> dict[str, Any]:
    for item in _load_manifest()["artifacts"]:
        if item["benchmark_id"] == benchmark_id:
            return item
    raise ValueError(f"unknown parity artifact: {benchmark_id}")


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


def _receipt_command(item: dict[str, Any]) -> list[str]:
    receipt = item.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError(f"{item['benchmark_id']} has no receipt metadata")
    if receipt.get("status") != "regenerable":
        raise ValueError(f"{item['benchmark_id']} receipt is not regenerable")

    command = receipt.get("command")
    if not isinstance(command, list) or not all(
        isinstance(arg, str) for arg in command
    ):
        raise ValueError(
            f"{item['benchmark_id']} receipt command must be a string list"
        )
    if command and command[0] == "python":
        return [sys.executable, *command[1:]]
    return command


def verify_receipt(benchmark_id: str) -> dict[str, Any]:
    """Regenerate one artifact report and verify it matches manifest claims."""
    item = _artifact(benchmark_id)
    report_path = ROOT / item["report_path"]
    if report_path.exists():
        report_path.unlink()

    subprocess.run(_receipt_command(item), cwd=ROOT, check=True)
    if not report_path.exists():
        raise AssertionError(
            f"receipt did not produce declared report_path: {report_path}"
        )

    report = json.loads(report_path.read_text())
    expected_caveats = set(item["caveats"])
    observed_caveats = _walk_caveat_ids(report)
    if observed_caveats != expected_caveats:
        raise AssertionError(
            f"{benchmark_id} caveats mismatch: "
            f"expected {sorted(expected_caveats)}, observed {sorted(observed_caveats)}"
        )
    if report.get("status") == "fail":
        raise AssertionError(f"{benchmark_id} regenerated report failed")

    return {
        "benchmark_id": benchmark_id,
        "report_path": str(report_path.relative_to(ROOT)),
        "status": report.get("status"),
        "caveats": sorted(observed_caveats),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_id")
    args = parser.parse_args(argv)
    print(json.dumps(verify_receipt(args.benchmark_id), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
