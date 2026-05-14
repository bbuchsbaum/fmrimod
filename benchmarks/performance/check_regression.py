"""Check parity performance JSONL history for large regressions."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

GATED_EVIDENCE_LEVELS = frozenset({"flagship_workflow", "workflow_parity"})


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load append-only parity performance trend records."""

    records = []
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _history_key(record: dict[str, Any]) -> tuple[str, str]:
    return (str(record.get("hardware_tag", "")), str(record.get("stage", "")))


def _group_records(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(_history_key(record), []).append(record)
    return groups


def find_history_gaps(
    records: list[dict[str, Any]],
    *,
    min_records: int = 2,
) -> list[dict[str, Any]]:
    """Return history groups that are too small to be trend evidence."""

    if min_records < 2:
        raise ValueError("min_records must be at least 2")
    if not records:
        return [
            {
                "hardware_tag": "",
                "stage": "*",
                "count": 0,
                "min_records": min_records,
                "reason": "no_records",
            }
        ]

    gaps = []
    for (hardware_tag, stage), rows in _group_records(records).items():
        reason = None
        if not hardware_tag:
            reason = "missing_hardware_tag"
        elif not stage:
            reason = "missing_stage"
        elif len(rows) < min_records:
            reason = "insufficient_history"
        if reason is not None:
            gaps.append(
                {
                    "hardware_tag": hardware_tag,
                    "stage": stage,
                    "count": len(rows),
                    "min_records": min_records,
                    "reason": reason,
                }
            )
    return sorted(gaps, key=lambda row: (row["hardware_tag"], row["stage"]))


def find_artifact_history_gaps(
    artifacts: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    min_records: int = 2,
) -> list[dict[str, Any]]:
    """Return recorded proof artifacts without matching performance history."""

    groups = _group_records(records)
    gaps = []
    for artifact in artifacts:
        if artifact.get("evidence_level") not in GATED_EVIDENCE_LEVELS:
            continue
        timings = artifact.get("timings")
        if not isinstance(timings, dict) or timings.get("status") != "recorded":
            continue
        hardware_tag = artifact.get("hardware_tag")
        if not isinstance(hardware_tag, str) or not hardware_tag:
            continue
        stage = str(artifact.get("performance_stage") or artifact["benchmark_id"])
        count = len(groups.get((hardware_tag, stage), []))
        if count < min_records:
            gaps.append(
                {
                    "benchmark_id": artifact["benchmark_id"],
                    "hardware_tag": hardware_tag,
                    "stage": stage,
                    "count": count,
                    "min_records": min_records,
                    "reason": "artifact_history_missing",
                }
            )
    return gaps


def find_regressions(
    records: list[dict[str, Any]],
    *,
    threshold: float = 0.25,
    window: int = 10,
) -> list[dict[str, Any]]:
    """Return latest rows slower than prior rolling median by threshold."""

    regressions = []
    for (hardware_tag, stage), rows in _group_records(records).items():
        rows = sorted(rows, key=lambda row: str(row.get("generated_at", "")))
        if len(rows) < 2:
            continue
        latest = rows[-1]
        previous = rows[:-1][-window:]
        baseline = statistics.median(float(row["seconds"]) for row in previous)
        latest_seconds = float(latest["seconds"])
        if baseline > 0.0 and latest_seconds > baseline * (1.0 + threshold):
            regressions.append(
                {
                    "hardware_tag": hardware_tag,
                    "stage": stage,
                    "latest_seconds": latest_seconds,
                    "baseline_seconds": baseline,
                    "threshold": threshold,
                }
            )
    return regressions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "history",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "parity_performance_trends.jsonl",
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-records", type=int, default=2)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional proof_artifacts.json file whose recorded timings must join the history.",
    )
    args = parser.parse_args()

    records = load_records(args.history)
    history_gaps = find_history_gaps(records, min_records=args.min_records)
    artifact_gaps: list[dict[str, Any]] = []
    if args.manifest is not None:
        artifacts = json.loads(args.manifest.read_text())["artifacts"]
        artifact_gaps = find_artifact_history_gaps(
            artifacts,
            records,
            min_records=args.min_records,
        )
    if history_gaps or artifact_gaps:
        print(
            json.dumps(
                {
                    "artifact_history_gaps": artifact_gaps,
                    "history_gaps": history_gaps,
                    "regressions": [],
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(2)

    regressions = find_regressions(
        records,
        threshold=args.threshold,
        window=args.window,
    )
    if regressions:
        print(json.dumps({"regressions": regressions}, indent=2, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps({"regressions": []}, sort_keys=True))


if __name__ == "__main__":
    main()
