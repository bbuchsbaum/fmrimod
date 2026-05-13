"""Check parity performance JSONL history for large regressions."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load append-only parity performance trend records."""

    records = []
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def find_regressions(
    records: list[dict[str, Any]],
    *,
    threshold: float = 0.25,
    window: int = 10,
) -> list[dict[str, Any]]:
    """Return latest rows slower than prior rolling median by threshold."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("hardware_tag", "")), str(record.get("stage", "")))
        groups.setdefault(key, []).append(record)

    regressions = []
    for (hardware_tag, stage), rows in groups.items():
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
    args = parser.parse_args()

    regressions = find_regressions(
        load_records(args.history),
        threshold=args.threshold,
        window=args.window,
    )
    if regressions:
        print(json.dumps({"regressions": regressions}, indent=2, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps({"regressions": []}, sort_keys=True))


if __name__ == "__main__":
    main()
