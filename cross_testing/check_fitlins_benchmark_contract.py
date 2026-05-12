#!/usr/bin/env python
"""Check fitlins parity benchmark contract booleans and speed floor."""

from __future__ import annotations

import argparse
import json
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate fitlins parity benchmark contract."
    )
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Path to benchmark_fitlins JSON report.",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.0,
        help="Minimum acceptable speedup_vs_reference.",
    )
    args = parser.parse_args(argv)

    with open(args.report, "r", encoding="utf-8") as fobj:
        report = json.load(fobj)

    parity_ok = bool(report.get("parity", {}).get("ok", False))
    speedup = float(
        report.get("speed", {})
        .get("summary", {})
        .get("speedup_vs_reference", 0.0)
    )

    failures = []
    if not parity_ok:
        failures.append("parity.ok is false")
    if speedup < args.min_speedup:
        failures.append(
            f"speedup_vs_reference={speedup:.3f} < min_speedup={args.min_speedup:.3f}"
        )

    if failures:
        print("fitlins benchmark contract FAILED:")
        for msg in failures:
            print(f"- {msg}")
        return 1

    print(
        "fitlins benchmark contract passed: "
        f"parity.ok={parity_ok}, speedup={speedup:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
