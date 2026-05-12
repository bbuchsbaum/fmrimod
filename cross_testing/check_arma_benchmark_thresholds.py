#!/usr/bin/env python
"""Validate AR/ARMA benchmark report against minimum speed thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence


def _get_metric(report: dict, key: str) -> float:
    rel = report.get("relative_speed_vs_ar_run_p2", {})
    if key not in rel:
        raise KeyError(f"Missing benchmark metric: {key}")
    return float(rel[key])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check AR/ARMA benchmark thresholds."
    )
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Path to JSON report from benchmark_arma_paths.py",
    )
    parser.add_argument("--min-arma-run", type=float, default=1.20)
    parser.add_argument("--min-arma-global", type=float, default=1.05)
    args = parser.parse_args(argv)

    with open(args.report, "r", encoding="utf-8") as fobj:
        report = json.load(fobj)

    arma_run = _get_metric(report, "arma_run_p2q1")
    arma_global = _get_metric(report, "arma_global_p2q1")

    failures = []
    if arma_run < args.min_arma_run:
        failures.append(
            f"arma_run_p2q1={arma_run:.3f} < min_arma_run={args.min_arma_run:.3f}"
        )
    if arma_global < args.min_arma_global:
        failures.append(
            f"arma_global_p2q1={arma_global:.3f} < min_arma_global={args.min_arma_global:.3f}"
        )

    if failures:
        print("AR/ARMA benchmark threshold check FAILED:")
        for msg in failures:
            print(f"- {msg}")
        return 1

    print(
        "AR/ARMA benchmark threshold check passed: "
        f"arma_run_p2q1={arma_run:.3f}, arma_global_p2q1={arma_global:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
