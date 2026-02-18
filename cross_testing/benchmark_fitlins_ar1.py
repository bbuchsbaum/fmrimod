#!/usr/bin/env python
"""Run fitlins-aligned AR(1) parity + speed benchmark and emit JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from cross_testing.fitlins_ar1_parity import (
    run_ar1_parity_and_benchmark,
    write_json_report,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fmrimod-vs-fitlins (nilearn reference) AR(1) parity "
            "and speed benchmarks."
        )
    )
    parser.add_argument("--n-timepoints", type=int, default=260)
    parser.add_argument("--n-regressors", type=int, default=10)
    parser.add_argument("--n-voxels", type=int, default=3000)
    parser.add_argument("--phi", type=float, default=0.45)
    parser.add_argument("--noise-sd", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--iter-gls",
        type=int,
        default=1,
        help="Number of AR(1) GLS iterations in candidate fmrimod path.",
    )
    parser.add_argument(
        "--voxelwise",
        action="store_true",
        help="Estimate AR(1) coefficients per voxel (candidate path).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cross_testing/reports/fitlins_ar1_parity_benchmark.json",
        help="Path to write the JSON report.",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print report JSON to stdout without writing a file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = run_ar1_parity_and_benchmark(
            n_timepoints=args.n_timepoints,
            n_regressors=args.n_regressors,
            n_voxels=args.n_voxels,
            phi=args.phi,
            noise_sd=args.noise_sd,
            seed=args.seed,
            repeats=args.repeats,
            warmup=args.warmup,
            iter_gls=args.iter_gls,
            voxelwise=args.voxelwise,
        )
    except ModuleNotFoundError as exc:
        print(
            "Missing optional dependency for fitlins AR(1) parity benchmark: "
            f"{exc}. Install nilearn to run this command.",
            file=sys.stderr,
        )
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.stdout_only:
        write_json_report(report, args.output)
        print(f"\nWrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
