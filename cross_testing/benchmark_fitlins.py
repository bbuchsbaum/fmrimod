#!/usr/bin/env python
"""Run fitlins-aligned parity + speed benchmark and emit JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from cross_testing.fitlins_parity import run_parity_and_benchmark, write_json_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fmrimod-vs-fitlins (nilearn reference) OLS parity "
            "and speed benchmarks."
        )
    )
    parser.add_argument("--n-timepoints", type=int, default=240)
    parser.add_argument("--n-regressors", type=int, default=8)
    parser.add_argument("--n-voxels", type=int, default=2000)
    parser.add_argument("--noise-sd", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--fmrimod-compute-dtype",
        choices=["float64", "float32"],
        default="float64",
        help="Internal compute dtype for fmrimod solver path.",
    )
    parser.add_argument(
        "--fmrimod-n-jobs",
        type=int,
        default=1,
        help="Parallel workers for chunked fmrimod path (1 disables chunk threading).",
    )
    parser.add_argument(
        "--fmrimod-chunk-size",
        type=int,
        default=5000,
        help="Voxel chunk size for threaded fmrimod benchmark path.",
    )
    parser.add_argument(
        "--fmrimod-blas-threads",
        type=int,
        default=None,
        help="Optional BLAS thread cap during threaded chunk solves.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cross_testing/reports/fitlins_parity_benchmark.json",
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
        report = run_parity_and_benchmark(
            n_timepoints=args.n_timepoints,
            n_regressors=args.n_regressors,
            n_voxels=args.n_voxels,
            noise_sd=args.noise_sd,
            seed=args.seed,
            repeats=args.repeats,
            warmup=args.warmup,
            fmrimod_compute_dtype=args.fmrimod_compute_dtype,
            fmrimod_n_jobs=args.fmrimod_n_jobs,
            fmrimod_chunk_size=args.fmrimod_chunk_size,
            fmrimod_blas_threads=args.fmrimod_blas_threads,
        )
    except ModuleNotFoundError as exc:
        print(
            "Missing optional dependency for fitlins parity benchmark: "
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
