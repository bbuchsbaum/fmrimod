#!/usr/bin/env python
"""Run fitlins-aligned parity + speed benchmark and emit JSON report."""

from __future__ import annotations

import argparse
import json
import os
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
        "--auto-tune",
        action="store_true",
        help=(
            "Auto-tune fmrimod OLS config on this machine before final report. "
            "When enabled, selected config overrides explicit fmrimod-* args."
        ),
    )
    parser.add_argument(
        "--auto-tune-repeats",
        type=int,
        default=2,
        help="Repeats per candidate during auto-tuning.",
    )
    parser.add_argument(
        "--auto-tune-warmup",
        type=int,
        default=0,
        help="Warmup iterations per candidate during auto-tuning.",
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
        selected = {
            "fmrimod_compute_dtype": args.fmrimod_compute_dtype,
            "fmrimod_n_jobs": int(args.fmrimod_n_jobs),
            "fmrimod_chunk_size": int(args.fmrimod_chunk_size),
            "fmrimod_blas_threads": args.fmrimod_blas_threads,
        }
        auto_tune_report = None

        if args.auto_tune:
            cpu_count = max(1, int(os.cpu_count() or 1))
            tuned_jobs = max(1, min(cpu_count, 4))
            candidate_configs = [
                selected,
                {
                    "fmrimod_compute_dtype": "float64",
                    "fmrimod_n_jobs": 1,
                    "fmrimod_chunk_size": 5000,
                    "fmrimod_blas_threads": None,
                },
                {
                    "fmrimod_compute_dtype": "float32",
                    "fmrimod_n_jobs": 1,
                    "fmrimod_chunk_size": 5000,
                    "fmrimod_blas_threads": None,
                },
                {
                    "fmrimod_compute_dtype": "float64",
                    "fmrimod_n_jobs": tuned_jobs,
                    "fmrimod_chunk_size": 1500,
                    "fmrimod_blas_threads": 1,
                },
                {
                    "fmrimod_compute_dtype": "float32",
                    "fmrimod_n_jobs": tuned_jobs,
                    "fmrimod_chunk_size": 1500,
                    "fmrimod_blas_threads": 1,
                },
            ]

            # Deduplicate candidate configs while preserving order.
            seen = set()
            unique_candidates = []
            for cfg in candidate_configs:
                key = (
                    cfg["fmrimod_compute_dtype"],
                    int(cfg["fmrimod_n_jobs"]),
                    int(cfg["fmrimod_chunk_size"]),
                    cfg["fmrimod_blas_threads"],
                )
                if key in seen:
                    continue
                seen.add(key)
                unique_candidates.append(cfg)

            tuning_trials = []
            for cfg in unique_candidates:
                trial = run_parity_and_benchmark(
                    n_timepoints=args.n_timepoints,
                    n_regressors=args.n_regressors,
                    n_voxels=args.n_voxels,
                    noise_sd=args.noise_sd,
                    seed=args.seed,
                    repeats=args.auto_tune_repeats,
                    warmup=args.auto_tune_warmup,
                    **cfg,
                )
                tuning_trials.append(
                    {
                        **cfg,
                        "speedup_vs_reference": float(
                            trial["speed"]["summary"]["speedup_vs_reference"]
                        ),
                        "parity_ok": bool(trial["parity"]["ok"]),
                    }
                )

            valid_trials = [t for t in tuning_trials if t["parity_ok"]]
            ranked = sorted(
                valid_trials if valid_trials else tuning_trials,
                key=lambda t: float(t["speedup_vs_reference"]),
                reverse=True,
            )
            best = ranked[0]
            selected = {
                "fmrimod_compute_dtype": best["fmrimod_compute_dtype"],
                "fmrimod_n_jobs": int(best["fmrimod_n_jobs"]),
                "fmrimod_chunk_size": int(best["fmrimod_chunk_size"]),
                "fmrimod_blas_threads": best["fmrimod_blas_threads"],
            }
            auto_tune_report = {
                "enabled": True,
                "repeats": int(args.auto_tune_repeats),
                "warmup": int(args.auto_tune_warmup),
                "selected": selected,
                "trials": tuning_trials,
            }

        report = run_parity_and_benchmark(
            n_timepoints=args.n_timepoints,
            n_regressors=args.n_regressors,
            n_voxels=args.n_voxels,
            noise_sd=args.noise_sd,
            seed=args.seed,
            repeats=args.repeats,
            warmup=args.warmup,
            **selected,
        )
        if auto_tune_report is not None:
            report["auto_tune"] = auto_tune_report
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
