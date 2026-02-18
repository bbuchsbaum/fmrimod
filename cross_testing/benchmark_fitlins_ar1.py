#!/usr/bin/env python
"""Run fitlins-aligned AR(1) parity + speed benchmark and emit JSON report."""

from __future__ import annotations

import argparse
import json
import os
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
        "--auto-tune",
        action="store_true",
        help=(
            "Auto-tune AR1 candidate config (iter_gls/voxelwise) on this machine "
            "before the final benchmark run."
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
        selected = {
            "iter_gls": int(args.iter_gls),
            "voxelwise": bool(args.voxelwise),
        }
        auto_tune_report = None

        if args.auto_tune:
            cpu_count = max(1, int(os.cpu_count() or 1))
            candidate_configs = [
                selected,
                {"iter_gls": 1, "voxelwise": False},
                {"iter_gls": 2, "voxelwise": False},
                {"iter_gls": 1, "voxelwise": cpu_count > 1},
            ]

            seen = set()
            unique_candidates = []
            for cfg in candidate_configs:
                key = (int(cfg["iter_gls"]), bool(cfg["voxelwise"]))
                if key in seen:
                    continue
                seen.add(key)
                unique_candidates.append(cfg)

            trials = []
            for cfg in unique_candidates:
                trial = run_ar1_parity_and_benchmark(
                    n_timepoints=args.n_timepoints,
                    n_regressors=args.n_regressors,
                    n_voxels=args.n_voxels,
                    phi=args.phi,
                    noise_sd=args.noise_sd,
                    seed=args.seed,
                    repeats=int(args.auto_tune_repeats),
                    warmup=int(args.auto_tune_warmup),
                    iter_gls=int(cfg["iter_gls"]),
                    voxelwise=bool(cfg["voxelwise"]),
                )
                trials.append(
                    {
                        **cfg,
                        "speedup_vs_reference": float(
                            trial["speed"]["summary"]["speedup_vs_reference"]
                        ),
                        "parity_ok": bool(trial["parity"]["ok"]),
                    }
                )

            valid_trials = [t for t in trials if t["parity_ok"]]
            ranked = sorted(
                valid_trials if valid_trials else trials,
                key=lambda t: float(t["speedup_vs_reference"]),
                reverse=True,
            )
            best = ranked[0]
            best_non_voxelwise = next(
                (t for t in ranked if not bool(t["voxelwise"])),
                None,
            )
            if (
                bool(best["voxelwise"])
                and best_non_voxelwise is not None
                and float(best["speedup_vs_reference"])
                < 1.30 * float(best_non_voxelwise["speedup_vs_reference"])
            ):
                best = best_non_voxelwise
            selected = {
                "iter_gls": int(best["iter_gls"]),
                "voxelwise": bool(best["voxelwise"]),
            }
            auto_tune_report = {
                "enabled": True,
                "repeats": int(args.auto_tune_repeats),
                "warmup": int(args.auto_tune_warmup),
                "selected": selected,
                "trials": trials,
            }

        report = run_ar1_parity_and_benchmark(
            n_timepoints=args.n_timepoints,
            n_regressors=args.n_regressors,
            n_voxels=args.n_voxels,
            phi=args.phi,
            noise_sd=args.noise_sd,
            seed=args.seed,
            repeats=args.repeats,
            warmup=args.warmup,
            iter_gls=selected["iter_gls"],
            voxelwise=selected["voxelwise"],
        )
        if auto_tune_report is not None:
            report["auto_tune"] = auto_tune_report
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
