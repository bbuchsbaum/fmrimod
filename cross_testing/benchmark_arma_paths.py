#!/usr/bin/env python
"""Benchmark AR/ARMA noise fit + whitening paths in fmrimod."""

from __future__ import annotations

import argparse
import json
from statistics import median
import time
from typing import Dict, List, Sequence

import numpy as np

from fmrimod.ar.estimation import fit_noise
from fmrimod.ar.whitening import whiten_apply


def _build_data(
    *,
    n_runs: int,
    n_timepoints: int,
    n_regressors: int,
    n_voxels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Xs: List[np.ndarray] = []
    Ys: List[np.ndarray] = []
    run_labels: List[int] = []

    for r in range(n_runs):
        Xr = np.column_stack(
            [np.ones(n_timepoints), rng.standard_normal((n_timepoints, n_regressors - 1))]
        ).astype(np.float64)
        Br = rng.standard_normal((n_regressors, n_voxels)).astype(np.float64)
        Yr = Xr @ Br + rng.standard_normal((n_timepoints, n_voxels)).astype(np.float64)
        Xs.append(Xr)
        Ys.append(Yr)
        run_labels.extend([r] * n_timepoints)

    X = np.vstack(Xs)
    Y = np.vstack(Ys)
    runs = np.asarray(run_labels, dtype=np.intp)
    return X, Y, runs


def _time_case(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    runs: np.ndarray,
    method: str,
    p: int,
    q: int,
    pooling: str,
    repeats: int,
    warmup: int,
) -> Dict[str, object]:
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef

    for _ in range(warmup):
        plan = fit_noise(resid=resid, runs=runs, method=method, p=p, q=q, pooling=pooling)
        _ = whiten_apply(plan, X, Y, runs=runs)

    fit_times: List[float] = []
    white_times: List[float] = []
    total_times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        plan = fit_noise(resid=resid, runs=runs, method=method, p=p, q=q, pooling=pooling)
        t1 = time.perf_counter()
        _ = whiten_apply(plan, X, Y, runs=runs)
        t2 = time.perf_counter()
        fit_times.append(t1 - t0)
        white_times.append(t2 - t1)
        total_times.append(t2 - t0)

    return {
        "fit_noise_runs_s": fit_times,
        "whiten_runs_s": white_times,
        "total_runs_s": total_times,
        "fit_noise_median_s": median(fit_times),
        "whiten_median_s": median(white_times),
        "total_median_s": median(total_times),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark AR/ARMA paths in fmrimod.")
    parser.add_argument("--n-runs", type=int, default=6)
    parser.add_argument("--n-timepoints", type=int, default=260)
    parser.add_argument("--n-regressors", type=int, default=10)
    parser.add_argument("--n-voxels", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--output",
        type=str,
        default="cross_testing/reports/fmrimod_arma_benchmark.json",
    )
    args = parser.parse_args(argv)

    X, Y, runs = _build_data(
        n_runs=args.n_runs,
        n_timepoints=args.n_timepoints,
        n_regressors=args.n_regressors,
        n_voxels=args.n_voxels,
        seed=args.seed,
    )

    cases = {
        "ar_run_p2": dict(method="ar", p=2, q=0, pooling="run"),
        "ar_global_p2": dict(method="ar", p=2, q=0, pooling="global"),
        "arma_run_p2q1": dict(method="arma", p=2, q=1, pooling="run"),
        "arma_global_p2q1": dict(method="arma", p=2, q=1, pooling="global"),
    }

    results: Dict[str, Dict[str, object]] = {}
    for name, cfg in cases.items():
        results[name] = _time_case(
            X=X,
            Y=Y,
            runs=runs,
            method=cfg["method"],
            p=int(cfg["p"]),
            q=int(cfg["q"]),
            pooling=str(cfg["pooling"]),
            repeats=args.repeats,
            warmup=args.warmup,
        )

    baseline = float(results["ar_run_p2"]["total_median_s"])
    relative = {k: baseline / float(v["total_median_s"]) for k, v in results.items()}

    report = {
        "config": {
            "n_runs": args.n_runs,
            "n_timepoints": args.n_timepoints,
            "n_regressors": args.n_regressors,
            "n_voxels": args.n_voxels,
            "seed": args.seed,
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
        "results": results,
        "relative_speed_vs_ar_run_p2": relative,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    with open(args.output, "w", encoding="utf-8") as fobj:
        json.dump(report, fobj, indent=2, sort_keys=True)
    print(f"\nWrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
