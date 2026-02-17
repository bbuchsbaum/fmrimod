#!/usr/bin/env python
"""Benchmark fmrimod optimization levers on synthetic multi-run GLM data."""

from __future__ import annotations

import argparse
import json
import os
from statistics import median
import time
from typing import Dict, List, Mapping, Sequence

import numpy as np

from fmrimod.glm.strategies import fit_runwise
from fmrimod.model.config import FmriLmConfig


class _BenchDataset:
    def __init__(self, ys: Sequence[np.ndarray]):
        self._ys = [np.asarray(y, dtype=np.float64) for y in ys]
        self.n_timepoints = [y.shape[0] for y in self._ys]
        self.n_runs = len(self._ys)

    def get_data(self, run: int) -> np.ndarray:
        return self._ys[run]

    def get_censor(self, run: int):
        return None


class _BenchModel:
    def __init__(self, xs: Sequence[np.ndarray], ys: Sequence[np.ndarray]):
        self._xs = [np.asarray(x, dtype=np.float64) for x in xs]
        self.dataset = _BenchDataset(ys)
        self.n_runs = len(self._xs)

    def design_matrix_array(self, run: int) -> np.ndarray:
        return self._xs[run]


def _build_model(
    *,
    n_runs: int,
    n_timepoints: int,
    n_regressors: int,
    n_voxels: int,
    noise_sd: float,
    seed: int,
    repeat_design: bool,
) -> _BenchModel:
    rng = np.random.default_rng(seed)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    X_shared = None
    if repeat_design:
        X_shared = np.column_stack(
            [np.ones(n_timepoints), rng.standard_normal((n_timepoints, n_regressors - 1))]
        ).astype(np.float64)

    for _ in range(n_runs):
        if X_shared is None:
            X = np.column_stack(
                [np.ones(n_timepoints), rng.standard_normal((n_timepoints, n_regressors - 1))]
            ).astype(np.float64)
        else:
            X = X_shared.copy()
        beta = rng.standard_normal((n_regressors, n_voxels)).astype(np.float64)
        Y = X @ beta + rng.standard_normal((n_timepoints, n_voxels)).astype(np.float64) * noise_sd
        xs.append(X)
        ys.append(Y)
    return _BenchModel(xs, ys)


def _time_scenario(
    model: _BenchModel,
    cfg: FmriLmConfig,
    kwargs: Mapping[str, object],
    repeats: int,
    warmup: int,
) -> Dict[str, object]:
    for _ in range(warmup):
        fit_runwise(model, cfg, **kwargs)

    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fit_runwise(model, cfg, **kwargs)
        times.append(time.perf_counter() - t0)
    return {"runs_s": times, "median_s": median(times)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark fmrimod runwise optimization levers."
    )
    parser.add_argument("--n-runs", type=int, default=6)
    parser.add_argument("--n-timepoints", type=int, default=240)
    parser.add_argument("--n-regressors", type=int, default=8)
    parser.add_argument("--n-voxels", type=int, default=4000)
    parser.add_argument("--noise-sd", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--repeat-design",
        action="store_true",
        help="Use identical design matrices across runs (helps projection cache).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cross_testing/reports/fmrimod_levers_benchmark.json",
    )
    args = parser.parse_args(argv)

    model = _build_model(
        n_runs=args.n_runs,
        n_timepoints=args.n_timepoints,
        n_regressors=args.n_regressors,
        n_voxels=args.n_voxels,
        noise_sd=args.noise_sd,
        seed=args.seed,
        repeat_design=args.repeat_design,
    )
    cfg = FmriLmConfig()
    cpu_count = max(1, (os.cpu_count() or 1))
    parallel_jobs = min(cpu_count, args.n_runs)

    scenarios = {
        "baseline_f64_serial": {
            "n_jobs": 1,
            "compute_dtype": "float64",
            "cache_projections": False,
        },
        "parallel_f64": {
            "n_jobs": parallel_jobs,
            "blas_threads": 1,
            "compute_dtype": "float64",
            "cache_projections": False,
        },
        "cached_f64_serial": {
            "n_jobs": 1,
            "compute_dtype": "float64",
            "cache_projections": True,
        },
        "f32_serial": {
            "n_jobs": 1,
            "compute_dtype": "float32",
            "cache_projections": False,
        },
        "f32_parallel": {
            "n_jobs": parallel_jobs,
            "blas_threads": 1,
            "compute_dtype": "float32",
            "cache_projections": False,
        },
    }

    results: Dict[str, Dict[str, object]] = {}
    for name, kw in scenarios.items():
        results[name] = _time_scenario(
            model,
            cfg,
            kw,
            repeats=args.repeats,
            warmup=args.warmup,
        )

    baseline = float(results["baseline_f64_serial"]["median_s"])
    speedups = {
        name: (baseline / float(payload["median_s"]))
        for name, payload in results.items()
    }

    report = {
        "config": {
            "n_runs": args.n_runs,
            "n_timepoints": args.n_timepoints,
            "n_regressors": args.n_regressors,
            "n_voxels": args.n_voxels,
            "noise_sd": args.noise_sd,
            "seed": args.seed,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "repeat_design": args.repeat_design,
            "parallel_jobs": parallel_jobs,
        },
        "results": results,
        "speedup_vs_baseline": speedups,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    with open(args.output, "w", encoding="utf-8") as fobj:
        json.dump(report, fobj, indent=2, sort_keys=True)
    print(f"\nWrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

