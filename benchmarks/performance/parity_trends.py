"""Generate lightweight performance trend rows for parity-critical stages."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from fmrimod.ar import estimate_ar
from fmrimod.glm.solver import fast_lm_matrix, fast_preproject
from fmrimod.single import estimate_single_trial


@dataclass(frozen=True)
class PerfRow:
    """One trendable benchmark row."""

    stage: str
    seconds: float
    n_time: int
    n_predictors: int
    n_voxels: int
    status: str
    details: dict[str, Any]


def _time(stage: str, func: Callable[[], Any], **shape: int) -> tuple[PerfRow, Any]:
    start = time.perf_counter()
    value = func()
    seconds = time.perf_counter() - start
    return (
        PerfRow(
            stage=stage,
            seconds=seconds,
            n_time=shape.get("n_time", 0),
            n_predictors=shape.get("n_predictors", 0),
            n_voxels=shape.get("n_voxels", 0),
            status="ok",
            details={},
        ),
        value,
    )


def run_trends(seed: int = 2029) -> list[PerfRow]:
    """Run deterministic micro-benchmarks for parity performance tracking."""

    rng = np.random.default_rng(seed)
    n_time = 160
    n_predictors = 7
    n_voxels = 512

    design_row, X = _time(
        "design_build",
        lambda: np.column_stack(
            [
                np.ones(n_time),
                np.linspace(-1.0, 1.0, n_time),
                *[rng.normal(size=n_time) for _ in range(n_predictors - 2)],
            ]
        ),
        n_time=n_time,
        n_predictors=n_predictors,
        n_voxels=0,
    )
    beta = rng.normal(scale=0.25, size=(n_predictors, n_voxels))
    Y = X @ beta + rng.normal(scale=0.2, size=(n_time, n_voxels))

    fit_row, fit = _time(
        "glm_fit",
        lambda: fast_lm_matrix(X, Y, fast_preproject(X)),
        n_time=n_time,
        n_predictors=n_predictors,
        n_voxels=n_voxels,
    )

    contrast = np.zeros(n_predictors)
    contrast[1] = 1.0
    contrast_row, _ = _time(
        "contrast",
        lambda: contrast @ fit.betas,
        n_time=n_time,
        n_predictors=n_predictors,
        n_voxels=n_voxels,
    )

    residuals = Y - X @ fit.betas
    ar_row, phi = _time(
        "ar_whitening",
        lambda: estimate_ar(residuals, order=1, voxelwise=False),
        n_time=n_time,
        n_predictors=n_predictors,
        n_voxels=n_voxels,
    )
    ar_row = PerfRow(
        **{
            **asdict(ar_row),
            "details": {"estimated_phi": np.asarray(phi).tolist()},
        }
    )

    run_effects = np.stack([fit.betas[1], fit.betas[1] + rng.normal(scale=0.01, size=n_voxels)])
    run_vars = np.full_like(run_effects, 0.04)
    combine_row, _ = _time(
        "run_combination",
        lambda: np.sum(run_effects / run_vars, axis=0) / np.sum(1.0 / run_vars, axis=0),
        n_time=2,
        n_predictors=1,
        n_voxels=n_voxels,
    )

    n_trials = 10
    X_lss = np.zeros((n_time, n_trials), dtype=np.float64)
    for trial, onset in enumerate(np.linspace(3, n_time - 6, n_trials, dtype=int)):
        X_lss[onset : onset + 3, trial] = [0.5, 1.0, 0.4]
    Y_lss = X_lss @ rng.normal(size=(n_trials, 96)) + rng.normal(scale=0.05, size=(n_time, 96))
    lss_row, _ = _time(
        "lss",
        lambda: estimate_single_trial(Y_lss, X_lss, method="lss", include_intercept=True),
        n_time=n_time,
        n_predictors=n_trials,
        n_voxels=96,
    )

    return [design_row, fit_row, contrast_row, ar_row, combine_row, lss_row]


def render(rows: list[PerfRow], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "parity_performance_trends.json"
    payload = {
        "name": "parity_performance_trends",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate_policy": "correctness-gated; performance tracked as trend",
        "rows": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    rows = run_trends()
    render(rows, Path(__file__).resolve().parent / "results")


if __name__ == "__main__":
    main()

