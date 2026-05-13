"""Generate lightweight performance trend rows for parity-critical stages."""

from __future__ import annotations

import json
import platform
import subprocess
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
    generated_at: str
    git_sha: str
    hardware_tag: str
    seconds: float
    seconds_iqr: float
    repetitions: int
    n_time: int
    n_predictors: int
    n_voxels: int
    status: str
    details: dict[str, Any]


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _hardware_tag() -> str:
    return f"{platform.system()}-{platform.machine()}-{platform.processor() or 'unknown'}"


def _time(
    stage: str,
    func: Callable[[], Any],
    *,
    generated_at: str,
    git_sha: str,
    hardware_tag: str,
    repetitions: int,
    **shape: int,
) -> tuple[PerfRow, Any]:
    timings = []
    value = None
    for _ in range(repetitions):
        start = time.perf_counter()
        value = func()
        timings.append(time.perf_counter() - start)
    q25, q75 = np.quantile(np.asarray(timings, dtype=np.float64), [0.25, 0.75])
    value = func()
    seconds = float(np.median(timings))
    return (
        PerfRow(
            stage=stage,
            generated_at=generated_at,
            git_sha=git_sha,
            hardware_tag=hardware_tag,
            seconds=seconds,
            seconds_iqr=float(q75 - q25),
            repetitions=repetitions,
            n_time=shape.get("n_time", 0),
            n_predictors=shape.get("n_predictors", 0),
            n_voxels=shape.get("n_voxels", 0),
            status="ok",
            details={},
        ),
        value,
    )


def run_trends(
    seed: int = 2029,
    repetitions: int = 5,
    generated_at: str | None = None,
    git_sha: str | None = None,
    hardware_tag: str | None = None,
) -> list[PerfRow]:
    """Run deterministic micro-benchmarks for parity performance tracking."""

    if repetitions < 5:
        raise ValueError("performance trend rows require at least 5 repetitions")
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    git_sha = git_sha or _git_sha()
    hardware_tag = hardware_tag or _hardware_tag()
    rng = np.random.default_rng(seed)
    n_time = 160
    n_predictors = 7
    n_voxels = 512
    design_noise = rng.normal(size=(n_time, n_predictors - 2))

    design_row, X = _time(
        "design_build",
        lambda: np.column_stack(
            [
                np.ones(n_time),
                np.linspace(-1.0, 1.0, n_time),
                *[design_noise[:, idx] for idx in range(n_predictors - 2)],
            ]
        ),
        generated_at=generated_at,
        git_sha=git_sha,
        hardware_tag=hardware_tag,
        repetitions=repetitions,
        n_time=n_time,
        n_predictors=n_predictors,
        n_voxels=0,
    )
    beta = rng.normal(scale=0.25, size=(n_predictors, n_voxels))
    Y = X @ beta + rng.normal(scale=0.2, size=(n_time, n_voxels))

    fit_row, fit = _time(
        "glm_fit",
        lambda: fast_lm_matrix(X, Y, fast_preproject(X)),
        generated_at=generated_at,
        git_sha=git_sha,
        hardware_tag=hardware_tag,
        repetitions=repetitions,
        n_time=n_time,
        n_predictors=n_predictors,
        n_voxels=n_voxels,
    )

    contrast = np.zeros(n_predictors)
    contrast[1] = 1.0
    contrast_row, _ = _time(
        "contrast",
        lambda: contrast @ fit.betas,
        generated_at=generated_at,
        git_sha=git_sha,
        hardware_tag=hardware_tag,
        repetitions=repetitions,
        n_time=n_time,
        n_predictors=n_predictors,
        n_voxels=n_voxels,
    )

    residuals = Y - X @ fit.betas
    ar_row, phi = _time(
        "ar_whitening",
        lambda: estimate_ar(residuals, order=1, voxelwise=False),
        generated_at=generated_at,
        git_sha=git_sha,
        hardware_tag=hardware_tag,
        repetitions=repetitions,
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
        generated_at=generated_at,
        git_sha=git_sha,
        hardware_tag=hardware_tag,
        repetitions=repetitions,
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
        generated_at=generated_at,
        git_sha=git_sha,
        hardware_tag=hardware_tag,
        repetitions=repetitions,
        n_time=n_time,
        n_predictors=n_trials,
        n_voxels=96,
    )

    return [design_row, fit_row, contrast_row, ar_row, combine_row, lss_row]


def render(rows: list[PerfRow], out_dir: Path, *, append: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "parity_performance_trends.json"
    history_path = out_dir / "parity_performance_trends.jsonl"
    mode = "a" if append else "w"
    with history_path.open(mode) as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    payload = {
        "name": "parity_performance_trends",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate_policy": "correctness-gated; performance tracked as trend",
        "history_file": history_path.name,
        "summary": "rows contain median seconds and IQR over repeated measurements",
        "rows": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    rows = run_trends()
    render(rows, Path(__file__).resolve().parent / "results")


if __name__ == "__main__":
    main()
