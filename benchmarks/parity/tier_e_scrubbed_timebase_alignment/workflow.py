"""Scrubbed-volume timebase alignment stress benchmark.

This Tier E canary targets a first-level GLM trap shared by fmrimod and
Nilearn users: after scrubbing/censoring volumes, the design must remain on the
original acquisition timebase and then drop the same rows as the data. Rebuilding
the design on a compacted post-scrub timeline gives the right shape and a wrong
hypothesis.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import run_glm
from numpy.typing import NDArray

import fmrimod as fm
from fmrimod.model.config import FmriLmConfig

Array = NDArray[np.float64]
SCHEMA_VERSION = "scrubbed-timebase-alignment/v1"
MAX_VOXELS = 96
N_SCANS = 160
CONTRAST_NAME = "task_vs_baseline"
COLUMNS = ("intercept", "task", "motion", "drift")


@dataclass(frozen=True)
class ScrubbedInputs:
    """Shared synthetic fixture for the scrubbed timebase trap."""

    original_design: Array
    compacted_design: Array
    data: Array
    keep_mask: NDArray[np.bool_]
    true_task_effect_median: float


@dataclass(frozen=True)
class EngineResult:
    """One engine's task-contrast output."""

    status: str
    effect_median: float | None
    stat_median: float | None
    finite_effect_fraction: float
    finite_stat_fraction: float
    touched_columns: tuple[str, ...]
    warning_messages: tuple[str, ...]
    exception_type: str | None = None
    exception_message: str | None = None


@dataclass(frozen=True)
class ScrubbedReport:
    """Structured report for the scrubbed-timebase benchmark."""

    schema_version: str
    name: str
    status: str
    n_scans_original: int
    n_scans_kept: int
    scrubbed_fraction: float
    fmrimod: EngineResult
    nilearn_aligned: EngineResult
    compacted_timebase: EngineResult
    comparisons: dict[str, float | None]
    pain_point: dict[str, Any]
    verdict: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _center_scale(values: Array) -> Array:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - float(np.mean(arr))
    scale = float(np.linalg.norm(arr))
    return arr if scale == 0.0 else arr / scale


def _boxcar(n_scans: int, starts: tuple[int, ...], width: int) -> Array:
    arr = np.zeros(n_scans, dtype=np.float64)
    for start in starts:
        arr[start : min(start + width, n_scans)] = 1.0
    return _center_scale(arr)


def _design(n_scans: int) -> Array:
    time = np.linspace(-1.0, 1.0, n_scans, dtype=np.float64)
    task = _boxcar(n_scans, (12, 37, 66, 101, 132), width=7)
    motion = _center_scale(np.sin(np.linspace(0.0, 8.0 * np.pi, n_scans)))
    drift = _center_scale(time + 0.35 * time * time)
    return np.column_stack([np.ones(n_scans), task, motion, drift]).astype(np.float64)


def _keep_mask(n_scans: int) -> NDArray[np.bool_]:
    keep = np.ones(n_scans, dtype=bool)
    scrubbed = np.array(
        [0, 1, 2, 18, 19, 20, 44, 45, 72, 73, 74, 115, 116, 142, 143],
        dtype=int,
    )
    keep[scrubbed] = False
    return keep


def _fraction(mask: Array) -> float:
    arr = np.asarray(mask, dtype=bool)
    return float(np.mean(arr)) if arr.size else 1.0


def _median(values: Array) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return None
    return float(np.median(arr[finite]))


def _max_abs_delta(candidate: Array, reference: Array) -> float | None:
    if candidate.shape != reference.shape or candidate.size == 0:
        return None
    finite = np.isfinite(candidate) & np.isfinite(reference)
    if not np.any(finite):
        return None
    return float(np.max(np.abs(candidate[finite] - reference[finite])))


def _median_abs_delta(candidate: Array, reference: Array) -> float | None:
    if candidate.shape != reference.shape or candidate.size == 0:
        return None
    finite = np.isfinite(candidate) & np.isfinite(reference)
    if not np.any(finite):
        return None
    return float(np.median(np.abs(candidate[finite] - reference[finite])))


def load_inputs(
    max_voxels: int = MAX_VOXELS,
    seed: int = 20260514,
) -> ScrubbedInputs:
    """Create original-timebase data plus aligned and compacted designs."""

    rng = np.random.default_rng(seed)
    n_voxels = min(int(max_voxels), MAX_VOXELS)
    original_design = _design(N_SCANS)
    keep = _keep_mask(N_SCANS)
    compacted_design = _design(int(np.sum(keep)))

    ramp = np.linspace(0.8, 1.3, n_voxels, dtype=np.float64)
    betas = np.zeros((len(COLUMNS), n_voxels), dtype=np.float64)
    betas[COLUMNS.index("intercept")] = 100.0 + rng.normal(
        scale=0.2,
        size=n_voxels,
    )
    betas[COLUMNS.index("task")] = 1.4 * ramp
    betas[COLUMNS.index("motion")] = rng.normal(scale=0.08, size=n_voxels)
    betas[COLUMNS.index("drift")] = rng.normal(scale=0.10, size=n_voxels)
    data = original_design @ betas + rng.normal(
        scale=0.025,
        size=(N_SCANS, n_voxels),
    )
    return ScrubbedInputs(
        original_design=original_design,
        compacted_design=compacted_design,
        data=data.astype(np.float64),
        keep_mask=keep,
        true_task_effect_median=float(np.median(betas[COLUMNS.index("task")])),
    )


def _engine_result(
    *,
    status: str,
    effect: Array,
    stat: Array,
    touched_columns: tuple[str, ...],
    warnings_seen: tuple[str, ...],
    exception: Exception | None = None,
) -> EngineResult:
    return EngineResult(
        status=status,
        effect_median=_median(effect),
        stat_median=_median(stat),
        finite_effect_fraction=_fraction(np.isfinite(effect)),
        finite_stat_fraction=_fraction(np.isfinite(stat)),
        touched_columns=touched_columns,
        warning_messages=warnings_seen,
        exception_type=None if exception is None else type(exception).__name__,
        exception_message=None if exception is None else str(exception),
    )


def fmrimod_probe(design: Array, data: Array) -> tuple[EngineResult, Array, Array]:
    """Fit fmrimod on scrubbed rows with the original-timebase design."""

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            design_frame = pd.DataFrame(design, columns=COLUMNS)
            fit = fm.fit_glm_from_matrix(
                design,
                data,
                model=design_frame,
                cfg=FmriLmConfig(),
            )
            result = fit.contrast(np.array([0.0, 1.0, 0.0, 0.0]), name=CONTRAST_NAME)
    except Exception as exc:  # pragma: no cover - defensive report path
        empty = np.array([], dtype=np.float64)
        engine = _engine_result(
            status="exception",
            effect=empty,
            stat=empty,
            touched_columns=(),
            warnings_seen=(),
            exception=exc,
        )
        return engine, empty, empty

    effect = np.asarray(result.estimate, dtype=np.float64)
    stat = np.asarray(result.stat, dtype=np.float64)
    engine = _engine_result(
        status="ok",
        effect=effect,
        stat=stat,
        touched_columns=tuple(result.touched_columns),
        warnings_seen=tuple(str(w.message) for w in captured),
    )
    return engine, effect, stat


def nilearn_probe(
    design: Array,
    data: Array,
) -> tuple[EngineResult, Array, Array]:
    """Fit Nilearn on a supplied design/data pair."""

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            labels, estimates = run_glm(data, design, noise_model="ols")
            result = compute_contrast(
                labels,
                estimates,
                np.array([0.0, 1.0, 0.0, 0.0]),
                stat_type="t",
            )
    except Exception as exc:  # pragma: no cover - defensive report path
        empty = np.array([], dtype=np.float64)
        engine = _engine_result(
            status="exception",
            effect=empty,
            stat=empty,
            touched_columns=(),
            warnings_seen=(),
            exception=exc,
        )
        return engine, empty, empty

    effect = np.asarray(result.effect_size(), dtype=np.float64)
    stat = np.asarray(result.stat(), dtype=np.float64)
    engine = _engine_result(
        status="ok",
        effect=effect,
        stat=stat,
        touched_columns=("task",),
        warnings_seen=tuple(str(w.message) for w in captured),
    )
    return engine, effect, stat


def run_benchmark(
    max_voxels: int = MAX_VOXELS,
    seed: int = 20260514,
) -> dict[str, Any]:
    """Run the scrubbed-volume timebase alignment canary."""

    inputs = load_inputs(max_voxels=max_voxels, seed=seed)
    scrubbed_design = inputs.original_design[inputs.keep_mask]
    scrubbed_data = inputs.data[inputs.keep_mask]

    f_engine, f_effect, f_stat = fmrimod_probe(scrubbed_design, scrubbed_data)
    n_engine, n_effect, n_stat = nilearn_probe(scrubbed_design, scrubbed_data)
    c_engine, c_effect, c_stat = nilearn_probe(
        inputs.compacted_design,
        scrubbed_data,
    )

    aligned_effect_delta = _max_abs_delta(f_effect, n_effect)
    aligned_stat_delta = _max_abs_delta(f_stat, n_stat)
    compacted_effect_drift = _median_abs_delta(n_effect, c_effect)
    compacted_stat_drift = _median_abs_delta(n_stat, c_stat)
    aligned_ok = (
        aligned_effect_delta is not None
        and aligned_effect_delta < 1e-8
        and aligned_stat_delta is not None
        and aligned_stat_delta < 1e-5
    )
    trap_observed = (
        compacted_effect_drift is not None
        and compacted_effect_drift > 0.15
        and compacted_stat_drift is not None
        and compacted_stat_drift > 5.0
    )
    status = "pass" if aligned_ok and trap_observed else "fail"

    report = ScrubbedReport(
        schema_version=SCHEMA_VERSION,
        name="tier_e_scrubbed_timebase_alignment",
        status=status,
        n_scans_original=int(inputs.data.shape[0]),
        n_scans_kept=int(np.sum(inputs.keep_mask)),
        scrubbed_fraction=float(1.0 - np.mean(inputs.keep_mask)),
        fmrimod=f_engine,
        nilearn_aligned=n_engine,
        compacted_timebase=c_engine,
        comparisons={
            "aligned_effect_delta": aligned_effect_delta,
            "aligned_stat_delta": aligned_stat_delta,
            "compacted_timebase_effect_median_abs_delta": compacted_effect_drift,
            "compacted_timebase_stat_median_abs_delta": compacted_stat_drift,
        },
        pain_point={
            "observed": bool(trap_observed),
            "verdict": (
                "a compacted post-scrub timeline has the right row count but "
                "targets a shifted task regressor"
            ),
        },
        verdict=(
            "fmrimod matches the correctly row-filtered Nilearn oracle; the "
            "compacted-timebase design visibly drifts"
            if status == "pass"
            else "scrubbed timebase alignment canary failed"
        ),
    )
    return _json_safe(report)


def render(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Write JSON and Markdown reports for the benchmark."""

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "scrubbed_timebase_alignment_report.json"
    md_path = out_dir / "REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Scrubbed Timebase Alignment",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Rows",
        "",
        f"- Original scans: `{report['n_scans_original']}`",
        f"- Kept scans: `{report['n_scans_kept']}`",
        f"- Scrubbed fraction: `{report['scrubbed_fraction']:.3f}`",
        "",
        "## Comparisons",
        "",
        "| comparison | value |",
        "| --- | ---: |",
    ]
    for name, value in report["comparisons"].items():
        rendered = "null" if value is None else f"{value:.6g}"
        lines.append(f"| {name} | {rendered} |")
    lines.extend(
        [
            "",
            "## Pain Point",
            "",
            report["pain_point"]["verdict"],
            "",
            report["verdict"],
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="Directory where the scrubbed-timebase report is written.",
    )
    parser.add_argument("--max-voxels", type=int, default=MAX_VOXELS)
    args = parser.parse_args(argv)

    report = run_benchmark(max_voxels=args.max_voxels)
    render(report, args.out_dir)
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
