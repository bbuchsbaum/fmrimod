"""Scrubbed-volume timebase alignment parity stress benchmark.

This Tier E workflow targets a common first-level GLM trap: after censoring
volumes, the design must stay on the original acquisition timebase and then
drop the same rows as the data. Rebuilding the design on a compacted
post-scrub timeline has the right shape, but it changes the modeled
hypothesis.

The fmrimod side uses the public seam ``fmri_dataset(..., censor=...) ->
fmri_lm -> semantic condition contrast``. The Nilearn reference uses the
same realized fmrimod design with the same rows removed. A second Nilearn fit
uses a compacted event timeline to quantify the pain point that fmrimod should
make hard to hit.
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
from fmrimod.contrast import condition
from fmrimod.design import DesignColumns
from fmrimod.spec import drift, hrf

Array = NDArray[np.float64]
SCHEMA_VERSION = "scrubbed-timebase-alignment/v1"
MAX_VOXELS = 96
N_SCANS = 180
TR = 2.0
CONTRAST_NAME = "A_minus_B"
ALIGNED_EFFECT_ATOL = 1e-8
ALIGNED_STAT_ATOL = 1e-6
COMPRESSED_EFFECT_FLOOR = 0.15
COMPRESSED_STAT_FLOOR = 0.10


@dataclass(frozen=True)
class ScrubbedInputs:
    """Shared synthetic fixture for the scrubbed-timebase stress case."""

    events: pd.DataFrame
    compressed_events: pd.DataFrame
    data: Array
    design: pd.DataFrame
    compressed_design: pd.DataFrame
    design_columns: DesignColumns
    compressed_design_columns: DesignColumns
    spec: Any
    semantic_contrast: Any
    contrast_weights: Array
    compressed_contrast_weights: Array
    censor: NDArray[np.bool_]
    keep_mask: NDArray[np.bool_]
    onset_shifts_seconds: Array


@dataclass(frozen=True)
class EngineResult:
    """One engine's task-contrast output."""

    status: str
    effect_median: float | None
    stat_median: float | None
    finite_effect_fraction: float
    finite_stat_fraction: float
    fitted_rows: int | None
    residual_df: float | None
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
    alignment_contract: dict[str, Any]
    fmrimod: EngineResult
    nilearn_aligned: EngineResult
    nilearn_compressed_timebase: EngineResult
    comparisons: dict[str, float | None]
    pain_point: dict[str, Any]
    win_ladder: tuple[str, ...]
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


def _make_spec() -> Any:
    return hrf("trial_type", basis="spm", norm="spm") + drift("poly", degree=2)


def _make_events(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.array(["A", "B"] * 18)
    onsets = np.linspace(10.0, N_SCANS * TR - 30.0, len(labels))
    onsets = onsets + rng.uniform(-1.0, 1.0, size=len(labels))
    events = pd.DataFrame(
        {
            "onset": onsets,
            "duration": np.where(labels == "A", 2.0, 3.0),
            "trial_type": labels,
            "run": 1,
        }
    )
    return events.sort_values("onset").reset_index(drop=True)


def _make_censor() -> NDArray[np.bool_]:
    censor = np.zeros(N_SCANS, dtype=bool)
    for start, stop in ((38, 47), (74, 83), (116, 126), (150, 156)):
        censor[start:stop] = True
    censor[[20, 21, 98, 99]] = True
    return censor


def _realize_design(
    spec: Any,
    events: pd.DataFrame,
    n_scans: int,
) -> tuple[pd.DataFrame, DesignColumns]:
    dataset = fm.fmri_dataset(
        np.zeros((n_scans, 1), dtype=np.float64),
        tr=TR,
        events=events,
    )
    fit = fm.fmri_lm(spec, dataset)
    return fit.model.design_matrix(run=0), fit.design_columns()


def _compress_events(
    events: pd.DataFrame,
    censor: NDArray[np.bool_],
) -> tuple[pd.DataFrame, Array]:
    """Move event onsets onto a naive post-scrub compressed timebase."""

    n_keep = int(np.sum(~censor))
    rows: list[dict[str, Any]] = []
    shifts: list[float] = []
    for event in events.to_dict("records"):
        onset = float(event["onset"])
        source_scan = int(np.floor(onset / TR))
        source_scan = max(0, min(source_scan, len(censor)))
        removed_before_onset = int(np.sum(censor[:source_scan]))
        shift = removed_before_onset * TR
        shifted_onset = onset - shift
        if 0.0 <= shifted_onset < n_keep * TR:
            shifted = dict(event)
            shifted["onset"] = shifted_onset
            rows.append(shifted)
            shifts.append(float(shift))

    compressed = pd.DataFrame(rows, columns=list(events.columns))
    compressed = compressed.sort_values("onset").reset_index(drop=True)
    return compressed, np.asarray(shifts, dtype=np.float64)


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


def _engine_result(
    *,
    status: str,
    effect: Array,
    stat: Array,
    fitted_rows: int | None,
    residual_df: float | None,
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
        fitted_rows=fitted_rows,
        residual_df=residual_df,
        touched_columns=touched_columns,
        warning_messages=warnings_seen,
        exception_type=None if exception is None else type(exception).__name__,
        exception_message=None if exception is None else str(exception),
    )


def load_inputs(
    max_voxels: int = MAX_VOXELS,
    seed: int = 20260514,
) -> ScrubbedInputs:
    """Create original-timebase data plus aligned and compressed designs."""

    n_voxels = max(1, min(int(max_voxels), MAX_VOXELS))
    rng = np.random.default_rng(seed)
    spec = _make_spec()
    semantic_contrast = condition("A", term="trial_type") - condition(
        "B",
        term="trial_type",
    )
    events = _make_events(seed)
    censor = _make_censor()
    keep = ~censor

    design, design_columns = _realize_design(spec, events, N_SCANS)
    contrast_weights = np.asarray(
        semantic_contrast.resolve(design_columns),
        dtype=np.float64,
    )

    compressed_events, shifts = _compress_events(events, censor)
    compressed_design, compressed_design_columns = _realize_design(
        spec,
        compressed_events,
        int(np.sum(keep)),
    )
    compressed_contrast_weights = np.asarray(
        semantic_contrast.resolve(compressed_design_columns),
        dtype=np.float64,
    )

    full_design = design.to_numpy(dtype=np.float64)
    betas = np.zeros((full_design.shape[1], n_voxels), dtype=np.float64)
    ramp = np.linspace(0.8, 1.2, n_voxels, dtype=np.float64)
    pos = np.flatnonzero(contrast_weights > 0.0)
    neg = np.flatnonzero(contrast_weights < 0.0)
    if len(pos) != 1 or len(neg) != 1:
        raise ValueError(
            "Expected one positive and one negative condition column for A - B"
        )
    betas[pos[0]] = 1.0 * ramp
    betas[neg[0]] = 0.25 * ramp
    nuisance_rows = [
        idx for idx in range(full_design.shape[1]) if idx not in (*pos, *neg)
    ]
    for idx, row in enumerate(nuisance_rows):
        if idx == 0:
            betas[row] = 2.0 * ramp
        elif idx == 1:
            betas[row] = -1.5 * ramp
        else:
            betas[row] = 100.0 + rng.normal(scale=0.1, size=n_voxels)

    data = full_design @ betas
    data = data + rng.normal(scale=0.05, size=(N_SCANS, n_voxels))

    return ScrubbedInputs(
        events=events,
        compressed_events=compressed_events,
        data=data.astype(np.float64),
        design=design,
        compressed_design=compressed_design,
        design_columns=design_columns,
        compressed_design_columns=compressed_design_columns,
        spec=spec,
        semantic_contrast=semantic_contrast,
        contrast_weights=contrast_weights,
        compressed_contrast_weights=compressed_contrast_weights,
        censor=censor,
        keep_mask=keep,
        onset_shifts_seconds=shifts,
    )


def fmrimod_pipeline(inputs: ScrubbedInputs) -> tuple[EngineResult, Array, Array]:
    """Run the public fmrimod path with dataset-level censoring."""

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            dataset = fm.fmri_dataset(
                inputs.data,
                tr=TR,
                events=inputs.events,
                censor=inputs.censor,
            )
            fit = fm.fmri_lm(inputs.spec, dataset)
            result = fit.contrast(inputs.semantic_contrast, name=CONTRAST_NAME)
    except Exception as exc:  # pragma: no cover - defensive report path
        empty = np.array([], dtype=np.float64)
        engine = _engine_result(
            status="exception",
            effect=empty,
            stat=empty,
            fitted_rows=None,
            residual_df=None,
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
        fitted_rows=int(np.sum(inputs.keep_mask)),
        residual_df=float(fit.residual_df),
        touched_columns=tuple(result.touched_columns),
        warnings_seen=tuple(str(w.message) for w in captured),
    )
    return engine, effect, stat


def nilearn_aligned_probe(
    inputs: ScrubbedInputs,
) -> tuple[EngineResult, Array, Array]:
    """Run Nilearn on the original-timebase design after row subsetting."""

    design = inputs.design.to_numpy(dtype=np.float64)[inputs.keep_mask]
    data = inputs.data[inputs.keep_mask]
    return _nilearn_probe(
        design=design,
        data=data,
        contrast_weights=inputs.contrast_weights,
        touched_columns=_touched_columns(
            tuple(inputs.design.columns),
            inputs.contrast_weights,
        ),
    )


def nilearn_compressed_timebase_probe(
    inputs: ScrubbedInputs,
) -> tuple[EngineResult, Array, Array]:
    """Run Nilearn on a naive compacted post-scrub event timeline."""

    design = inputs.compressed_design.to_numpy(dtype=np.float64)
    data = inputs.data[inputs.keep_mask]
    return _nilearn_probe(
        design=design,
        data=data,
        contrast_weights=inputs.compressed_contrast_weights,
        touched_columns=_touched_columns(
            tuple(inputs.compressed_design.columns),
            inputs.compressed_contrast_weights,
        ),
    )


def _nilearn_probe(
    *,
    design: Array,
    data: Array,
    contrast_weights: Array,
    touched_columns: tuple[str, ...],
) -> tuple[EngineResult, Array, Array]:
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            labels, estimates = run_glm(data, design, noise_model="ols")
            result = compute_contrast(
                labels,
                estimates,
                contrast_weights,
                stat_type="t",
            )
    except Exception as exc:  # pragma: no cover - defensive report path
        empty = np.array([], dtype=np.float64)
        engine = _engine_result(
            status="exception",
            effect=empty,
            stat=empty,
            fitted_rows=None,
            residual_df=None,
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
        fitted_rows=int(data.shape[0]),
        residual_df=float(data.shape[0] - np.linalg.matrix_rank(design)),
        touched_columns=touched_columns,
        warnings_seen=tuple(str(w.message) for w in captured),
    )
    return engine, effect, stat


def _touched_columns(
    columns: tuple[str, ...],
    weights: Array,
) -> tuple[str, ...]:
    return tuple(
        name for name, weight in zip(columns, weights) if abs(float(weight)) > 0.0
    )


def run_benchmark(
    max_voxels: int = MAX_VOXELS,
    seed: int = 20260514,
) -> dict[str, Any]:
    """Run the scrubbed-volume timebase alignment benchmark."""

    inputs = load_inputs(max_voxels=max_voxels, seed=seed)
    f_engine, f_effect, f_stat = fmrimod_pipeline(inputs)
    n_engine, n_effect, n_stat = nilearn_aligned_probe(inputs)
    c_engine, c_effect, c_stat = nilearn_compressed_timebase_probe(inputs)

    aligned_effect_delta = _max_abs_delta(f_effect, n_effect)
    aligned_stat_delta = _max_abs_delta(f_stat, n_stat)
    compressed_effect_drift = _median_abs_delta(n_effect, c_effect)
    compressed_stat_drift = _median_abs_delta(n_stat, c_stat)

    aligned_ok = (
        f_engine.status == "ok"
        and n_engine.status == "ok"
        and aligned_effect_delta is not None
        and aligned_effect_delta < ALIGNED_EFFECT_ATOL
        and aligned_stat_delta is not None
        and aligned_stat_delta < ALIGNED_STAT_ATOL
    )
    trap_observed = (
        c_engine.status == "ok"
        and compressed_effect_drift is not None
        and compressed_effect_drift > COMPRESSED_EFFECT_FLOOR
        and compressed_stat_drift is not None
        and compressed_stat_drift > COMPRESSED_STAT_FLOOR
    )
    status = "pass" if aligned_ok and trap_observed else "fail"

    n_original = int(inputs.data.shape[0])
    n_kept = int(np.sum(inputs.keep_mask))
    n_censored = n_original - n_kept
    report = ScrubbedReport(
        schema_version=SCHEMA_VERSION,
        name="tier_e_scrubbed_timebase_alignment",
        status=status,
        alignment_contract={
            "censor_policy": "row_subset_original_timebase",
            "original_timebase_rows": n_original,
            "kept_rows": n_kept,
            "censored_rows": n_censored,
            "scrubbed_fraction": float(n_censored / n_original),
            "design_columns": list(inputs.design.columns),
            "semantic_contrast": (
                "condition('A', term='trial_type') - "
                "condition('B', term='trial_type')"
            ),
        },
        fmrimod=f_engine,
        nilearn_aligned=n_engine,
        nilearn_compressed_timebase=c_engine,
        comparisons={
            "aligned_effect_max_abs_delta": aligned_effect_delta,
            "aligned_stat_max_abs_delta": aligned_stat_delta,
            "compressed_timebase_effect_median_abs_delta": compressed_effect_drift,
            "compressed_timebase_stat_median_abs_delta": compressed_stat_drift,
        },
        pain_point={
            "observed": bool(trap_observed),
            "trap": "compressed_post_scrub_timebase_rebuild",
            "thresholds": {
                "effect_median_abs_delta_gt": COMPRESSED_EFFECT_FLOOR,
                "stat_median_abs_delta_gt": COMPRESSED_STAT_FLOOR,
            },
            "max_event_onset_shift_seconds": _median_or_max(
                inputs.onset_shifts_seconds,
                kind="max",
            ),
            "median_event_onset_shift_seconds": _median_or_max(
                inputs.onset_shifts_seconds,
                kind="median",
            ),
            "verdict": (
                "The compressed timeline has the right row count but targets "
                "shifted event regressors."
            ),
        },
        win_ladder=(
            "fmrimod carries censoring on FmriDataset instead of rebuilding a "
            "shorter experiment clock",
            "fmrimod resolves A - B through declared design-column provenance",
            "Nilearn matches when handed the same row-subset design explicitly",
            "Nilearn also accepts the compressed-timebase design, exposing the "
            "manual bookkeeping pain point",
        ),
        verdict=(
            "fmrimod matches the correctly row-filtered Nilearn oracle and the "
            "compressed-timebase variant visibly drifts"
            if status == "pass"
            else "scrubbed timebase alignment benchmark failed"
        ),
    )
    return _json_safe(report)


def _median_or_max(values: Array, *, kind: str) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    if kind == "median":
        return float(np.median(arr))
    if kind == "max":
        return float(np.max(arr))
    raise ValueError(f"unknown summary kind: {kind}")


def render(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Write JSON and Markdown reports for the benchmark."""

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "scrubbed_timebase_alignment_report.json"
    md_path = out_dir / "REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    rows = report["alignment_contract"]
    lines = [
        "# Scrubbed Timebase Alignment",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Alignment Contract",
        "",
        f"- Policy: `{rows['censor_policy']}`",
        f"- Original rows: `{rows['original_timebase_rows']}`",
        f"- Kept rows: `{rows['kept_rows']}`",
        f"- Censored rows: `{rows['censored_rows']}`",
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
            "## Verdict",
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
