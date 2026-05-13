"""Parametric-modulator centering stress benchmark.

This Tier E workflow targets a user-facing trap rather than a solver
pathology. The raw trials are identical across two fits, but the RT
modulator is centered either within condition or globally. Both fmrimod
and Nilearn should agree on each realised design. The pain point is that
the main-effect contrast changes when centering scope changes, so
numerical parity alone is not enough: the modeled quantity moved.
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
from fmrimod.design.columns import DesignColumns
from fmrimod.spec import hrf

Array = NDArray[np.float64]
SCHEMA_VERSION = "parametric-centering-stress/v1"
MAX_VOXELS = 96
TR = 2.0
N_SCANS = 220
N_TRIALS = 60


@dataclass(frozen=True)
class CenteringInputs:
    """Shared data for the global-vs-within centering comparison."""

    events: pd.DataFrame
    data: Array
    rt_mean_a: float
    rt_mean_b: float
    rt_mean_global: float
    true_within_main_median: float


@dataclass(frozen=True)
class EngineResult:
    """One engine's outputs for one centering scope."""

    status: str
    main_effect_median: float | None
    main_stat_median: float | None
    slope_effect_median: float | None
    slope_stat_median: float | None
    finite_main_effect_fraction: float
    finite_main_stat_fraction: float
    finite_slope_effect_fraction: float
    finite_slope_stat_fraction: float
    warning_messages: tuple[str, ...]
    exception_type: str | None = None
    exception_message: str | None = None


@dataclass(frozen=True)
class CaseReport:
    """Structured report for one centering scope."""

    case_id: str
    purpose: str
    status: str
    centering_scope: str
    modulator_column: str
    design_shape: tuple[int, int]
    design_rank: int
    design_condition: float
    fmrimod: EngineResult
    nilearn: EngineResult
    comparisons: dict[str, float | None]
    verdict: str


@dataclass(frozen=True)
class CaseOutput:
    """Private holder for report plus arrays needed for cross-case deltas."""

    report: CaseReport
    fmrimod_main_effect: Array
    fmrimod_main_stat: Array


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


def _fraction(mask: Array) -> float:
    values = np.asarray(mask, dtype=bool)
    return float(np.mean(values)) if values.size else 1.0


def _max_abs_delta(candidate: Array, reference: Array) -> float | None:
    if candidate.shape != reference.shape or candidate.size == 0:
        return None
    mask = np.isfinite(candidate) & np.isfinite(reference)
    if not np.any(mask):
        return None
    return float(np.max(np.abs(candidate[mask] - reference[mask])))


def _median_abs_delta(candidate: Array, reference: Array) -> float | None:
    if candidate.shape != reference.shape or candidate.size == 0:
        return None
    mask = np.isfinite(candidate) & np.isfinite(reference)
    if not np.any(mask):
        return None
    return float(np.median(np.abs(candidate[mask] - reference[mask])))


def _median(values: Array) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    if not np.any(mask):
        return None
    return float(np.median(values[mask]))


def _make_events(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.array(["A", "B"] * (N_TRIALS // 2))
    onsets = np.linspace(8.0, N_SCANS * TR - 24.0, N_TRIALS, dtype=np.float64)
    rt = np.empty(N_TRIALS, dtype=np.float64)
    rt[labels == "A"] = rng.normal(1.25, 0.04, int(np.sum(labels == "A")))
    rt[labels == "B"] = rng.normal(0.55, 0.04, int(np.sum(labels == "B")))
    events = pd.DataFrame(
        {
            "onset": onsets,
            "duration": np.zeros(N_TRIALS, dtype=np.float64),
            "trial_type": labels,
            "rt": rt,
            "run": 1,
        }
    )
    events["rt_global_c"] = events["rt"] - float(events["rt"].mean())
    events["rt_within_c"] = (
        events["rt"]
        - events.groupby("trial_type", observed=True)["rt"].transform("mean")
    )
    return events


def _realize_fit(
    events: pd.DataFrame,
    data: Array,
    modulator_column: str,
):
    spec = hrf(
        "trial_type",
        basis="spm",
        norm="spm",
        modulators=[modulator_column],
    )
    dataset = fm.fmri_dataset(data, tr=TR, events=events)
    fit = fm.fmri_lm(spec, dataset)
    design = np.asarray(fit.model.design_matrix_array(run=0), dtype=np.float64)
    return fit, design, fit.design_columns()


def _task_indices(
    columns: DesignColumns,
    modulator_column: str,
) -> tuple[int, int, int, int]:
    main_a = columns.where(term="trial_type", level="A").one().index
    main_b = columns.where(term="trial_type", level="B").one().index
    slope_a = columns.where(
        term=f"trial_type:{modulator_column}",
        level="A",
    ).one().index
    slope_b = columns.where(
        term=f"trial_type:{modulator_column}",
        level="B",
    ).one().index
    return int(main_a), int(main_b), int(slope_a), int(slope_b)


def _contrasts(columns: DesignColumns, modulator_column: str) -> tuple[Array, Array]:
    main_a, main_b, slope_a, slope_b = _task_indices(columns, modulator_column)
    n_columns = len(columns)
    main = np.zeros(n_columns, dtype=np.float64)
    main[main_a] = 1.0
    main[main_b] = -1.0
    slope = np.zeros(n_columns, dtype=np.float64)
    slope[slope_a] = 1.0
    slope[slope_b] = -1.0
    return main, slope


def load_inputs(
    max_voxels: int = MAX_VOXELS,
    seed: int = 20260513,
) -> CenteringInputs:
    """Create an RT-imbalanced fixture with within-centered ground truth."""

    rng = np.random.default_rng(seed + 17)
    events = _make_events(seed)
    zero = np.zeros((N_SCANS, 1), dtype=np.float64)
    _, within_design, within_columns = _realize_fit(events, zero, "rt_within_c")
    main_a, main_b, slope_a, slope_b = _task_indices(
        within_columns,
        "rt_within_c",
    )
    n_voxels = min(int(max_voxels), MAX_VOXELS)
    ramp = np.linspace(0.8, 1.2, n_voxels, dtype=np.float64)
    betas = np.zeros((within_design.shape[1], n_voxels), dtype=np.float64)
    betas[main_a] = 1.00 * ramp
    betas[main_b] = 0.35 * ramp
    betas[slope_a] = 2.00 * ramp
    betas[slope_b] = -1.50 * ramp
    betas[-1] = 100.0 + rng.normal(scale=0.1, size=n_voxels)
    data = within_design @ betas + rng.normal(
        scale=1e-4,
        size=(N_SCANS, n_voxels),
    )
    means = events.groupby("trial_type", observed=True)["rt"].mean()
    return CenteringInputs(
        events=events,
        data=data.astype(np.float64),
        rt_mean_a=float(means["A"]),
        rt_mean_b=float(means["B"]),
        rt_mean_global=float(events["rt"].mean()),
        true_within_main_median=float(np.median(betas[main_a] - betas[main_b])),
    )


def _engine_result(
    *,
    status: str,
    main_effect: Array,
    main_stat: Array,
    slope_effect: Array,
    slope_stat: Array,
    warnings_seen: tuple[str, ...],
    exception: Exception | None = None,
) -> EngineResult:
    return EngineResult(
        status=status,
        main_effect_median=_median(main_effect),
        main_stat_median=_median(main_stat),
        slope_effect_median=_median(slope_effect),
        slope_stat_median=_median(slope_stat),
        finite_main_effect_fraction=_fraction(np.isfinite(main_effect)),
        finite_main_stat_fraction=_fraction(np.isfinite(main_stat)),
        finite_slope_effect_fraction=_fraction(np.isfinite(slope_effect)),
        finite_slope_stat_fraction=_fraction(np.isfinite(slope_stat)),
        warning_messages=warnings_seen,
        exception_type=None if exception is None else type(exception).__name__,
        exception_message=None if exception is None else str(exception),
    )


def fmrimod_probe(
    inputs: CenteringInputs,
    modulator_column: str,
) -> tuple[EngineResult, Array, Array, Array, Array, Array]:
    """Run the public fmrimod seam for one centering scope."""

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fit, design, columns = _realize_fit(
                inputs.events,
                inputs.data,
                modulator_column,
            )
            main_contrast, slope_contrast = _contrasts(columns, modulator_column)
            main = fit.contrast(main_contrast, name="A_minus_B_main")
            slope = fit.contrast(slope_contrast, name="A_minus_B_slope")
    except Exception as exc:  # pragma: no cover - defensive report path
        empty = np.array([], dtype=np.float64)
        engine = _engine_result(
            status="exception",
            main_effect=empty,
            main_stat=empty,
            slope_effect=empty,
            slope_stat=empty,
            warnings_seen=(),
            exception=exc,
        )
        return engine, empty, empty, empty, empty, empty

    warnings_seen = tuple(str(w.message) for w in captured)
    main_effect = np.asarray(main.estimate, dtype=np.float64)
    main_stat = np.asarray(main.stat, dtype=np.float64)
    slope_effect = np.asarray(slope.estimate, dtype=np.float64)
    slope_stat = np.asarray(slope.stat, dtype=np.float64)
    engine = _engine_result(
        status="ok",
        main_effect=main_effect,
        main_stat=main_stat,
        slope_effect=slope_effect,
        slope_stat=slope_stat,
        warnings_seen=warnings_seen,
    )
    return engine, main_effect, main_stat, slope_effect, slope_stat, design


def nilearn_probe(
    inputs: CenteringInputs,
    modulator_column: str,
    design: Array,
) -> tuple[EngineResult, Array, Array, Array, Array]:
    """Run Nilearn on fmrimod's realised design for one centering scope."""

    zero = np.zeros((N_SCANS, 1), dtype=np.float64)
    _, _, columns = _realize_fit(inputs.events, zero, modulator_column)
    main_contrast, slope_contrast = _contrasts(columns, modulator_column)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            labels, estimates = run_glm(inputs.data, design, noise_model="ols")
            main = compute_contrast(
                labels,
                estimates,
                main_contrast,
                stat_type="t",
            )
            slope = compute_contrast(
                labels,
                estimates,
                slope_contrast,
                stat_type="t",
            )
    except Exception as exc:  # pragma: no cover - defensive report path
        empty = np.array([], dtype=np.float64)
        engine = _engine_result(
            status="exception",
            main_effect=empty,
            main_stat=empty,
            slope_effect=empty,
            slope_stat=empty,
            warnings_seen=(),
            exception=exc,
        )
        return engine, empty, empty, empty, empty

    warnings_seen = tuple(str(w.message) for w in captured)
    main_effect = np.asarray(main.effect_size(), dtype=np.float64)
    main_stat = np.asarray(main.stat(), dtype=np.float64)
    slope_effect = np.asarray(slope.effect_size(), dtype=np.float64)
    slope_stat = np.asarray(slope.stat(), dtype=np.float64)
    engine = _engine_result(
        status="ok",
        main_effect=main_effect,
        main_stat=main_stat,
        slope_effect=slope_effect,
        slope_stat=slope_stat,
        warnings_seen=warnings_seen,
    )
    return engine, main_effect, main_stat, slope_effect, slope_stat


def _case_status(comparisons: dict[str, float | None]) -> tuple[str, str]:
    tight = (
        comparisons["main_effect_delta"] is not None
        and comparisons["main_effect_delta"] < 1e-7
        and comparisons["main_stat_delta"] is not None
        and comparisons["main_stat_delta"] < 1e-5
        and comparisons["slope_effect_delta"] is not None
        and comparisons["slope_effect_delta"] < 1e-7
        and comparisons["slope_stat_delta"] is not None
        and comparisons["slope_stat_delta"] < 1e-5
    )
    if tight:
        return "pass", "fmrimod and Nilearn agree on the realised design"
    return "fail", "cross-engine parity drift under a fixed centering scope"


def run_case(
    inputs: CenteringInputs,
    *,
    case_id: str,
    centering_scope: str,
    modulator_column: str,
) -> CaseOutput:
    """Run both engines for one RT-centering scope."""

    f_engine, f_main_eff, f_main_stat, f_slope_eff, f_slope_stat, design = (
        fmrimod_probe(inputs, modulator_column)
    )
    n_engine, n_main_eff, n_main_stat, n_slope_eff, n_slope_stat = (
        nilearn_probe(inputs, modulator_column, design)
    )
    comparisons = {
        "main_effect_delta": _max_abs_delta(f_main_eff, n_main_eff),
        "main_stat_delta": _max_abs_delta(f_main_stat, n_main_stat),
        "slope_effect_delta": _max_abs_delta(f_slope_eff, n_slope_eff),
        "slope_stat_delta": _max_abs_delta(f_slope_stat, n_slope_stat),
        "main_effect_median_abs_delta": _median_abs_delta(
            f_main_eff,
            n_main_eff,
        ),
        "slope_effect_median_abs_delta": _median_abs_delta(
            f_slope_eff,
            n_slope_eff,
        ),
    }
    status, verdict = _case_status(comparisons)
    report = CaseReport(
        case_id=case_id,
        purpose=(
            "Same trials and same raw RT values; only the parametric "
            "modulator centering scope changes."
        ),
        status=status,
        centering_scope=centering_scope,
        modulator_column=modulator_column,
        design_shape=tuple(int(x) for x in design.shape),
        design_rank=int(np.linalg.matrix_rank(design)),
        design_condition=float(np.linalg.cond(design)),
        fmrimod=f_engine,
        nilearn=n_engine,
        comparisons=comparisons,
        verdict=verdict,
    )
    return CaseOutput(
        report=report,
        fmrimod_main_effect=f_main_eff,
        fmrimod_main_stat=f_main_stat,
    )


def run_benchmark(
    max_voxels: int = MAX_VOXELS,
    seed: int = 20260513,
) -> dict[str, Any]:
    """Run the full centering-scope stress benchmark."""

    inputs = load_inputs(max_voxels=max_voxels, seed=seed)
    within = run_case(
        inputs,
        case_id="within_condition_centering",
        centering_scope="within_condition",
        modulator_column="rt_within_c",
    )
    global_case = run_case(
        inputs,
        case_id="global_centering_with_imbalanced_rt",
        centering_scope="global",
        modulator_column="rt_global_c",
    )
    main_shift = _median_abs_delta(
        global_case.fmrimod_main_effect,
        within.fmrimod_main_effect,
    )
    stat_shift = _median_abs_delta(
        global_case.fmrimod_main_stat,
        within.fmrimod_main_stat,
    )
    shift_is_visible = main_shift is not None and main_shift > 0.10
    cases = (within.report, global_case.report)
    status = (
        "pass"
        if all(case.status == "pass" for case in cases) and shift_is_visible
        else "fail"
    )
    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "name": "tier_e_parametric_centering",
            "status": status,
            "summary": (
                "Stress test for RT-modulator centering scope: fmrimod and "
                "Nilearn agree on each design, while the main-effect "
                "meaning moves when the centering reference changes."
            ),
            "rt_means": {
                "A": inputs.rt_mean_a,
                "B": inputs.rt_mean_b,
                "global": inputs.rt_mean_global,
                "A_minus_B": inputs.rt_mean_a - inputs.rt_mean_b,
            },
            "pain_points": {
                "main_effect_shift_median": main_shift,
                "main_stat_shift_median": stat_shift,
                "true_within_main_effect_median": inputs.true_within_main_median,
                "threshold": 0.10,
                "observed": bool(shift_is_visible),
                "verdict": (
                    "main effect changed under global centering despite "
                    "tight fmrimod/Nilearn parity within each design"
                ),
            },
            "cases": cases,
        }
    )


def render(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Write JSON and Markdown reports for the benchmark."""

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "centering_stress_report.json"
    md_path = out_dir / "REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Parametric Centering Stress",
        "",
        f"Status: `{report['status']}`",
        "",
        "## RT Means",
        "",
        "| condition | mean RT |",
        "| --- | ---: |",
        f"| A | {report['rt_means']['A']:.6g} |",
        f"| B | {report['rt_means']['B']:.6g} |",
        f"| global | {report['rt_means']['global']:.6g} |",
        "",
        "## Cases",
        "",
        "| case | status | scope | main effect delta | main stat delta | verdict |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for case in report["cases"]:
        lines.append(
            "| {case_id} | {status} | {scope} | {eff:.3g} | {stat:.3g} | {verdict} |".format(
                case_id=case["case_id"],
                status=case["status"],
                scope=case["centering_scope"],
                eff=case["comparisons"]["main_effect_delta"],
                stat=case["comparisons"]["main_stat_delta"],
                verdict=case["verdict"],
            )
        )
    pain = report["pain_points"]
    lines.extend(
        [
            "",
            "## Pain Point",
            "",
            (
                "Median main-effect shift under global vs within-condition "
                f"centering: `{pain['main_effect_shift_median']:.6g}`."
            ),
            pain["verdict"],
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
        help="Directory where the centering stress report should be written.",
    )
    parser.add_argument("--max-voxels", type=int, default=MAX_VOXELS)
    args = parser.parse_args(argv)

    report = run_benchmark(max_voxels=args.max_voxels)
    render(report, args.out_dir)
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
