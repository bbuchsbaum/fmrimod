"""Semantic contrast-alignment stress benchmark.

This Tier E workflow targets a common first-level analysis failure mode:
reusing a positional contrast vector after the design matrix column order has
changed. The statistical model is intentionally ordinary. The stress is in
the semantic boundary between "the hypothesis is gain minus loss" and "the
second column minus the third column".

Two mathematically identical designs are fit: one in a canonical column order
and one with the same columns permuted. fmrimod evaluates the contrast through
``column_contrast("^gain$", pattern_B="^loss$")``, so the hypothesis resolves
against column names in each fit. Nilearn is checked in two modes:

- an aligned reference vector built from the column names, which should match
  fmrimod tightly;
- a deliberately reused positional vector from the canonical design, which
  should be visibly wrong after the permutation.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import run_glm
from numpy.typing import NDArray

import fmrimod as fm
from fmrimod.contrast import column_contrast
from fmrimod.model.config import FmriLmConfig

Array = NDArray[np.float64]
SCHEMA_VERSION = "semantic-contrast-alignment/v1"
MAX_VOXELS = 96
N_SCANS = 144
BASE_ORDER: tuple[str, ...] = ("intercept", "gain", "loss", "motion", "drift")
PERMUTED_ORDER: tuple[str, ...] = (
    "motion",
    "loss",
    "intercept",
    "drift",
    "gain",
)
CONTRAST_NAME = "gain_minus_loss"


@dataclass(frozen=True)
class AlignmentInputs:
    """Shared fixture for the contrast-alignment stress case."""

    base_design: Array
    data: Array
    base_order: tuple[str, ...]
    permuted_order: tuple[str, ...]
    true_effect_median: float


@dataclass(frozen=True)
class EngineResult:
    """One engine's contrast result under one column order."""

    status: str
    effect_median: float | None
    stat_median: float | None
    finite_effect_fraction: float
    finite_stat_fraction: float
    touched_columns: tuple[str, ...]
    contrast_vector: tuple[float, ...]
    warning_messages: tuple[str, ...]
    exception_type: str | None = None
    exception_message: str | None = None


@dataclass(frozen=True)
class CaseReport:
    """Report for one design-column ordering."""

    case_id: str
    status: str
    column_order: tuple[str, ...]
    design_shape: tuple[int, int]
    design_rank: int
    design_condition: float
    fmrimod: EngineResult
    nilearn_aligned: EngineResult
    nilearn_positional_reuse: EngineResult
    comparisons: dict[str, float | None]
    verdict: str


@dataclass(frozen=True)
class CaseOutput:
    """Private holder for report plus arrays used across cases."""

    report: CaseReport
    fmrimod_effect: Array
    fmrimod_stat: Array
    nilearn_aligned_effect: Array
    nilearn_aligned_stat: Array
    nilearn_positional_effect: Array
    nilearn_positional_stat: Array


class _MatrixDataset:
    """Minimal dataset object for the legacy matrix-model path."""

    def __init__(self, data: Array):
        self._data = np.asarray(data, dtype=np.float64)

    def get_data(self, run: int) -> Array:
        if run != 0:
            raise IndexError("only one run is available")
        return self._data

    def get_censor(self, run: int) -> None:
        if run != 0:
            raise IndexError("only one run is available")
        return None


class _NamedMatrixModel:
    """Minimal named-design model accepted by ``fmri_lm``."""

    def __init__(self, design: Array, data: Array, column_order: tuple[str, ...]):
        self._design = np.asarray(design, dtype=np.float64)
        self.dataset = _MatrixDataset(data)
        self.n_runs = 1
        self._column_order = tuple(column_order)

    def design_matrix_array(self, run: int) -> Array:
        if run != 0:
            raise IndexError("only one run is available")
        return self._design

    def design_columns(self) -> tuple[str, ...]:
        return self._column_order


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


def _make_base_design() -> dict[str, Array]:
    time = np.linspace(-1.0, 1.0, N_SCANS, dtype=np.float64)
    return {
        "intercept": np.ones(N_SCANS, dtype=np.float64),
        "gain": _boxcar(N_SCANS, (9, 39, 73, 111), width=6),
        "loss": _boxcar(N_SCANS, (22, 55, 91, 128), width=5),
        "motion": _center_scale(np.sin(np.linspace(0.0, 7.0 * np.pi, N_SCANS))),
        "drift": _center_scale(time + 0.25 * time * time),
    }


def _design_from_order(columns: dict[str, Array], order: tuple[str, ...]) -> Array:
    return np.column_stack([columns[name] for name in order]).astype(np.float64)


def _reorder_design(
    design: Array, from_order: tuple[str, ...], to_order: tuple[str, ...]
) -> Array:
    lookup = {name: idx for idx, name in enumerate(from_order)}
    return design[:, [lookup[name] for name in to_order]].astype(np.float64)


def _contrast_vector(order: tuple[str, ...]) -> Array:
    weights = np.zeros(len(order), dtype=np.float64)
    weights[order.index("gain")] = 1.0
    weights[order.index("loss")] = -1.0
    return weights


def _touched_columns(order: tuple[str, ...], weights: Array) -> tuple[str, ...]:
    return tuple(name for name, weight in zip(order, weights) if abs(weight) > 0.0)


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
    touched_columns: tuple[str, ...],
    contrast_vector: Array,
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
        contrast_vector=tuple(float(x) for x in contrast_vector),
        warning_messages=warnings_seen,
        exception_type=None if exception is None else type(exception).__name__,
        exception_message=None if exception is None else str(exception),
    )


def load_inputs(
    max_voxels: int = MAX_VOXELS,
    seed: int = 20260513,
) -> AlignmentInputs:
    """Create a named-design fixture with a known gain-minus-loss effect."""

    rng = np.random.default_rng(seed)
    columns = _make_base_design()
    base_design = _design_from_order(columns, BASE_ORDER)
    n_voxels = min(int(max_voxels), MAX_VOXELS)
    ramp = np.linspace(0.75, 1.25, n_voxels, dtype=np.float64)
    betas = np.zeros((len(BASE_ORDER), n_voxels), dtype=np.float64)
    betas[BASE_ORDER.index("intercept")] = 100.0 + rng.normal(
        scale=0.15,
        size=n_voxels,
    )
    betas[BASE_ORDER.index("gain")] = 1.10 * ramp
    betas[BASE_ORDER.index("loss")] = -0.45 * ramp
    betas[BASE_ORDER.index("motion")] = rng.normal(scale=0.08, size=n_voxels)
    betas[BASE_ORDER.index("drift")] = rng.normal(scale=0.10, size=n_voxels)
    data = base_design @ betas + rng.normal(
        scale=0.02,
        size=(N_SCANS, n_voxels),
    )
    true_effect = betas[BASE_ORDER.index("gain")] - betas[BASE_ORDER.index("loss")]
    return AlignmentInputs(
        base_design=base_design,
        data=data.astype(np.float64),
        base_order=BASE_ORDER,
        permuted_order=PERMUTED_ORDER,
        true_effect_median=float(np.median(true_effect)),
    )


def fmrimod_probe(
    design: Array,
    data: Array,
    column_order: tuple[str, ...],
) -> tuple[EngineResult, Array, Array]:
    """Fit fmrimod and resolve the contrast by column name."""

    weights = _contrast_vector(column_order)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = _NamedMatrixModel(design, data, column_order)
            fit = fm.fmri_lm(model, FmriLmConfig())
            result = fit.contrast(
                column_contrast(
                    "^gain$",
                    pattern_B="^loss$",
                    name=CONTRAST_NAME,
                )
            )
    except Exception as exc:  # pragma: no cover - defensive report path
        empty = np.array([], dtype=np.float64)
        engine = _engine_result(
            status="exception",
            effect=empty,
            stat=empty,
            touched_columns=(),
            contrast_vector=weights,
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
        contrast_vector=weights,
        warnings_seen=tuple(str(w.message) for w in captured),
    )
    return engine, effect, stat


def nilearn_probe(
    design: Array,
    data: Array,
    column_order: tuple[str, ...],
    weights: Array,
) -> tuple[EngineResult, Array, Array]:
    """Fit Nilearn with an explicit numeric contrast vector."""

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            labels, estimates = run_glm(data, design, noise_model="ols")
            result = compute_contrast(labels, estimates, weights, stat_type="t")
    except Exception as exc:  # pragma: no cover - defensive report path
        empty = np.array([], dtype=np.float64)
        engine = _engine_result(
            status="exception",
            effect=empty,
            stat=empty,
            touched_columns=_touched_columns(column_order, weights),
            contrast_vector=weights,
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
        touched_columns=_touched_columns(column_order, weights),
        contrast_vector=weights,
        warnings_seen=tuple(str(w.message) for w in captured),
    )
    return engine, effect, stat


def _case_status(comparisons: dict[str, float | None]) -> tuple[str, str]:
    aligned_ok = (
        comparisons["aligned_effect_delta"] is not None
        and comparisons["aligned_effect_delta"] < 1e-8
        and comparisons["aligned_stat_delta"] is not None
        and comparisons["aligned_stat_delta"] < 1e-5
    )
    if aligned_ok:
        return "pass", "fmrimod typed contrast matches the aligned Nilearn vector"
    return "fail", "semantic contrast alignment drifted against the Nilearn oracle"


def run_case(
    inputs: AlignmentInputs,
    *,
    case_id: str,
    column_order: tuple[str, ...],
) -> CaseOutput:
    """Run both engines for one column ordering."""

    design = _reorder_design(inputs.base_design, inputs.base_order, column_order)
    aligned_weights = _contrast_vector(column_order)
    positional_weights = _contrast_vector(inputs.base_order)
    f_engine, f_effect, f_stat = fmrimod_probe(design, inputs.data, column_order)
    n_engine, n_effect, n_stat = nilearn_probe(
        design,
        inputs.data,
        column_order,
        aligned_weights,
    )
    p_engine, p_effect, p_stat = nilearn_probe(
        design,
        inputs.data,
        column_order,
        positional_weights,
    )
    comparisons = {
        "aligned_effect_delta": _max_abs_delta(f_effect, n_effect),
        "aligned_stat_delta": _max_abs_delta(f_stat, n_stat),
        "positional_effect_median_abs_delta": _median_abs_delta(n_effect, p_effect),
        "positional_stat_median_abs_delta": _median_abs_delta(n_stat, p_stat),
    }
    status, verdict = _case_status(comparisons)
    report = CaseReport(
        case_id=case_id,
        status=status,
        column_order=column_order,
        design_shape=tuple(int(x) for x in design.shape),
        design_rank=int(np.linalg.matrix_rank(design)),
        design_condition=float(np.linalg.cond(design)),
        fmrimod=f_engine,
        nilearn_aligned=n_engine,
        nilearn_positional_reuse=p_engine,
        comparisons=comparisons,
        verdict=verdict,
    )
    return CaseOutput(
        report=report,
        fmrimod_effect=f_effect,
        fmrimod_stat=f_stat,
        nilearn_aligned_effect=n_effect,
        nilearn_aligned_stat=n_stat,
        nilearn_positional_effect=p_effect,
        nilearn_positional_stat=p_stat,
    )


def run_benchmark(
    max_voxels: int = MAX_VOXELS,
    seed: int = 20260513,
) -> dict[str, Any]:
    """Run the full semantic contrast-alignment stress benchmark."""

    inputs = load_inputs(max_voxels=max_voxels, seed=seed)
    base = run_case(inputs, case_id="canonical_order", column_order=inputs.base_order)
    permuted = run_case(
        inputs,
        case_id="permuted_order",
        column_order=inputs.permuted_order,
    )
    invariance = {
        "fmrimod_effect_delta": _max_abs_delta(
            base.fmrimod_effect,
            permuted.fmrimod_effect,
        ),
        "fmrimod_stat_delta": _max_abs_delta(base.fmrimod_stat, permuted.fmrimod_stat),
        "nilearn_aligned_effect_delta": _max_abs_delta(
            base.nilearn_aligned_effect,
            permuted.nilearn_aligned_effect,
        ),
        "nilearn_aligned_stat_delta": _max_abs_delta(
            base.nilearn_aligned_stat,
            permuted.nilearn_aligned_stat,
        ),
    }
    positional_effect_shift = permuted.report.comparisons[
        "positional_effect_median_abs_delta"
    ]
    positional_stat_shift = permuted.report.comparisons[
        "positional_stat_median_abs_delta"
    ]
    pain_observed = (
        positional_effect_shift is not None
        and positional_effect_shift > 10.0
        and positional_stat_shift is not None
        and positional_stat_shift > 10.0
    )
    invariant = (
        invariance["fmrimod_effect_delta"] is not None
        and invariance["fmrimod_effect_delta"] < 1e-8
        and invariance["fmrimod_stat_delta"] is not None
        and invariance["fmrimod_stat_delta"] < 1e-5
        and invariance["nilearn_aligned_effect_delta"] is not None
        and invariance["nilearn_aligned_effect_delta"] < 1e-8
        and invariance["nilearn_aligned_stat_delta"] is not None
        and invariance["nilearn_aligned_stat_delta"] < 1e-5
    )
    cases = (base.report, permuted.report)
    status = (
        "pass"
        if all(case.status == "pass" for case in cases) and invariant and pain_observed
        else "fail"
    )
    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "name": "tier_e_semantic_contrast_alignment",
            "status": status,
            "summary": (
                "Stress test for semantic contrast alignment under design-column "
                "permutation: fmrimod resolves gain-minus-loss by name, while a "
                "reused positional Nilearn vector visibly targets the wrong columns."
            ),
            "contrast": {
                "name": CONTRAST_NAME,
                "positive": "gain",
                "negative": "loss",
            },
            "design_column_orders": {
                "canonical": inputs.base_order,
                "permuted": inputs.permuted_order,
            },
            "true_effect_median": inputs.true_effect_median,
            "invariance": invariance,
            "pain_points": {
                "nilearn_positional_effect_median_abs_delta": positional_effect_shift,
                "nilearn_positional_stat_median_abs_delta": positional_stat_shift,
                "threshold": 10.0,
                "observed": bool(pain_observed),
                "verdict": (
                    "the positional vector from the canonical design silently "
                    "targets different columns after permutation"
                ),
            },
            "cases": cases,
        }
    )


def render(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Write JSON and Markdown reports for the benchmark."""

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "semantic_contrast_alignment_report.json"
    md_path = out_dir / "REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Semantic Contrast Alignment",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Column Orders",
        "",
        f"- Canonical: `{', '.join(report['design_column_orders']['canonical'])}`",
        f"- Permuted: `{', '.join(report['design_column_orders']['permuted'])}`",
        "",
        "## Cases",
        "",
        "| case | status | fmrimod touched | aligned delta | positional effect drift | verdict |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for case in report["cases"]:
        lines.append(
            "| {case_id} | {status} | {touched} | {aligned:.3g} | {drift:.3g} | {verdict} |".format(
                case_id=case["case_id"],
                status=case["status"],
                touched=", ".join(case["fmrimod"]["touched_columns"]),
                aligned=case["comparisons"]["aligned_effect_delta"],
                drift=case["comparisons"]["positional_effect_median_abs_delta"],
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
                "Median effect drift from reusing the canonical positional "
                "vector after permutation: "
                f"`{pain['nilearn_positional_effect_median_abs_delta']:.6g}`."
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
        help="Directory where the semantic contrast-alignment report is written.",
    )
    parser.add_argument("--max-voxels", type=int, default=MAX_VOXELS)
    args = parser.parse_args(argv)

    report = run_benchmark(max_voxels=args.max_voxels)
    render(report, args.out_dir)
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
