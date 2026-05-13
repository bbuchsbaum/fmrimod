"""Design-reparameterization invariance stress benchmark.

This Tier E canary asks whether fmrimod and Nilearn preserve the same
modeled hypothesis after an invertible but badly scaled column
transformation. It is intentionally matrix-level: the public-seam version
belongs later, once typed stress-design construction exists.
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

from fmrimod.glm.contrasts import contrast_t
from fmrimod.glm.solver import fast_lm_matrix, fast_preproject

Array = NDArray[np.float64]
SCHEMA_VERSION = "reparameterization-invariance/v1"
MAX_VOXELS = 96


@dataclass(frozen=True)
class ProbeInputs:
    """Inputs for one design-coordinate stress case."""

    case_id: str
    purpose: str
    design: Array
    transformed_design: Array
    data: Array
    contrast: Array
    transformed_contrast: Array
    scale_span: float
    expected_boundary: str


@dataclass(frozen=True)
class EngineResult:
    """One engine's base/transformed-design comparison."""

    status: str
    base_rank: int | None
    transformed_rank: int | None
    base_df_residual: float | None
    transformed_df_residual: float | None
    base_condition: float
    transformed_condition: float
    finite_effect_fraction: float
    finite_stat_fraction: float
    warning_messages: tuple[str, ...]
    exception_type: str | None = None
    exception_message: str | None = None


@dataclass(frozen=True)
class CaseReport:
    """Structured report for one reparameterization stress case."""

    case_id: str
    purpose: str
    status: str
    expected_boundary: str
    scale_span: float
    fmrimod: EngineResult
    nilearn: EngineResult
    comparisons: dict[str, float | None]
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


def _standardize(values: Array) -> Array:
    values = np.asarray(values, dtype=np.float64)
    values = values - float(np.mean(values))
    scale = float(np.std(values))
    return values if scale == 0.0 else values / scale


def _make_base_design(n_scans: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    time = np.linspace(-1.0, 1.0, n_scans, dtype=np.float64)
    columns = [
        np.ones(n_scans, dtype=np.float64),
        _standardize(np.sin(2.0 * np.pi * time)),
        _standardize(np.cos(3.0 * np.pi * time)),
        _standardize(time),
        _standardize(time * time),
        _standardize(rng.normal(size=n_scans)),
        _standardize(rng.normal(size=n_scans)),
    ]
    return np.column_stack(columns).astype(np.float64)


def _make_inputs(
    case_id: str,
    *,
    scale_span: float,
    expected_boundary: str,
    max_voxels: int,
    seed: int,
) -> ProbeInputs:
    rng = np.random.default_rng(seed + 101)
    design = _make_base_design(n_scans=120, seed=seed)
    n_scans, n_columns = design.shape
    n_voxels = min(int(max_voxels), MAX_VOXELS)

    betas = rng.normal(scale=0.2, size=(n_columns, n_voxels))
    betas[0] = 100.0 + rng.normal(scale=0.2, size=n_voxels)
    betas[1] = np.linspace(0.2, 1.0, n_voxels)
    betas[2] = np.linspace(0.8, 0.1, n_voxels)
    data = design @ betas + rng.normal(scale=0.1, size=(n_scans, n_voxels))

    contrast = np.zeros(n_columns, dtype=np.float64)
    contrast[1] = 1.0
    contrast[2] = -0.5

    scale = np.geomspace(1.0 / scale_span, scale_span, n_columns)
    transform = np.diag(scale)
    transformed_design = design @ transform
    transformed_contrast = transform.T @ contrast

    purpose = (
        "Invertible design-coordinate transform should preserve the authored "
        "task contrast until numerical rank loss becomes the boundary."
    )
    return ProbeInputs(
        case_id=case_id,
        purpose=purpose,
        design=design,
        transformed_design=transformed_design.astype(np.float64),
        data=data.astype(np.float64),
        contrast=contrast,
        transformed_contrast=transformed_contrast.astype(np.float64),
        scale_span=float(scale_span),
        expected_boundary=expected_boundary,
    )


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


def _pearson(candidate: Array, reference: Array) -> float | None:
    if candidate.shape != reference.shape or candidate.size < 2:
        return None
    mask = np.isfinite(candidate) & np.isfinite(reference)
    if int(np.sum(mask)) < 2:
        return None
    cand = candidate[mask]
    ref = reference[mask]
    if float(np.std(cand)) == 0.0 or float(np.std(ref)) == 0.0:
        return 1.0 if np.allclose(cand, ref) else 0.0
    return float(np.corrcoef(cand, ref)[0, 1])


def _fmrimod_once(design: Array, data: Array, contrast: Array) -> tuple[EngineResult, Array, Array]:
    proj = fast_preproject(design)
    fit = fast_lm_matrix(design, data, proj)
    result = contrast_t(
        contrast,
        fit.betas,
        proj.XtXinv,
        np.sqrt(fit.sigma2),
        fit.dfres,
    )
    engine = EngineResult(
        status="ok",
        base_rank=None,
        transformed_rank=int(proj.rank),
        base_df_residual=None,
        transformed_df_residual=float(fit.dfres),
        base_condition=float(np.linalg.cond(design)),
        transformed_condition=float(np.linalg.cond(design)),
        finite_effect_fraction=_fraction(np.isfinite(result.estimate)),
        finite_stat_fraction=_fraction(np.isfinite(result.stat)),
        warning_messages=(),
    )
    return engine, np.asarray(result.estimate, np.float64), np.asarray(result.stat, np.float64)


def fmrimod_probe(inputs: ProbeInputs) -> tuple[EngineResult, Array, Array, Array, Array]:
    """Run fmrimod on base and transformed designs."""

    try:
        base_engine, base_effect, base_stat = _fmrimod_once(
            inputs.design,
            inputs.data,
            inputs.contrast,
        )
        scaled_engine, scaled_effect, scaled_stat = _fmrimod_once(
            inputs.transformed_design,
            inputs.data,
            inputs.transformed_contrast,
        )
    except Exception as exc:  # pragma: no cover - defensive report path
        engine = EngineResult(
            status="exception",
            base_rank=None,
            transformed_rank=None,
            base_df_residual=None,
            transformed_df_residual=None,
            base_condition=float(np.linalg.cond(inputs.design)),
            transformed_condition=float(np.linalg.cond(inputs.transformed_design)),
            finite_effect_fraction=0.0,
            finite_stat_fraction=0.0,
            warning_messages=(),
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
        empty = np.array([], dtype=np.float64)
        return engine, empty, empty, empty, empty

    engine = EngineResult(
        status="ok",
        base_rank=base_engine.transformed_rank,
        transformed_rank=scaled_engine.transformed_rank,
        base_df_residual=base_engine.transformed_df_residual,
        transformed_df_residual=scaled_engine.transformed_df_residual,
        base_condition=base_engine.base_condition,
        transformed_condition=scaled_engine.transformed_condition,
        finite_effect_fraction=scaled_engine.finite_effect_fraction,
        finite_stat_fraction=scaled_engine.finite_stat_fraction,
        warning_messages=(),
    )
    return engine, base_effect, base_stat, scaled_effect, scaled_stat


def _nilearn_once(
    design: Array,
    data: Array,
    contrast: Array,
) -> tuple[EngineResult, Array, Array, tuple[str, ...]]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        labels, estimates = run_glm(data, design, noise_model="ols")
        result = compute_contrast(labels, estimates, contrast, stat_type="t")
    regression = next(iter(estimates.values()))
    rank = getattr(regression, "df_model", None)
    df_residual = getattr(regression, "df_residuals", None)
    effect = np.asarray(result.effect_size(), dtype=np.float64)
    stat = np.asarray(result.stat(), dtype=np.float64)
    messages = tuple(str(w.message) for w in captured)
    engine = EngineResult(
        status="ok",
        base_rank=None,
        transformed_rank=None if rank is None else int(rank),
        base_df_residual=None,
        transformed_df_residual=None if df_residual is None else float(df_residual),
        base_condition=float(np.linalg.cond(design)),
        transformed_condition=float(np.linalg.cond(design)),
        finite_effect_fraction=_fraction(np.isfinite(effect)),
        finite_stat_fraction=_fraction(np.isfinite(stat)),
        warning_messages=messages,
    )
    return engine, effect, stat, messages


def nilearn_probe(inputs: ProbeInputs) -> tuple[EngineResult, Array, Array, Array, Array]:
    """Run Nilearn on base and transformed designs."""

    try:
        base_engine, base_effect, base_stat, base_warnings = _nilearn_once(
            inputs.design,
            inputs.data,
            inputs.contrast,
        )
        scaled_engine, scaled_effect, scaled_stat, scaled_warnings = _nilearn_once(
            inputs.transformed_design,
            inputs.data,
            inputs.transformed_contrast,
        )
    except Exception as exc:  # pragma: no cover - defensive report path
        engine = EngineResult(
            status="exception",
            base_rank=None,
            transformed_rank=None,
            base_df_residual=None,
            transformed_df_residual=None,
            base_condition=float(np.linalg.cond(inputs.design)),
            transformed_condition=float(np.linalg.cond(inputs.transformed_design)),
            finite_effect_fraction=0.0,
            finite_stat_fraction=0.0,
            warning_messages=(),
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
        empty = np.array([], dtype=np.float64)
        return engine, empty, empty, empty, empty

    engine = EngineResult(
        status="ok",
        base_rank=base_engine.transformed_rank,
        transformed_rank=scaled_engine.transformed_rank,
        base_df_residual=base_engine.transformed_df_residual,
        transformed_df_residual=scaled_engine.transformed_df_residual,
        base_condition=base_engine.base_condition,
        transformed_condition=scaled_engine.transformed_condition,
        finite_effect_fraction=scaled_engine.finite_effect_fraction,
        finite_stat_fraction=scaled_engine.finite_stat_fraction,
        warning_messages=base_warnings + scaled_warnings,
    )
    return engine, base_effect, base_stat, scaled_effect, scaled_stat


def _case_status(inputs: ProbeInputs, fmrimod: EngineResult, nilearn: EngineResult, comparisons: dict[str, float | None]) -> tuple[str, str]:
    if inputs.case_id == "moderate_scale_reparameterization":
        strict = (
            fmrimod.base_rank == fmrimod.transformed_rank
            and nilearn.base_rank == nilearn.transformed_rank
            and comparisons["fmrimod_effect_delta"] is not None
            and comparisons["fmrimod_effect_delta"] < 1e-8
            and comparisons["fmrimod_stat_delta"] is not None
            and comparisons["fmrimod_stat_delta"] < 1e-5
            and comparisons["nilearn_effect_delta"] is not None
            and comparisons["nilearn_effect_delta"] < 1e-8
            and comparisons["nilearn_stat_delta"] is not None
            and comparisons["nilearn_stat_delta"] < 1e-5
        )
        return (
            ("pass", "both engines preserve the hypothesis under survivable scaling")
            if strict
            else ("fail", "moderate reparameterization invariance regressed")
        )

    rank_loss = (
        fmrimod.transformed_rank is not None
        and fmrimod.base_rank is not None
        and fmrimod.transformed_rank < fmrimod.base_rank
        and nilearn.transformed_rank is not None
        and nilearn.base_rank is not None
        and nilearn.transformed_rank < nilearn.base_rank
    )
    drift = (
        comparisons["fmrimod_effect_delta"] is not None
        and comparisons["fmrimod_effect_delta"] > 1e-3
        and comparisons["nilearn_effect_delta"] is not None
        and comparisons["nilearn_effect_delta"] > 1e-3
    )
    if rank_loss and drift:
        return (
            "boundary_observed",
            "extreme scaling loses numerical rank; both engines preserve the same degraded hypothesis",
        )
    return "fail", "extreme reparameterization boundary was not recorded"


def run_case(inputs: ProbeInputs) -> CaseReport:
    """Run both engines on one reparameterization stress input."""

    f_engine, f_base_eff, f_base_stat, f_scaled_eff, f_scaled_stat = fmrimod_probe(inputs)
    n_engine, n_base_eff, n_base_stat, n_scaled_eff, n_scaled_stat = nilearn_probe(inputs)
    comparisons = {
        "fmrimod_effect_delta": _max_abs_delta(f_scaled_eff, f_base_eff),
        "fmrimod_stat_delta": _max_abs_delta(f_scaled_stat, f_base_stat),
        "fmrimod_effect_pearson": _pearson(f_scaled_eff, f_base_eff),
        "fmrimod_stat_pearson": _pearson(f_scaled_stat, f_base_stat),
        "nilearn_effect_delta": _max_abs_delta(n_scaled_eff, n_base_eff),
        "nilearn_stat_delta": _max_abs_delta(n_scaled_stat, n_base_stat),
        "nilearn_effect_pearson": _pearson(n_scaled_eff, n_base_eff),
        "nilearn_stat_pearson": _pearson(n_scaled_stat, n_base_stat),
        "scaled_cross_engine_effect_delta": _max_abs_delta(f_scaled_eff, n_scaled_eff),
        "scaled_cross_engine_stat_delta": _max_abs_delta(f_scaled_stat, n_scaled_stat),
        "scaled_cross_engine_effect_pearson": _pearson(f_scaled_eff, n_scaled_eff),
        "scaled_cross_engine_stat_pearson": _pearson(f_scaled_stat, n_scaled_stat),
    }
    status, verdict = _case_status(inputs, f_engine, n_engine, comparisons)
    return CaseReport(
        case_id=inputs.case_id,
        purpose=inputs.purpose,
        status=status,
        expected_boundary=inputs.expected_boundary,
        scale_span=inputs.scale_span,
        fmrimod=f_engine,
        nilearn=n_engine,
        comparisons=comparisons,
        verdict=verdict,
    )


def run_benchmark(max_voxels: int = MAX_VOXELS, seed: int = 20260513) -> dict[str, Any]:
    """Run the full reparameterization-invariance canary."""

    inputs = (
        _make_inputs(
            "moderate_scale_reparameterization",
            scale_span=1.0e4,
            expected_boundary="strict_invariance",
            max_voxels=max_voxels,
            seed=seed,
        ),
        _make_inputs(
            "extreme_scale_rank_boundary",
            scale_span=1.0e12,
            expected_boundary="rank_loss_under_extreme_scaling",
            max_voxels=max_voxels,
            seed=seed,
        ),
    )
    cases = tuple(run_case(item) for item in inputs)
    status = "pass" if all(case.status in {"pass", "boundary_observed"} for case in cases) else "fail"
    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "name": "tier_e_reparameterization_invariance",
            "status": status,
            "summary": (
                "Matrix-level canary for hypothesis invariance under invertible "
                "but badly scaled design-coordinate transformations."
            ),
            "cases": cases,
        }
    )


def render(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Write JSON and Markdown reports for the benchmark."""

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "reparameterization_invariance_report.json"
    md_path = out_dir / "REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Reparameterization Invariance Canary",
        "",
        f"Status: `{report['status']}`",
        "",
        "| case | status | scale span | fmrimod rank | nilearn rank | verdict |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for case in report["cases"]:
        f_rank = f"{case['fmrimod']['base_rank']}->{case['fmrimod']['transformed_rank']}"
        n_rank = f"{case['nilearn']['base_rank']}->{case['nilearn']['transformed_rank']}"
        lines.append(
            "| {case_id} | {status} | {span:.1e} | {f_rank} | {n_rank} | {verdict} |".format(
                case_id=case["case_id"],
                status=case["status"],
                span=case["scale_span"],
                f_rank=f_rank,
                n_rank=n_rank,
                verdict=case["verdict"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="Directory where the reparameterization report should be written.",
    )
    parser.add_argument("--max-voxels", type=int, default=MAX_VOXELS)
    args = parser.parse_args(argv)

    report = run_benchmark(max_voxels=args.max_voxels)
    render(report, args.out_dir)
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
