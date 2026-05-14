"""Adversarial fmrimod-vs-Nilearn GLM failure-boundary gauntlet.

This workflow is intentionally not another happy-path parity receipt. It
contains two deterministic probes:

1. A survivable rank-deficient design where both engines should recover and
   agree on the identifiable t-contrast, while fmrimod exposes structured
   projection diagnostics.
2. A wide design with zero residual degrees of freedom, where the t-statistic
   is undefined. This records whether both engines expose undefined
   statistics as NaN instead of hiding them behind finite placeholders.

The point is to make "who breaks first, and how neatly?" executable.
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

from benchmarks.parity.adversarial_schema import validate_adversarial_report
from fmrimod.glm.contrasts import contrast_t
from fmrimod.glm.solver import fast_lm_matrix, fast_preproject

Array = NDArray[np.float64]
SCHEMA_VERSION = "adversarial-gauntlet/v1"
MAX_VOXELS = 128


@dataclass(frozen=True)
class ProbeInputs:
    """Shared matrix-level inputs for one adversarial probe."""

    case_id: str
    purpose: str
    design: Array
    data: Array
    columns: tuple[str, ...]
    contrast: Array
    expected_boundary: str


@dataclass(frozen=True)
class EngineProbe:
    """Summary of one engine's behavior on a probe."""

    status: str
    rank: int | None
    df_residual: float | None
    is_full_rank: bool | None
    ill_conditioned: bool | None
    aliased_columns: tuple[str, ...]
    finite_effect_fraction: float
    finite_stat_fraction: float
    nan_se_fraction: float
    nan_p_fraction: float
    warning_messages: tuple[str, ...]
    exception_type: str | None = None
    exception_message: str | None = None
    undefined_t_policy: str | None = None


@dataclass(frozen=True)
class CaseReport:
    """One adversarial case comparison."""

    case_id: str
    purpose: str
    status: str
    expected_boundary: str
    design_shape: tuple[int, int]
    design_rank: int
    design_condition: float
    fmrimod: EngineProbe
    nilearn: EngineProbe
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


def _center_scale(values: Array) -> Array:
    values = np.asarray(values, dtype=np.float64)
    values = values - float(np.mean(values))
    scale = float(np.linalg.norm(values))
    return values if scale == 0.0 else values / scale


def _task_boxcar(n_scans: int, start: int, period: int, width: int) -> Array:
    signal = np.zeros(n_scans, dtype=np.float64)
    for onset in range(start, n_scans, period):
        signal[onset : min(onset + width, n_scans)] = 1.0
    return _center_scale(signal)


def _make_survivable_inputs(max_voxels: int, seed: int) -> ProbeInputs:
    rng = np.random.default_rng(seed)
    n_scans = 96
    n_voxels = min(int(max_voxels), MAX_VOXELS)
    time = np.linspace(-1.0, 1.0, n_scans, dtype=np.float64)
    task_a = _task_boxcar(n_scans, start=8, period=24, width=5)
    task_b = _task_boxcar(n_scans, start=18, period=24, width=5)
    motion_x = _center_scale(np.sin(np.linspace(0.0, 5.0 * np.pi, n_scans)))
    motion_y = _center_scale(np.cos(np.linspace(0.0, 3.0 * np.pi, n_scans)))
    drift = _center_scale(time)
    drift2 = _center_scale(time * time)
    composite_motion = 1.7 * motion_x - 0.9 * motion_y + 0.25 * drift
    columns = (
        "intercept",
        "task_a",
        "task_b",
        "motion_x",
        "motion_y",
        "drift_linear",
        "drift_quadratic",
        "composite_motion",
    )
    design = np.column_stack(
        [
            np.ones(n_scans, dtype=np.float64),
            task_a,
            task_b,
            motion_x,
            motion_y,
            drift,
            drift2,
            composite_motion,
        ]
    )

    betas = rng.normal(scale=0.15, size=(len(columns), n_voxels))
    betas[0] = 100.0 + rng.normal(scale=0.2, size=n_voxels)
    betas[columns.index("task_a")] = np.linspace(0.5, 1.2, n_voxels)
    betas[columns.index("task_b")] = np.linspace(1.1, 0.3, n_voxels)
    betas[columns.index("composite_motion")] = 0.0
    data = design @ betas + rng.normal(scale=0.05, size=(n_scans, n_voxels))

    contrast = np.zeros(len(columns), dtype=np.float64)
    contrast[columns.index("task_a")] = 1.0
    contrast[columns.index("task_b")] = -1.0
    return ProbeInputs(
        case_id="survivable_rank_deficiency",
        purpose=(
            "Rank-deficient nuisance column; "
            "identifiable task contrast should survive."
        ),
        design=design.astype(np.float64),
        data=data.astype(np.float64),
        columns=columns,
        contrast=contrast,
        expected_boundary="recover_with_rank_diagnostics",
    )


def _make_wide_inputs(max_voxels: int, seed: int) -> ProbeInputs:
    rng = np.random.default_rng(seed + 1)
    n_scans = 16
    n_predictors = 24
    n_voxels = min(int(max_voxels), MAX_VOXELS)
    raw = rng.normal(size=(n_scans, n_predictors))
    raw[:, 0] = 1.0
    raw[:, 5] = raw[:, 1] + raw[:, 2]
    raw[:, 17] = raw[:, 3] - raw[:, 4]
    columns = tuple(f"x{i:02d}" for i in range(n_predictors))
    columns = ("intercept", *columns[1:])

    betas = rng.normal(scale=0.25, size=(n_predictors, n_voxels))
    data = raw @ betas
    contrast = np.zeros(n_predictors, dtype=np.float64)
    contrast[1] = 1.0
    return ProbeInputs(
        case_id="zero_residual_dof_wide_design",
        purpose=(
            "More columns than scans, full row rank, and no residual degrees "
            "of freedom; t-statistics are undefined."
        ),
        design=raw.astype(np.float64),
        data=data.astype(np.float64),
        columns=columns,
        contrast=contrast,
        expected_boundary="undefined_t_statistic",
    )


def _fraction(mask: Array) -> float:
    values = np.asarray(mask, dtype=bool)
    return float(np.mean(values)) if values.size else 1.0


def _undefined_policy(stat: Array, se: Array | None, p_value: Array | None) -> str:
    stat_arr = np.asarray(stat, dtype=np.float64)
    se_arr = np.asarray(se, dtype=np.float64) if se is not None else None
    p_arr = np.asarray(p_value, dtype=np.float64) if p_value is not None else None
    if se_arr is not None and np.all(np.isnan(se_arr)) and np.all(stat_arr == 0.0):
        return "zero_filled_t_with_nan_se"
    if np.all(np.isnan(stat_arr)):
        return "nan_t"
    if p_arr is not None and np.all(np.isnan(p_arr)):
        return "finite_t_with_nan_p"
    return "finite_or_mixed"


def fmrimod_probe(inputs: ProbeInputs) -> tuple[EngineProbe, Array, Array]:
    """Run fmrimod's low-level solver and return diagnostics plus arrays."""

    try:
        proj = fast_preproject(inputs.design)
        fit = fast_lm_matrix(inputs.design, inputs.data, proj)
        result = contrast_t(
            inputs.contrast,
            fit.betas,
            proj.XtXinv,
            np.sqrt(fit.sigma2),
            fit.dfres,
            name=inputs.case_id,
        )
    except Exception as exc:  # pragma: no cover - defensive report path
        probe = EngineProbe(
            status="exception",
            rank=None,
            df_residual=None,
            is_full_rank=None,
            ill_conditioned=None,
            aliased_columns=(),
            finite_effect_fraction=0.0,
            finite_stat_fraction=0.0,
            nan_se_fraction=0.0,
            nan_p_fraction=0.0,
            warning_messages=(),
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
        return probe, np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    aliased = tuple(inputs.columns[i] for i in proj.aliased_indices)
    probe = EngineProbe(
        status="ok",
        rank=int(proj.rank),
        df_residual=float(fit.dfres),
        is_full_rank=bool(proj.is_full_rank),
        ill_conditioned=bool(proj.ill_conditioned),
        aliased_columns=aliased,
        finite_effect_fraction=_fraction(np.isfinite(result.estimate)),
        finite_stat_fraction=_fraction(np.isfinite(result.stat)),
        nan_se_fraction=_fraction(np.isnan(result.se)),
        nan_p_fraction=_fraction(np.isnan(result.p_value)),
        warning_messages=(),
        undefined_t_policy=_undefined_policy(result.stat, result.se, result.p_value),
    )
    return probe, np.asarray(result.estimate, np.float64), np.asarray(result.stat, np.float64)


def nilearn_probe(inputs: ProbeInputs) -> tuple[EngineProbe, Array, Array]:
    """Run Nilearn's low-level GLM and return diagnostics plus arrays."""

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            labels, estimates = run_glm(inputs.data, inputs.design, noise_model="ols")
            result = compute_contrast(
                labels,
                estimates,
                inputs.contrast,
                stat_type="t",
            )
        except Exception as exc:  # pragma: no cover - defensive report path
            probe = EngineProbe(
                status="exception",
                rank=None,
                df_residual=None,
                is_full_rank=None,
                ill_conditioned=None,
                aliased_columns=(),
                finite_effect_fraction=0.0,
                finite_stat_fraction=0.0,
                nan_se_fraction=0.0,
                nan_p_fraction=0.0,
                warning_messages=tuple(str(w.message) for w in captured),
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            return probe, np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    regression = next(iter(estimates.values()))
    effect = np.asarray(result.effect_size(), dtype=np.float64)
    stat = np.asarray(result.stat(), dtype=np.float64)
    se = np.asarray(result.sd, dtype=np.float64) if hasattr(result, "sd") else None
    p_value = np.asarray(result.p_value(), dtype=np.float64)
    rank = getattr(regression, "df_model", None)
    df_residual = getattr(regression, "df_residuals", None)
    probe = EngineProbe(
        status="ok",
        rank=None if rank is None else int(rank),
        df_residual=None if df_residual is None else float(df_residual),
        is_full_rank=None,
        ill_conditioned=None,
        aliased_columns=(),
        finite_effect_fraction=_fraction(np.isfinite(effect)),
        finite_stat_fraction=_fraction(np.isfinite(stat)),
        nan_se_fraction=_fraction(np.isnan(se)) if se is not None else 0.0,
        nan_p_fraction=_fraction(np.isnan(p_value)),
        warning_messages=tuple(str(w.message) for w in captured),
        undefined_t_policy=_undefined_policy(stat, se, p_value),
    )
    return probe, effect, stat


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


def _median_ratio(candidate: Array, reference: Array) -> float | None:
    if candidate.shape != reference.shape or candidate.size == 0:
        return None
    mask = (
        np.isfinite(candidate)
        & np.isfinite(reference)
        & (np.abs(reference) > np.finfo(np.float64).eps)
    )
    if not np.any(mask):
        return None
    return float(np.median(candidate[mask] / reference[mask]))


def _rank_deficient_stat_scale_diagnostics(
    inputs: ProbeInputs,
    fmrimod: EngineProbe,
) -> dict[str, float | None]:
    """Explain the known rank-deficient t-stat scale delta.

    Nilearn's OLS result reports ``df_residuals = n - rank(X)``, but its
    dispersion estimate divides RSS by ``n - p``. fmrimod uses ``n - rank(X)``
    for both the reported residual df and residual-variance denominator.
    In a rank-deficient design this creates a global t-stat scale ratio while
    leaving effects and voxel rankings unchanged.
    """

    n_scans, n_columns = inputs.design.shape
    nilearn_dispersion_denom = float(n_scans - n_columns)
    fmrimod_dispersion_denom = fmrimod.df_residual
    proj = fast_preproject(inputs.design)
    pinv = np.linalg.pinv(inputs.design)
    nilearn_cov = pinv @ pinv.T
    fmrimod_cov_factor = float(inputs.contrast @ proj.XtXinv @ inputs.contrast)
    nilearn_cov_factor = float(inputs.contrast @ nilearn_cov @ inputs.contrast)
    expected_ratio: float | None
    if (
        fmrimod_dispersion_denom is not None
        and np.isfinite(fmrimod_dispersion_denom)
        and nilearn_dispersion_denom > 0.0
    ):
        expected_ratio = float(
            np.sqrt(fmrimod_dispersion_denom / nilearn_dispersion_denom)
        )
    else:
        expected_ratio = None
    return {
        "contrast_covariance_factor_delta": abs(
            fmrimod_cov_factor - nilearn_cov_factor
        ),
        "fmrimod_dispersion_denominator_rank_df": fmrimod_dispersion_denom,
        "nilearn_dispersion_denominator_column_df": nilearn_dispersion_denom,
        "expected_stat_scale_ratio_from_dof": expected_ratio,
    }


def _case_status(
    inputs: ProbeInputs,
    fmrimod: EngineProbe,
    nilearn: EngineProbe,
    comparisons: dict[str, float | None],
) -> tuple[str, str]:
    if inputs.case_id == "survivable_rank_deficiency":
        effect_ok = (
            comparisons["max_abs_effect_delta"] is not None
            and comparisons["max_abs_effect_delta"] < 1e-8
        )
        stat_ok = (
            comparisons["stat_pearson"] is not None
            and comparisons["stat_pearson"] > 0.999999
            and fmrimod.finite_stat_fraction == 1.0
            and nilearn.finite_stat_fraction == 1.0
        )
        diagnostic_ok = bool(fmrimod.aliased_columns) and fmrimod.ill_conditioned is True
        scale_explained = (
            comparisons["stat_scale_ratio_median"] is not None
            and comparisons["expected_stat_scale_ratio_from_dof"] is not None
            and comparisons["contrast_covariance_factor_delta"] is not None
            and comparisons["contrast_covariance_factor_delta"] < 1e-10
            and np.isclose(
                comparisons["stat_scale_ratio_median"],
                comparisons["expected_stat_scale_ratio_from_dof"],
                # numerical_floor: this compares two derived scale ratios in a
                # deliberately ill-conditioned rank-deficient fixture.
                rtol=1e-10,
                atol=1e-12,
            )
        )
        status = (
            "pass"
            if effect_ok and stat_ok and diagnostic_ok and scale_explained
            else "fail"
        )
        verdict = (
            "effects agree; t-stat scale drift is explained by the "
            "residual-variance DoF convention, not covariance pseudoinverse "
            "choice; fmrimod is neater on diagnostics"
            if status == "pass"
            else "recoverable rank-deficiency contract regressed"
        )
        return status, verdict

    fmrimod_nan = fmrimod.undefined_t_policy == "nan_t"
    nilearn_nan = nilearn.undefined_t_policy == "nan_t"
    status = "pass" if fmrimod_nan and nilearn_nan else "changed"
    verdict = (
        "both engines expose undefined t-statistics as NaN; fmrimod also "
        "records projection DoF/rank diagnostics"
        if status == "pass"
        else "zero-DoF behavior changed; review the recorded policies"
    )
    return status, verdict


def run_case(inputs: ProbeInputs) -> CaseReport:
    """Run both engines on one adversarial input set."""

    fmrimod, fmrimod_effect, fmrimod_stat = fmrimod_probe(inputs)
    nilearn, nilearn_effect, nilearn_stat = nilearn_probe(inputs)
    comparisons = {
        "max_abs_effect_delta": _max_abs_delta(fmrimod_effect, nilearn_effect),
        "max_abs_stat_delta": _max_abs_delta(fmrimod_stat, nilearn_stat),
        "effect_pearson": _pearson(fmrimod_effect, nilearn_effect),
        "stat_pearson": _pearson(fmrimod_stat, nilearn_stat),
        "stat_scale_ratio_median": _median_ratio(fmrimod_stat, nilearn_stat),
    }
    if inputs.case_id == "survivable_rank_deficiency":
        comparisons.update(_rank_deficient_stat_scale_diagnostics(inputs, fmrimod))
    status, verdict = _case_status(inputs, fmrimod, nilearn, comparisons)
    return CaseReport(
        case_id=inputs.case_id,
        purpose=inputs.purpose,
        status=status,
        expected_boundary=inputs.expected_boundary,
        design_shape=tuple(int(v) for v in inputs.design.shape),
        design_rank=int(np.linalg.matrix_rank(inputs.design)),
        design_condition=float(np.linalg.cond(inputs.design)),
        fmrimod=fmrimod,
        nilearn=nilearn,
        comparisons=comparisons,
        verdict=verdict,
    )


def run_gauntlet(max_voxels: int = MAX_VOXELS, seed: int = 20260513) -> dict[str, Any]:
    """Run the full adversarial gauntlet and return a JSON-ready report."""

    inputs = (
        _make_survivable_inputs(max_voxels=max_voxels, seed=seed),
        _make_wide_inputs(max_voxels=max_voxels, seed=seed),
    )
    cases = tuple(run_case(item) for item in inputs)
    status = "pass" if all(case.status in {"pass", "boundary_observed"} for case in cases) else "fail"
    report = _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "name": "tier_e_adversarial_gauntlet",
            "status": status,
            "summary": (
                "Executable stress report for fmrimod and Nilearn low-level "
                "GLM boundaries: recovery, diagnostics, and undefined-stat behavior."
            ),
            "cases": cases,
        }
    )
    validate_adversarial_report(report)
    return report


def render(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Write JSON and Markdown reports for the gauntlet."""

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "adversarial_gauntlet_report.json"
    md_path = out_dir / "REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Adversarial Parity Gauntlet",
        "",
        f"Status: `{report['status']}`",
        "",
        "| case | status | boundary | fmrimod policy | nilearn policy | verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        lines.append(
            "| {case_id} | {status} | {boundary} | {fpolicy} | {npolicy} | {verdict} |".format(
                case_id=case["case_id"],
                status=case["status"],
                boundary=case["expected_boundary"],
                fpolicy=case["fmrimod"]["undefined_t_policy"],
                npolicy=case["nilearn"]["undefined_t_policy"],
                verdict=case["verdict"],
            )
        )
    lines.extend(["", "## Scale Diagnostics", ""])
    for case in report["cases"]:
        ratio = case["comparisons"].get("stat_scale_ratio_median")
        expected = case["comparisons"].get("expected_stat_scale_ratio_from_dof")
        if ratio is None or expected is None:
            continue
        lines.append(
            "- `{case_id}`: median fmrimod/nilearn t-stat ratio "
            "`{ratio:.12g}`; expected from DoF convention `{expected:.12g}` "
            "(fmrimod dispersion denominator n-rank, Nilearn dispersion "
            "denominator n-p; covariance-factor delta `{cov_delta:.3g}`).".format(
                case_id=case["case_id"],
                ratio=ratio,
                expected=expected,
                cov_delta=case["comparisons"].get(
                    "contrast_covariance_factor_delta",
                    float("nan"),
                ),
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
        help="Directory where the adversarial gauntlet report should be written.",
    )
    parser.add_argument("--max-voxels", type=int, default=MAX_VOXELS)
    args = parser.parse_args(argv)

    report = run_gauntlet(max_voxels=args.max_voxels)
    render(report, args.out_dir)
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
