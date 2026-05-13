"""Synthetic first-level F-contrast parity against Nilearn.

This case targets an overlap slice not covered by the existing Tier A
workflows: a joint F-test over task regressors while nuisance/confound and
polynomial drift columns are present in the same first-level design.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import run_glm
from numpy.typing import NDArray

from cross_testing.harness import (
    ParityCase,
    ParityTolerance,
    PipelineOutput,
    render,
    run,
)
from fmrimod.glm.contrasts import contrast_f, contrast_t
from fmrimod.glm.solver import fast_lm_matrix, fast_preproject

Array = NDArray[np.float64]
MAX_VOXELS = 2048


@dataclass(frozen=True)
class SyntheticFContrastInputs:
    """Inputs shared by the fmrimod and Nilearn synthetic GLM paths."""

    design: Array
    data: Array
    columns: tuple[str, ...]
    t_contrast: Array
    f_contrast: Array


def _center_scale(values: Array) -> Array:
    values = np.asarray(values, dtype=np.float64)
    values = values - values.mean()
    scale = np.linalg.norm(values)
    if scale == 0:
        return values
    return values / scale


def _task_boxcar(n_scans: int, onsets: range, width: int) -> Array:
    signal = np.zeros(n_scans, dtype=np.float64)
    for onset in onsets:
        signal[onset : min(onset + width, n_scans)] = 1.0
    return _center_scale(signal)


def _make_design(n_scans: int) -> tuple[Array, tuple[str, ...]]:
    time = np.linspace(-1.0, 1.0, n_scans, dtype=np.float64)
    condition_a = _task_boxcar(n_scans, range(8, n_scans, 24), width=6)
    condition_b = _task_boxcar(n_scans, range(18, n_scans, 24), width=5)
    motion_x = _center_scale(np.sin(np.linspace(0.0, 4.0 * np.pi, n_scans)))
    motion_y = _center_scale(np.cos(np.linspace(0.0, 3.0 * np.pi, n_scans)))
    drift_linear = _center_scale(time)
    drift_quadratic = _center_scale(time * time)

    columns = (
        "intercept",
        "condition_a",
        "condition_b",
        "motion_x",
        "motion_y",
        "drift_linear",
        "drift_quadratic",
    )
    design = np.column_stack(
        [
            np.ones(n_scans, dtype=np.float64),
            condition_a,
            condition_b,
            motion_x,
            motion_y,
            drift_linear,
            drift_quadratic,
        ]
    ).astype(np.float64)
    return design, columns


def load_inputs(max_voxels: int = MAX_VOXELS) -> SyntheticFContrastInputs:
    """Build deterministic synthetic first-level inputs."""

    n_scans = 120
    n_voxels = min(int(max_voxels), MAX_VOXELS)
    design, columns = _make_design(n_scans)
    rng = np.random.default_rng(20260513)

    betas = np.zeros((design.shape[1], n_voxels), dtype=np.float64)
    betas[0, :] = 100.0 + rng.normal(scale=0.5, size=n_voxels)
    betas[1, :] = np.linspace(0.2, 2.0, n_voxels)
    betas[2, :] = np.linspace(1.5, -0.3, n_voxels)
    betas[3, :] = rng.normal(scale=0.35, size=n_voxels)
    betas[4, :] = rng.normal(scale=0.25, size=n_voxels)
    betas[5, :] = rng.normal(scale=0.15, size=n_voxels)
    betas[6, :] = rng.normal(scale=0.10, size=n_voxels)

    noise = rng.normal(scale=0.2, size=(n_scans, n_voxels)).astype(np.float64)
    data = design @ betas + noise

    t_contrast = np.zeros(design.shape[1], dtype=np.float64)
    t_contrast[columns.index("condition_a")] = 1.0
    t_contrast[columns.index("condition_b")] = -1.0

    f_contrast = np.zeros((2, design.shape[1]), dtype=np.float64)
    f_contrast[0, columns.index("condition_a")] = 1.0
    f_contrast[1, columns.index("condition_b")] = 1.0

    return SyntheticFContrastInputs(
        design=design,
        data=data,
        columns=columns,
        t_contrast=t_contrast,
        f_contrast=f_contrast,
    )


def nilearn_pipeline(inputs: SyntheticFContrastInputs) -> PipelineOutput:
    """Run Nilearn's low-level OLS GLM and contrast machinery."""

    labels, estimates = run_glm(inputs.data, inputs.design, noise_model="ols")
    t_result = compute_contrast(
        labels,
        estimates,
        inputs.t_contrast,
        stat_type="t",
    )
    f_result = compute_contrast(
        labels,
        estimates,
        inputs.f_contrast,
        stat_type="F",
    )
    return PipelineOutput(
        arrays={
            "design": inputs.design,
            "effect_condition_a_minus_b": np.asarray(
                t_result.effect_size(),
                dtype=np.float64,
            ),
            "t_condition_a_minus_b": np.asarray(t_result.stat(), dtype=np.float64),
            "f_conditions_omnibus": np.asarray(f_result.stat(), dtype=np.float64),
        },
    )


def fmrimod_pipeline(inputs: SyntheticFContrastInputs) -> PipelineOutput:
    """Run fmrimod's low-level OLS solver and contrast machinery."""

    proj = fast_preproject(inputs.design)
    fit = fast_lm_matrix(inputs.design, inputs.data, proj)
    sigma = np.sqrt(fit.sigma2)
    t_result = contrast_t(
        inputs.t_contrast,
        fit.betas,
        proj.XtXinv,
        sigma,
        fit.dfres,
        name="condition_a_minus_b",
    )
    f_result = contrast_f(
        inputs.f_contrast,
        fit.betas,
        proj.XtXinv,
        sigma,
        fit.dfres,
        name="conditions_omnibus",
    )
    return PipelineOutput(
        arrays={
            "design": inputs.design,
            "effect_condition_a_minus_b": t_result.estimate,
            "t_condition_a_minus_b": t_result.stat,
            "f_conditions_omnibus": f_result.stat,
        },
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    return ParityCase(
        name="tier_a_synthetic_f_confound_drift",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_condition_a_minus_b": ParityTolerance(rtol=1e-9, atol=1e-9),
            "t_condition_a_minus_b": ParityTolerance(rtol=1e-8, atol=1e-8),
            "f_conditions_omnibus": ParityTolerance(rtol=1e-8, atol=1e-8),
        },
    )


def main() -> None:
    result = run(make_case())
    out_dir = Path(__file__).resolve().parent / "reports"
    render(result, out_dir)
    if result.status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
