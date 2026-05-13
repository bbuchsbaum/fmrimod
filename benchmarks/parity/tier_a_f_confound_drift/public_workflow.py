"""Public-seam F-contrast parity against Nilearn.

This companion to ``workflow.py`` runs the fmrimod side through the mission
path ``fmri_dataset -> fmri_lm -> contrast`` while using Nilearn's low-level
GLM as the numerical reference on the realised public design matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import run_glm
from numpy.typing import NDArray

import fmrimod as fm
from cross_testing.harness import (
    ParityCase,
    ParityTolerance,
    PipelineOutput,
    render,
    run,
)
from fmrimod.contrast import OmnibusContrast
from fmrimod.design import DesignColumns
from fmrimod.spec import confounds, drift, hrf, intercept

Array = NDArray[np.float64]
MAX_VOXELS = 2048
TR = 1.0


@dataclass(frozen=True)
class PublicFContrastInputs:
    """Inputs for the typed public-seam parity path."""

    events: pd.DataFrame
    confounds: pd.DataFrame
    data: Array
    design: pd.DataFrame
    design_columns: DesignColumns
    spec: Any
    t_contrast: Array
    omnibus: OmnibusContrast
    reference_f_contrast: Array


def _center_scale(values: Array) -> Array:
    values = np.asarray(values, dtype=np.float64)
    values = values - values.mean()
    scale = np.linalg.norm(values)
    if scale == 0:
        return values
    return values / scale


def _events() -> pd.DataFrame:
    rows = []
    for onset in range(8, 120, 24):
        rows.append(
            {
                "onset": float(onset),
                "duration": 6.0,
                "trial_type": "condition_a",
                "run": 1,
            }
        )
    for onset in range(18, 120, 24):
        rows.append(
            {
                "onset": float(onset),
                "duration": 5.0,
                "trial_type": "condition_b",
                "run": 1,
            }
        )
    return pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)


def _confounds(n_scans: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "motion_x": _center_scale(
                np.sin(np.linspace(0.0, 4.0 * np.pi, n_scans))
            ),
            "motion_y": _center_scale(
                np.cos(np.linspace(0.0, 3.0 * np.pi, n_scans))
            ),
        }
    )


def _make_spec(confound_df: pd.DataFrame) -> Any:
    return (
        hrf("trial_type", basis="spm")
        + confounds("motion_x", "motion_y", source=confound_df)
        + drift("poly", degree=2)
        + intercept("run")
    )


def _public_design(
    events: pd.DataFrame,
    confound_df: pd.DataFrame,
    n_scans: int,
) -> tuple[Any, pd.DataFrame, DesignColumns]:
    spec = _make_spec(confound_df)
    dataset = fm.fmri_dataset(
        np.zeros((n_scans, 1), dtype=np.float64),
        tr=TR,
        events=events,
    )
    fit = fm.fmri_lm(spec, dataset)
    return spec, fit.model.design_matrix(run=0), fit.design_columns()


def _contrast_vector(columns: list[str], weights: dict[str, float]) -> Array:
    vector = np.zeros(len(columns), dtype=np.float64)
    for name, weight in weights.items():
        vector[columns.index(name)] = float(weight)
    return vector


def _find_condition_columns(columns: list[str]) -> tuple[str, str]:
    condition_a = [
        col for col in columns
        if "trial_type" in col and "condition_a" in col
    ]
    condition_b = [
        col for col in columns
        if "trial_type" in col and "condition_b" in col
    ]
    if len(condition_a) != 1 or len(condition_b) != 1:
        raise ValueError(
            "Expected exactly one named design column for each condition; "
            f"got condition_a={condition_a}, condition_b={condition_b}"
        )
    return condition_a[0], condition_b[0]


def load_inputs(max_voxels: int = MAX_VOXELS) -> PublicFContrastInputs:
    """Build deterministic public-seam synthetic inputs."""

    n_scans = 120
    n_voxels = min(int(max_voxels), MAX_VOXELS)
    events = _events()
    confound_df = _confounds(n_scans)
    spec, design, design_columns = _public_design(events, confound_df, n_scans)
    columns = list(design.columns)
    condition_a, condition_b = _find_condition_columns(columns)
    omnibus = OmnibusContrast(
        "trial_type",
        levels=("condition_a", "condition_b"),
        name="conditions_omnibus",
    )

    t_contrast = _contrast_vector(
        columns,
        {
            condition_a: 1.0,
            condition_b: -1.0,
        },
    )
    reference_f_contrast = omnibus.resolve(design_columns)

    rng = np.random.default_rng(20260513)
    betas = np.zeros((design.shape[1], n_voxels), dtype=np.float64)
    betas[0, :] = np.linspace(0.2, 2.0, n_voxels)
    betas[1, :] = np.linspace(1.5, -0.3, n_voxels)
    for row in range(2, design.shape[1]):
        betas[row, :] = rng.normal(scale=0.20, size=n_voxels)
    data = np.asarray(design, dtype=np.float64) @ betas
    data += rng.normal(scale=0.2, size=(n_scans, n_voxels))

    return PublicFContrastInputs(
        events=events,
        confounds=confound_df,
        data=data.astype(np.float64),
        design=design,
        design_columns=design_columns,
        spec=spec,
        t_contrast=t_contrast,
        omnibus=omnibus,
        reference_f_contrast=reference_f_contrast,
    )


def nilearn_pipeline(inputs: PublicFContrastInputs) -> PipelineOutput:
    """Run Nilearn's low-level GLM on the realised public fmrimod design."""

    design = np.asarray(inputs.design, dtype=np.float64)
    labels, estimates = run_glm(inputs.data, design, noise_model="ols")
    t_result = compute_contrast(
        labels,
        estimates,
        inputs.t_contrast,
        stat_type="t",
    )
    f_result = compute_contrast(
        labels,
        estimates,
        inputs.reference_f_contrast,
        stat_type="F",
    )
    return PipelineOutput(
        arrays={
            "design": design,
            "effect_condition_a_minus_b": np.asarray(
                t_result.effect_size(),
                dtype=np.float64,
            ),
            "t_condition_a_minus_b": np.asarray(t_result.stat(), dtype=np.float64),
            "f_conditions_omnibus": np.asarray(f_result.stat(), dtype=np.float64),
        },
    )


def fmrimod_pipeline(inputs: PublicFContrastInputs) -> PipelineOutput:
    """Run the fmrimod typed public seam and public contrast method."""

    dataset = fm.fmri_dataset(inputs.data, tr=TR, events=inputs.events)
    fit = fm.fmri_lm(inputs.spec, dataset)
    t_result = fit.contrast(
        inputs.t_contrast,
        name="condition_a_minus_b",
    )
    f_result = fit.contrast(inputs.omnibus)
    return PipelineOutput(
        arrays={
            "design": fit.model.design_matrix_array(run=0),
            "effect_condition_a_minus_b": t_result.estimate,
            "t_condition_a_minus_b": t_result.stat,
            "f_conditions_omnibus": f_result.stat,
        },
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    return ParityCase(
        name="tier_a_public_f_confound_drift",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design": ParityTolerance(rtol=0.0, atol=0.0),
            "effect_condition_a_minus_b": ParityTolerance(rtol=1e-8, atol=1e-8),
            "t_condition_a_minus_b": ParityTolerance(rtol=1e-7, atol=1e-7),
            "f_conditions_omnibus": ParityTolerance(rtol=1e-7, atol=1e-7),
        },
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports" / "public",
        help="Directory where the parity report should be rendered.",
    )
    args = parser.parse_args(argv)

    result = run(make_case())
    render(result, args.out_dir)
    if result.status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
