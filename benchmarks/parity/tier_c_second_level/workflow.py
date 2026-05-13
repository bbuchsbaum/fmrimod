"""Synthetic second-level parity against Nilearn SecondLevelModel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.second_level import SecondLevelModel
from numpy.typing import NDArray

from cross_testing.harness import Caveat, ParityCase, ParityTolerance, PipelineOutput, render, run
from fmrimod.dataset import group_data_from_csv
from fmrimod.stats import GroupFitRequest, group_fit


Array = NDArray[np.float64]


@dataclass(frozen=True)
class SecondLevelInputs:
    """Synthetic maps and tabular copy for second-level parity."""

    images: list[Any]
    mask_img: Any
    data: pd.DataFrame
    covariates: pd.DataFrame
    values: Array
    age: Array


def make_inputs(seed: int = 2026) -> SecondLevelInputs:
    rng = np.random.default_rng(seed)
    n_subjects = 10
    shape = (3, 3, 3)
    mask = np.ones(shape, dtype=np.uint8)
    affine = np.eye(4)
    mask_img = nib.Nifti1Image(mask, affine)
    age = np.linspace(-1.0, 1.0, n_subjects)

    intercept = np.linspace(0.1, 0.4, int(mask.sum()))
    slope = np.linspace(-0.25, 0.25, int(mask.sum()))
    values = np.vstack(
        [
            intercept + subj_age * slope + rng.normal(0.0, 0.03, int(mask.sum()))
            for subj_age in age
        ]
    )
    images = [nib.Nifti1Image(row.reshape(shape).astype(np.float32), affine) for row in values]
    subjects = [f"s{i + 1:02d}" for i in range(n_subjects)]
    rows = []
    for i, subject in enumerate(subjects):
        for j, beta in enumerate(values[i]):
            rows.append(
                {
                    "subject": subject,
                    "feature": f"v{j:03d}",
                    "beta": float(beta),
                    "se": 1.0,
                }
            )
    return SecondLevelInputs(
        images=images,
        mask_img=mask_img,
        data=pd.DataFrame(rows),
        covariates=pd.DataFrame({"subject": subjects, "age": age}),
        values=values,
        age=age,
    )


def nilearn_pipeline(inputs: SecondLevelInputs) -> PipelineOutput:
    design_one = pd.DataFrame({"intercept": np.ones(len(inputs.images))})
    one = SecondLevelModel(mask_img=inputs.mask_img, smoothing_fwhm=None)
    one.fit(inputs.images, design_matrix=design_one)
    one_maps = one.compute_contrast("intercept", output_type="all")

    design_reg = pd.DataFrame({"intercept": np.ones(len(inputs.images)), "age": inputs.age})
    reg = SecondLevelModel(mask_img=inputs.mask_img, smoothing_fwhm=None)
    reg.fit(inputs.images, design_matrix=design_reg)
    age_maps = reg.compute_contrast("age", output_type="all")

    return PipelineOutput(
        arrays={
            "one_sample_effect": one.masker_.transform(one_maps["effect_size"]),
            "one_sample_t": one.masker_.transform(one_maps["stat"]),
            "age_effect": reg.masker_.transform(age_maps["effect_size"]),
            "age_t": reg.masker_.transform(age_maps["stat"]),
        }
    )


def fmrimod_pipeline(inputs: SecondLevelInputs) -> PipelineOutput:
    gd_one = group_data_from_csv(
        inputs.data,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="feature",
    )
    one = group_fit(
        GroupFitRequest(
            data=gd_one,
            model="ttest",
            ttest_engine="classic",
            effects="fixed",
        )
    )

    gd_reg = group_data_from_csv(
        inputs.data,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="feature",
        covariates=inputs.covariates,
    )
    reg = group_fit(
        GroupFitRequest(
            data=gd_reg,
            model="meta",
            formula="~ age",
            method="fe",
            weights="equal",
        )
    )

    return PipelineOutput(
        arrays={
            "one_sample_effect": one.estimate[:, 0],
            "one_sample_t": one.statistic[:, 0],
            "age_effect": reg.estimate[:, 1],
            "age_t": reg.statistic[:, 1],
        }
    )


def make_case() -> ParityCase:
    caveat = Caveat(
        caveat_id="second-level-normal-vs-t-pvalues",
        quantity="age_t",
        reason=(
            "fmrimod fixed-effect meta regression uses supplied effect-size "
            "standard errors, while Nilearn SecondLevelModel uses a second-level "
            "OLS residual variance estimate. Coefficients are compared directly; "
            "the statistic is rank/correlation checked as a documented API gap."
        ),
        expected="effect parity and statistic-rank agreement; no p-value assertion",
        link="docs/contracts/second_level_parity_v1.md",
    )
    return ParityCase(
        name="tier_c_second_level_synthetic",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=make_inputs(),
        declared_caveats=(caveat,),
        tolerances={
            "one_sample_effect": ParityTolerance(rtol=1e-7, atol=1e-8),
            "one_sample_t": ParityTolerance(rtol=1e-6, atol=1e-6),
            "age_effect": ParityTolerance(rtol=1e-7, atol=1e-8),
            "age_t": ParityTolerance(
                check_allclose=False,
                min_pearson=0.95,
                min_spearman=0.95,
            ),
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
