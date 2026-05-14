"""Synthetic second-level parity against Nilearn SecondLevelModel."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from numpy.typing import NDArray
from scipy import stats as sp_stats

from cross_testing.harness import (
    ParityCase,
    ParityTolerance,
    PipelineOutput,
    render,
    run,
)
from fmrimod.dataset import group_data_from_csv
from fmrimod.group import group_dataset_from_group_data, group_model, ols_voxelwise

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
    images = [
        nib.Nifti1Image(row.reshape(shape).astype(np.float32), affine)
        for row in values
    ]
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

    design_reg = pd.DataFrame(
        {"intercept": np.ones(len(inputs.images)), "age": inputs.age}
    )
    reg = SecondLevelModel(mask_img=inputs.mask_img, smoothing_fwhm=None)
    reg.fit(inputs.images, design_matrix=design_reg)
    age_maps = reg.compute_contrast("age", output_type="all")

    return PipelineOutput(
        arrays={
            "one_sample_effect": one.masker_.transform(one_maps["effect_size"]),
            "one_sample_t": one.masker_.transform(one_maps["stat"]),
            "age_effect": reg.masker_.transform(age_maps["effect_size"]),
            "age_t": reg.masker_.transform(age_maps["stat"]),
            "age_p_signed_one_sided": reg.masker_.transform(age_maps["p_value"]),
        }
    )


def fmrimod_pipeline(
    inputs: SecondLevelInputs,
    *,
    timing_sink: dict[str, float] | None = None,
) -> PipelineOutput:
    one_sample_start = time.perf_counter()
    gd_one = group_data_from_csv(
        inputs.data,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="feature",
    )
    one = ols_voxelwise(group_dataset_from_group_data(gd_one), model=group_model())
    if timing_sink is not None:
        timing_sink["fmrimod_one_sample_seconds"] = (
            time.perf_counter() - one_sample_start
        )

    regression_start = time.perf_counter()
    gd_reg = group_data_from_csv(
        inputs.data,
        effect_cols={"beta": "beta", "se": "se"},
        subject_col="subject",
        roi_col="feature",
        covariates=inputs.covariates,
    )
    reg = ols_voxelwise(group_dataset_from_group_data(gd_reg), model=group_model("age"))
    if timing_sink is not None:
        timing_sink["fmrimod_age_regression_seconds"] = (
            time.perf_counter() - regression_start
        )

    return PipelineOutput(
        arrays={
            "one_sample_effect": one.assay("coef:Intercept")[:, 0, 0],
            "one_sample_t": one.assay("t_coef:Intercept")[:, 0, 0],
            "age_effect": reg.assay("coef:age")[:, 0, 0],
            "age_t": reg.assay("t_coef:age")[:, 0, 0],
            # Nilearn's SecondLevelModel exposes a signed one-sided p-value
            # derived from the t statistic. fmrimod's native p_coef assays
            # stay two-sided; this benchmark quantity is named explicitly so
            # parity does not redefine the public group p-value contract.
            "age_p_signed_one_sided": sp_stats.t.sf(
                reg.assay("t_coef:age")[:, 0, 0],
                df=len(inputs.images) - 2,
            ),
        }
    )


def make_case(
    *,
    timing_sink: dict[str, float] | None = None,
) -> ParityCase:
    if timing_sink is not None:
        start = time.perf_counter()
        inputs = make_inputs()
        timing_sink["make_inputs_seconds"] = time.perf_counter() - start

        def _timed_fmrimod(shared_inputs: SecondLevelInputs) -> PipelineOutput:
            return fmrimod_pipeline(shared_inputs, timing_sink=timing_sink)

        def _timed_nilearn(shared_inputs: SecondLevelInputs) -> PipelineOutput:
            nilearn_start = time.perf_counter()
            output = nilearn_pipeline(shared_inputs)
            timing_sink["nilearn_pipeline_seconds"] = (
                time.perf_counter() - nilearn_start
            )
            return output

        fmrimod_fn = _timed_fmrimod
        nilearn_fn = _timed_nilearn
    else:
        inputs = make_inputs()
        fmrimod_fn = fmrimod_pipeline
        nilearn_fn = nilearn_pipeline

    return ParityCase(
        name="tier_c_second_level_synthetic",
        fmrimod_pipeline=fmrimod_fn,
        reference_pipeline=nilearn_fn,
        inputs=inputs,
        declared_caveats=(),
        tolerances={
            "one_sample_effect": ParityTolerance(rtol=1e-7, atol=1e-8),
            # numerical_floor: Nilearn's second-level path rounds the derived
            # t statistic differently from the native grouped reducer.
            "one_sample_t": ParityTolerance(rtol=1e-6, atol=1e-6),
            "age_effect": ParityTolerance(rtol=1e-7, atol=1e-8),
            # numerical_floor: same second-level t-statistic rounding floor as
            # the intercept test above.
            "age_t": ParityTolerance(rtol=1e-6, atol=1e-6),
            # numerical_floor: signed one-sided p is reconstructed from t/df
            # to mirror Nilearn's exposed quantity, so tiny tail-probability
            # differences are expected near scipy/Nilearn rounding boundaries.
            "age_p_signed_one_sided": ParityTolerance(rtol=2e-5, atol=2e-8),
        },
    )


def _write_timing_payload(json_path: Path, timings: dict[str, float]) -> None:
    payload = json.loads(json_path.read_text())
    stages = {key: float(value) for key, value in sorted(timings.items())}
    payload["timings"] = {
        "status": "recorded",
        "seconds": float(sum(stages.values())),
        "stages": stages,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    timings: dict[str, float] = {}
    result = run(make_case(timing_sink=timings))
    out_dir = Path(__file__).resolve().parent / "reports"
    json_path, _ = render(result, out_dir)
    _write_timing_payload(json_path, timings)
    if result.status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
