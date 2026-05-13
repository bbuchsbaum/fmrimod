"""SPM auditory fmrimod-vs-Nilearn parity workflow.

The case intentionally uses a deterministic sparse mask to keep the public
workflow fast while still exercising real NIfTI data, Nilearn's
``FirstLevelModel``, fmrimod's HRF/design construction, and fmrimod's OLS
solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_spm_auditory
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import load_img, new_img_like
from numpy.typing import NDArray

import fmrimod as fm
from cross_testing.fitlins_parity import fit_fmrimod_ols
from cross_testing.harness import (
    Caveat,
    ParityCase,
    ParityTolerance,
    PipelineOutput,
    render,
    run,
)


Array = NDArray[np.float64]


TR = 7.0
MAX_VOXELS = 2048
CONTRAST_NAME = "listening"


@dataclass(frozen=True)
class SpmAuditoryInputs:
    """Inputs shared by the fmrimod and Nilearn pipelines."""

    img: Any
    events: pd.DataFrame
    mask_img: Any
    reference_design: pd.DataFrame


def _sparse_mask_img(img: Any, max_voxels: int = MAX_VOXELS) -> Any:
    first_volume = np.asarray(img.dataobj[..., 0])
    base_mask = first_volume != 0
    candidate = np.flatnonzero(base_mask.ravel())
    if candidate.size == 0:
        raise ValueError("SPM auditory first volume produced an empty mask")

    selected = candidate[
        np.linspace(0, candidate.size - 1, min(max_voxels, candidate.size), dtype=int)
    ]
    mask_flat = np.zeros(base_mask.size, dtype=np.uint8)
    mask_flat[selected] = 1
    return new_img_like(img, mask_flat.reshape(base_mask.shape))


def load_inputs(max_voxels: int = MAX_VOXELS) -> SpmAuditoryInputs:
    """Fetch SPM auditory data and construct the fixed sparse mask."""

    bunch = fetch_spm_auditory(verbose=0)
    img = load_img(bunch.func[0])
    events = pd.read_csv(bunch.events, sep="\t")
    mask_img = _sparse_mask_img(img, max_voxels=max_voxels)

    reference = FirstLevelModel(
        t_r=TR,
        hrf_model="spm",
        drift_model=None,
        noise_model="ols",
        mask_img=mask_img,
        standardize=False,
        signal_scaling=False,
        minimize_memory=False,
        verbose=0,
    )
    reference.fit(img, events=events)
    return SpmAuditoryInputs(
        img=img,
        events=events,
        mask_img=mask_img,
        reference_design=reference.design_matrices_[0],
    )


def nilearn_pipeline(inputs: SpmAuditoryInputs) -> PipelineOutput:
    """Run Nilearn FirstLevelModel on the SPM auditory case."""

    model = FirstLevelModel(
        t_r=TR,
        hrf_model="spm",
        drift_model=None,
        noise_model="ols",
        mask_img=inputs.mask_img,
        standardize=False,
        signal_scaling=False,
        minimize_memory=False,
        verbose=0,
    )
    model.fit(inputs.img, events=inputs.events)
    maps = model.compute_contrast(CONTRAST_NAME, output_type="all")
    return PipelineOutput(
        arrays={
            "design_listening": model.design_matrices_[0][CONTRAST_NAME].to_numpy(),
            "effect_listening": model.masker_.transform(maps["effect_size"]),
            "t_listening": model.masker_.transform(maps["stat"]),
        }
    )


def fmrimod_pipeline(inputs: SpmAuditoryInputs) -> PipelineOutput:
    """Run the matching fmrimod design and OLS fit on the same masked data."""

    n_scans = int(inputs.img.shape[-1])
    sampling_frame = fm.SamplingFrame(blocklens=[n_scans], tr=TR)
    events = inputs.events.assign(run=1)

    event_model = fm.event_model(
        "hrf(trial_type)",
        data=events,
        sampling_frame=sampling_frame,
        block="run",
        durations="duration",
        precision=0.02,
    )
    baseline = fm.baseline_model(
        basis="constant",
        sframe=sampling_frame,
        intercept="global",
    )

    event_column = np.asarray(event_model.design_matrix[:, 0], dtype=np.float64)
    intercept = np.asarray(baseline.design_matrix[:, 0], dtype=np.float64)
    design = np.column_stack([event_column, intercept])

    reference_event = inputs.reference_design[CONTRAST_NAME].to_numpy()
    scale = float((event_column @ reference_event) / (event_column @ event_column))
    data = FirstLevelModel(
        t_r=TR,
        hrf_model="spm",
        drift_model=None,
        noise_model="ols",
        mask_img=inputs.mask_img,
        standardize=False,
        signal_scaling=False,
        minimize_memory=False,
        verbose=0,
    )
    data.fit(inputs.img, events=inputs.events)
    Y = data.masker_.transform(inputs.img)

    fit = fit_fmrimod_ols(design, Y, np.array([1.0, 0.0], dtype=np.float64))
    return PipelineOutput(
        arrays={
            "design_listening": event_column,
            "effect_listening": fit["betas"][0] / scale,
            "t_listening": fit["t"],
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    """Build the P1 SPM auditory parity case."""

    caveat = Caveat(
        caveat_id="spm-auditory-hrf-grid-scale",
        quantity="design_listening,effect_listening,t_listening",
        reason=(
            "fmrimod and Nilearn use different SPM HRF normalization and "
            "sampling-grid conventions; design is evaluated with explicit "
            "scale alignment and maps are judged by rank/correlation envelopes."
        ),
        expected="high correlation with non-zero absolute deltas",
        link="docs/contracts/fitlins_nilearn_overlap_v1.md#p1-harness-and-canary",
    )
    return ParityCase(
        name="tier_a_spm_auditory",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        declared_caveats=(caveat,),
        tolerances={
            "design_listening": ParityTolerance(
                check_allclose=False,
                allow_rescale=True,
                min_pearson=0.98,
                min_spearman=0.93,
                max_abs=0.20,
            ),
            "effect_listening": ParityTolerance(
                check_allclose=False,
                min_pearson=0.99,
                min_spearman=0.98,
                max_mae=0.90,
            ),
            "t_listening": ParityTolerance(
                check_allclose=False,
                min_pearson=0.985,
                min_spearman=0.98,
                max_mae=0.20,
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
