"""SPM auditory fmrimod-vs-Nilearn parity workflow.

The fmrimod side uses the canonical ``fm.fmri_dataset`` + ``fm.fmri_lm`` entry
points: three lines of user code to go from a NIfTI + events to a fitted
contrast. The Nilearn reference pipeline still uses ``FirstLevelModel``
verbatim.
"""

from __future__ import annotations

import json
import time
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
from cross_testing.harness import (
    ParityCase,
    ParityTolerance,
    PipelineOutput,
    render,
    run,
)
from fmrimod.spec import hrf as hrf_term

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


def _elapsed_seconds(fn: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    return value, float(time.perf_counter() - start)


def fmrimod_pipeline(
    inputs: SpmAuditoryInputs,
    *,
    timing_sink: dict[str, float] | None = None,
) -> PipelineOutput:
    """Run the SPM auditory parity case through the canonical fmrimod API.

    Three lines of user code: dataset → fit → contrast. ``norm="spm"`` puts
    the HRF on Nilearn's unit-integral scale; no post-hoc rescaling needed.
    """
    ds, dataset_seconds = _elapsed_seconds(
        lambda: fm.fmri_dataset(
            inputs.img,
            mask=inputs.mask_img,
            tr=TR,
            events=inputs.events.assign(run=1),
        )
    )
    fit, fit_seconds = _elapsed_seconds(
        lambda: fm.fmri_lm(hrf_term("trial_type", norm="spm"), ds, precision=0.02)
    )
    cres, contrast_seconds = _elapsed_seconds(
        lambda: fit.contrast(np.array([1.0, 0.0], dtype=np.float64))
    )

    if timing_sink is not None:
        timing_sink.update(
            {
                "dataset_seconds": dataset_seconds,
                "fit_seconds": fit_seconds,
                "contrast_seconds": contrast_seconds,
            }
        )

    return PipelineOutput(
        arrays={
            "design_listening": np.asarray(
                fit.model.event_model.design_matrix[:, 0], dtype=np.float64
            ),
            "effect_listening": cres.estimate,
            "t_listening": cres.stat,
        }
    )


def make_case(
    max_voxels: int = MAX_VOXELS,
    *,
    timing_sink: dict[str, float] | None = None,
) -> ParityCase:
    """Build the P1 SPM auditory parity case.

    With ``hrf(..., norm="spm")`` the design column lands on Nilearn's
    unit-integral scale, closing the historical ~0.002 amplitude gap. A
    small residual (~4 %) remains because Nilearn's ``_gamma_difference_hrf``
    uses ``scipy.stats.gamma.pdf`` while fmrimod evaluates the SPM
    ``exp(-t) * (a1*t^P1 - C*t^P2)`` parameterization at the same grid.
    """
    return ParityCase(
        name="tier_a_spm_auditory",
        fmrimod_pipeline=lambda inputs: fmrimod_pipeline(
            inputs,
            timing_sink=timing_sink,
        ),
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
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
    timings: dict[str, float] = {}
    result = run(make_case(timing_sink=timings))
    out_dir = Path(__file__).resolve().parent / "reports"
    json_path, _ = render(result, out_dir)
    payload = json.loads(json_path.read_text())
    payload["timings"] = {
        "status": "recorded",
        "seconds": float(sum(timings.values())),
        "stages": timings,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if result.status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
