"""Localizer first-level contrast parity against Nilearn."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_localizer_first_level
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import load_img, new_img_like
from numpy.typing import NDArray

from cross_testing.fitlins_parity import fit_fitlins_reference_ols
from cross_testing.harness import (
    Caveat,
    ParityCase,
    ParityTolerance,
    PipelineOutput,
    render,
    run,
)
from fmrimod.glm import fit_glm_from_matrix


Array = NDArray[np.float64]
TR = 2.4
MAX_VOXELS = 2048


@dataclass(frozen=True)
class LocalizerInputs:
    """Inputs shared by the localizer reference and fmrimod paths."""

    img: Any
    events: pd.DataFrame
    mask_img: Any
    contrast: Array


def _sparse_mask_img(img: Any, max_voxels: int = MAX_VOXELS) -> Any:
    first_volume = np.asarray(img.dataobj[..., 0])
    base_mask = first_volume != 0
    candidate = np.flatnonzero(base_mask.ravel())
    if candidate.size == 0:
        raise ValueError("Localizer first volume produced an empty mask")
    selected = candidate[
        np.linspace(0, candidate.size - 1, min(max_voxels, candidate.size), dtype=int)
    ]
    mask_flat = np.zeros(base_mask.size, dtype=np.uint8)
    mask_flat[selected] = 1
    return new_img_like(img, mask_flat.reshape(base_mask.shape))


def _contrast(columns: list[str]) -> Array:
    vector = np.zeros(len(columns), dtype=np.float64)
    vector[columns.index("audio_computation")] = 1.0
    vector[columns.index("visual_computation")] = -1.0
    return vector


def _fit_reference(
    img: Any,
    events: pd.DataFrame,
    mask_img: Any,
) -> FirstLevelModel:
    model = FirstLevelModel(
        t_r=TR,
        hrf_model="spm",
        drift_model="cosine",
        noise_model="ols",
        mask_img=mask_img,
        standardize=False,
        signal_scaling=False,
        minimize_memory=False,
        verbose=0,
    )
    model.fit(img, events=events)
    return model


def load_inputs(max_voxels: int = MAX_VOXELS) -> LocalizerInputs:
    """Load localizer image/events and derive the canonical contrast."""

    data = fetch_localizer_first_level(verbose=0)
    img = load_img(data.epi_img)
    events = pd.read_csv(data.events, sep="\t")
    mask_img = _sparse_mask_img(img, max_voxels=max_voxels)
    model = _fit_reference(img, events, mask_img)
    return LocalizerInputs(
        img=img,
        events=events,
        mask_img=mask_img,
        contrast=_contrast(list(model.design_matrices_[0].columns)),
    )


def nilearn_pipeline(inputs: LocalizerInputs) -> PipelineOutput:
    """Run Nilearn's GLM path on the FirstLevelModel localizer design."""

    model = _fit_reference(inputs.img, inputs.events, inputs.mask_img)
    X = model.design_matrices_[0].to_numpy(dtype=np.float64)
    Y = model.masker_.transform(inputs.img)
    fit = fit_fitlins_reference_ols(X, Y, inputs.contrast)
    effect = inputs.contrast @ fit["betas"]
    return PipelineOutput(
        arrays={
            "effect_audio_gt_visual": effect,
            "t_audio_gt_visual": fit["t"],
        }
    )


def fmrimod_pipeline(inputs: LocalizerInputs) -> PipelineOutput:
    """Fit Nilearn's localizer design with fmrimod's public OLS path.

    The X/Y-aware path preserves residual information needed for
    cancellation-safe variance recovery on sparse boundary voxels.
    """
    model = _fit_reference(inputs.img, inputs.events, inputs.mask_img)
    X = model.design_matrices_[0].to_numpy(dtype=np.float64)
    Y = model.masker_.transform(inputs.img)

    fit = fit_glm_from_matrix(X, Y)
    cres = fit.contrast(inputs.contrast)
    return PipelineOutput(
        arrays={
            "effect_audio_gt_visual": cres.estimate,
            "t_audio_gt_visual": cres.stat,
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    caveat = Caveat(
        caveat_id="localizer-tstat-variance-outliers",
        quantity="t_audio_gt_visual",
        reason=(
            "fmrimod now uses an X/Y-aware OLS path with explicit residual RSS "
            "recovery, but Nilearn run_glm still differs on a small set of "
            "near-zero-dispersion sparse-mask voxels; the parity gate therefore "
            "uses MAE and rank stability for the t map."
        ),
        expected="effect equality with low-MAE, high-rank-correlation t statistics",
        link="docs/contracts/fitlins_nilearn_overlap_v1.md#reference-workflow-tiers",
    )
    return ParityCase(
        name="tier_a_localizer_audio_gt_visual",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        declared_caveats=(caveat,),
        tolerances={
            "effect_audio_gt_visual": ParityTolerance(rtol=1e-6, atol=1e-7),
            "t_audio_gt_visual": ParityTolerance(
                check_allclose=False,
                min_pearson=0.95,
                min_spearman=0.99,
                max_mae=0.03,
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
