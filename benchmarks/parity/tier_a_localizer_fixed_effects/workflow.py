"""Localizer first-level contrast parity against Nilearn."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_localizer_first_level
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import load_img, new_img_like
from numpy.typing import NDArray

import fmrimod as fm
from cross_testing.fitlins_parity import fit_fitlins_reference_ols
from cross_testing.harness import (
    ParityCase,
    ParityTolerance,
    PipelineOutput,
    render,
    run,
)
from fmrimod.design import RealizedDesign
from fmrimod.model.config import FmriLmConfig

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


def fmrimod_pipeline(
    inputs: LocalizerInputs,
    *,
    timing_sink: dict[str, float] | None = None,
) -> PipelineOutput:
    """Fit Nilearn's realized localizer design through fmrimod's public seam.

    Strict reference parity uses the explicit Moore-Penrose solver backend.
    The default fast path is numerically stable, but it can produce slightly
    different residual variance on sparse boundary voxels because it takes a
    different algebraic route through the same least-squares problem.
    """
    reference_start = time.perf_counter()
    model = _fit_reference(inputs.img, inputs.events, inputs.mask_img)
    if timing_sink is not None:
        timing_sink["fmrimod_reference_design_seconds"] = (
            time.perf_counter() - reference_start
        )
    design_frame = model.design_matrices_[0]
    X = design_frame.to_numpy(dtype=np.float64)
    Y = model.masker_.transform(inputs.img)

    dataset_start = time.perf_counter()
    dataset = fm.fmri_dataset(
        Y,
        tr=TR,
        mask=np.ones(Y.shape[1], dtype=bool),
        events=inputs.events,
    )
    if timing_sink is not None:
        timing_sink["fmrimod_dataset_seconds"] = time.perf_counter() - dataset_start

    design_start = time.perf_counter()
    design = RealizedDesign.from_array(
        X,
        columns=tuple(str(column) for column in design_frame.columns),
        source="nilearn",
    )
    if timing_sink is not None:
        timing_sink["fmrimod_design_seconds"] = time.perf_counter() - design_start

    fit_start = time.perf_counter()
    fit = fm.fmri_lm(design, dataset, config=FmriLmConfig(solver="pinv"))
    if timing_sink is not None:
        timing_sink["fmrimod_fit_seconds"] = time.perf_counter() - fit_start

    contrast_start = time.perf_counter()
    cres = fit.contrast(
        fm.column_contrast(
            "^audio_computation$",
            pattern_B="^visual_computation$",
            name="tier_a_localizer_audio_gt_visual",
        )
    )
    if timing_sink is not None:
        timing_sink["fmrimod_contrast_seconds"] = time.perf_counter() - contrast_start
    if fit.provenance is None or fit.provenance.design_source != "nilearn":
        raise AssertionError("fmrimod localizer path did not carry design source")
    return PipelineOutput(
        arrays={
            "effect_audio_gt_visual": cres.estimate,
            "t_audio_gt_visual": cres.stat,
        }
    )


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    return ParityCase(
        name="tier_a_localizer_audio_gt_visual",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        declared_caveats=(),
        tolerances={
            "effect_audio_gt_visual": ParityTolerance(),
            "t_audio_gt_visual": ParityTolerance(),
        },
    )


def _make_timed_case(timings: dict[str, float]) -> ParityCase:
    start = time.perf_counter()
    inputs = load_inputs(max_voxels=MAX_VOXELS)
    timings["load_inputs_seconds"] = time.perf_counter() - start

    def _timed_fmrimod(shared_inputs: LocalizerInputs) -> PipelineOutput:
        return fmrimod_pipeline(shared_inputs, timing_sink=timings)

    def _timed_nilearn(shared_inputs: LocalizerInputs) -> PipelineOutput:
        start = time.perf_counter()
        output = nilearn_pipeline(shared_inputs)
        timings["nilearn_pipeline_seconds"] = time.perf_counter() - start
        return output

    return ParityCase(
        name="tier_a_localizer_audio_gt_visual",
        fmrimod_pipeline=_timed_fmrimod,
        reference_pipeline=_timed_nilearn,
        inputs=inputs,
        declared_caveats=(),
        tolerances={
            "effect_audio_gt_visual": ParityTolerance(),
            "t_audio_gt_visual": ParityTolerance(),
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
    result = run(_make_timed_case(timings))
    out_dir = Path(__file__).resolve().parent / "reports"
    json_path, _ = render(result, out_dir)
    _write_timing_payload(json_path, timings)
    if result.status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
