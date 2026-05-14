"""FIAC event-related and fixed-effects parity against Nilearn."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from nilearn.datasets import fetch_fiac_first_level
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import load_img, new_img_like
from nilearn.maskers import NiftiMasker
from numpy.typing import NDArray

import fmrimod as fm
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
MAX_VOXELS = 2048


@dataclass(frozen=True)
class FiacInputs:
    """Inputs shared by the FIAC reference and fmrimod paths.

    ``contrast_vectors`` is the nilearn-side weight realisation (one entry
    per contrast name); the fmrimod path constructs the same contrasts
    via :func:`fmrimod.column_contrast` against the design columns and
    never consumes ``contrast_vectors`` itself.
    """

    run_imgs: tuple[Any, Any]
    design_matrices: tuple[Any, Any]
    mask_img: Any
    contrast_vectors: dict[str, Array]


def _sparse_mask_from_mask(mask_img: Any, max_voxels: int) -> Any:
    mask = np.asarray(mask_img.get_fdata(), dtype=bool)
    candidate = np.flatnonzero(mask.ravel())
    if candidate.size == 0:
        raise ValueError("FIAC mask is empty")
    selected = candidate[
        np.linspace(0, candidate.size - 1, min(max_voxels, candidate.size), dtype=int)
    ]
    sparse = np.zeros(mask.size, dtype=np.uint8)
    sparse[selected] = 1
    return new_img_like(mask_img, sparse.reshape(mask.shape))


def _contrast_vector(columns: list[str], weights: dict[str, float]) -> Array:
    """Reference-side weight realisation for Nilearn's ``compute_contrast``."""
    vector = np.zeros(len(columns), dtype=np.float64)
    for name, weight in weights.items():
        vector[columns.index(name)] = float(weight)
    return vector


# The FIAC reference design names its 2x2 factorial cells as
# ``{SSt|DSt}-{SSp|DSp}`` (Same/Different Sentence x Same/Different
# Speaker). The main effects collapse the off-axis dimension.
_FIAC_CONTRAST_PATTERNS: dict[str, tuple[str, str]] = {
    "sentence_effect": (r"^SSt-", r"^DSt-"),
    "speaker_effect": (r"-SSp$", r"-DSp$"),
}


def load_inputs(max_voxels: int = MAX_VOXELS) -> FiacInputs:
    """Load FIAC run images, design matrices, sparse mask, and contrasts."""

    data = fetch_fiac_first_level(verbose=0)
    design1 = data.design_matrix1.copy()
    design2 = data.design_matrix2.copy()
    columns = list(design1.columns)
    contrast_vectors = {
        "sentence_effect": _contrast_vector(
            columns,
            {
                "SSt-SSp": 0.5,
                "SSt-DSp": 0.5,
                "DSt-SSp": -0.5,
                "DSt-DSp": -0.5,
            },
        ),
        "speaker_effect": _contrast_vector(
            columns,
            {
                "SSt-SSp": 0.5,
                "SSt-DSp": -0.5,
                "DSt-SSp": 0.5,
                "DSt-DSp": -0.5,
            },
        ),
    }
    return FiacInputs(
        run_imgs=(load_img(data.func1), load_img(data.func2)),
        design_matrices=(design1, design2),
        mask_img=_sparse_mask_from_mask(load_img(data.mask), max_voxels=max_voxels),
        contrast_vectors=contrast_vectors,
    )


def nilearn_pipeline(inputs: FiacInputs) -> PipelineOutput:
    """Run Nilearn FirstLevelModel on both FIAC runs."""

    model = FirstLevelModel(
        noise_model="ols",
        mask_img=inputs.mask_img,
        standardize=False,
        signal_scaling=False,
        minimize_memory=False,
        verbose=0,
    )
    model.fit(
        list(inputs.run_imgs),
        design_matrices=list(inputs.design_matrices),
    )
    arrays = {}
    for name, contrast in inputs.contrast_vectors.items():
        maps = model.compute_contrast(contrast, output_type="all")
        arrays[f"effect_{name}"] = model.masker_.transform(maps["effect_size"])
        arrays[f"t_{name}"] = model.masker_.transform(maps["stat"])
    return PipelineOutput(arrays=arrays)


def fmrimod_pipeline(
    inputs: FiacInputs,
    *,
    timing_sink: dict[str, float] | None = None,
) -> PipelineOutput:
    """FIAC public-seam path: per-run RealizedDesign -> fmri_lm -> combine_runs.

    Each run is fit through the public ``fmri_dataset -> fmri_lm``
    seam: Nilearn's externally realized design matrix is lifted into a
    typed :class:`~fmrimod.design.RealizedDesign` (with
    ``source='nilearn'`` so the provenance survives), and the per-run
    fits are pooled with :func:`fmrimod.combine_runs`. The two 2x2
    factorial main effects (``sentence_effect``, ``speaker_effect``)
    resolve as :func:`fmrimod.column_contrast` patterns against the
    realised design columns — no raw weight vectors are passed through
    the flagship path. The Moore-Penrose solver matches Nilearn's
    least-squares route for strict numerical parity.
    """
    start = time.perf_counter()
    masker = NiftiMasker(mask_img=inputs.mask_img).fit()
    if timing_sink is not None:
        timing_sink["fmrimod_masker_seconds"] = time.perf_counter() - start
    fits = []
    for index, (img, design) in enumerate(
        zip(inputs.run_imgs, inputs.design_matrices),
        start=1,
    ):
        dataset_start = time.perf_counter()
        X = np.asarray(design, dtype=np.float64)
        Y = masker.transform(img)
        dataset = fm.fmri_dataset(
            Y,
            tr=float(np.diff(design.index.values[:2])[0]),
            mask=np.ones(Y.shape[1], dtype=bool),
        )
        if timing_sink is not None:
            timing_sink[f"fmrimod_run{index}_dataset_seconds"] = (
                time.perf_counter() - dataset_start
            )
        design_start = time.perf_counter()
        realised = RealizedDesign.from_array(
            X,
            columns=tuple(str(column) for column in design.columns),
            source="nilearn",
        )
        if timing_sink is not None:
            timing_sink[f"fmrimod_run{index}_design_seconds"] = (
                time.perf_counter() - design_start
            )
        fit_start = time.perf_counter()
        fits.append(fm.fmri_lm(realised, dataset, config=FmriLmConfig(solver="pinv")))
        if timing_sink is not None:
            timing_sink[f"fmrimod_run{index}_fit_seconds"] = (
                time.perf_counter() - fit_start
            )

    for fit in fits:
        if fit.provenance is None or fit.provenance.design_source != "nilearn":
            raise AssertionError(
                "FIAC fmrimod path did not carry design_source='nilearn'"
            )

    combine_start = time.perf_counter()
    combined = fm.combine_runs(fits)
    if timing_sink is not None:
        timing_sink["fmrimod_combine_seconds"] = time.perf_counter() - combine_start

    arrays = {}
    contrast_start = time.perf_counter()
    for name, (pattern_A, pattern_B) in _FIAC_CONTRAST_PATTERNS.items():
        result = combined.contrast(
            fm.column_contrast(pattern_A=pattern_A, pattern_B=pattern_B, name=name)
        )
        arrays[f"effect_{name}"] = result.estimate
        arrays[f"t_{name}"] = result.stat
    if timing_sink is not None:
        timing_sink["fmrimod_contrast_seconds"] = time.perf_counter() - contrast_start
    return PipelineOutput(arrays=arrays)


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    return ParityCase(
        name="tier_a_fiac_fixed_effects",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "effect_sentence_effect": ParityTolerance(),
            "t_sentence_effect": ParityTolerance(),
            "effect_speaker_effect": ParityTolerance(),
            "t_speaker_effect": ParityTolerance(),
        },
    )


def _make_timed_case(timings: dict[str, float]) -> ParityCase:
    start = time.perf_counter()
    inputs = load_inputs(max_voxels=MAX_VOXELS)
    timings["load_inputs_seconds"] = time.perf_counter() - start

    def _timed_fmrimod(shared_inputs: FiacInputs) -> PipelineOutput:
        return fmrimod_pipeline(shared_inputs, timing_sink=timings)

    def _timed_nilearn(shared_inputs: FiacInputs) -> PipelineOutput:
        start = time.perf_counter()
        output = nilearn_pipeline(shared_inputs)
        timings["nilearn_pipeline_seconds"] = time.perf_counter() - start
        return output

    return ParityCase(
        name="tier_a_fiac_fixed_effects",
        fmrimod_pipeline=_timed_fmrimod,
        reference_pipeline=_timed_nilearn,
        inputs=inputs,
        tolerances={
            "effect_sentence_effect": ParityTolerance(),
            "t_sentence_effect": ParityTolerance(),
            "effect_speaker_effect": ParityTolerance(),
            "t_speaker_effect": ParityTolerance(),
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
