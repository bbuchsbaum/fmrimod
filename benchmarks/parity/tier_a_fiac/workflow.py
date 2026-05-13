"""FIAC event-related and fixed-effects parity against Nilearn."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from nilearn.datasets import fetch_fiac_first_level
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import load_img, new_img_like
from nilearn.maskers import NiftiMasker
from numpy.typing import NDArray

from cross_testing.fitlins_parity import fit_fmrimod_ols
from cross_testing.harness import ParityCase, ParityTolerance, PipelineOutput, render, run


Array = NDArray[np.float64]
MAX_VOXELS = 2048


@dataclass(frozen=True)
class FiacInputs:
    """Inputs shared by the FIAC reference and fmrimod paths."""

    run_imgs: tuple[Any, Any]
    design_matrices: tuple[Any, Any]
    mask_img: Any
    contrasts: dict[str, Array]


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


def _contrast(columns: list[str], weights: dict[str, float]) -> Array:
    vector = np.zeros(len(columns), dtype=np.float64)
    for name, weight in weights.items():
        vector[columns.index(name)] = float(weight)
    return vector


def load_inputs(max_voxels: int = MAX_VOXELS) -> FiacInputs:
    """Load FIAC run images, design matrices, sparse mask, and contrasts."""

    data = fetch_fiac_first_level(verbose=0)
    design1 = data.design_matrix1.copy()
    design2 = data.design_matrix2.copy()
    columns = list(design1.columns)
    contrasts = {
        "sentence_effect": _contrast(
            columns,
            {
                "SSt-SSp": 0.5,
                "SSt-DSp": 0.5,
                "DSt-SSp": -0.5,
                "DSt-DSp": -0.5,
            },
        ),
        "speaker_effect": _contrast(
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
        contrasts=contrasts,
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
    for name, contrast in inputs.contrasts.items():
        maps = model.compute_contrast(contrast, output_type="all")
        arrays[f"effect_{name}"] = model.masker_.transform(maps["effect_size"])
        arrays[f"t_{name}"] = model.masker_.transform(maps["stat"])
    return PipelineOutput(arrays=arrays)


def _fixed_effects_from_runs(
    run_imgs: tuple[Any, Any],
    design_matrices: tuple[Any, Any],
    mask_img: Any,
    contrast: Array,
) -> tuple[Array, Array]:
    masker = NiftiMasker(mask_img=mask_img).fit()
    effects = []
    variances = []
    for img, design in zip(run_imgs, design_matrices):
        X = np.asarray(design, dtype=np.float64)
        Y = masker.transform(img)
        fit = fit_fmrimod_ols(X, Y, contrast)
        XtXinv = np.linalg.pinv(X.T @ X)
        var_factor = float(contrast @ XtXinv @ contrast)
        effects.append(contrast @ fit["betas"])
        variances.append(np.maximum(var_factor * fit["sigma2"], 1e-30))

    var_stack = np.vstack(variances)
    effect_stack = np.vstack(effects)
    fixed_effect = np.mean(effect_stack, axis=0)
    fixed_var = np.mean(var_stack, axis=0) / var_stack.shape[0]
    fixed_t = fixed_effect / np.sqrt(fixed_var)
    return fixed_effect, fixed_t


def fmrimod_pipeline(inputs: FiacInputs) -> PipelineOutput:
    """Run fmrimod OLS per run and combine contrasts by fixed effects."""

    arrays = {}
    for name, contrast in inputs.contrasts.items():
        effect, t_stat = _fixed_effects_from_runs(
            inputs.run_imgs,
            inputs.design_matrices,
            inputs.mask_img,
            contrast,
        )
        arrays[f"effect_{name}"] = effect
        arrays[f"t_{name}"] = t_stat
    return PipelineOutput(arrays=arrays)


def make_case(max_voxels: int = MAX_VOXELS) -> ParityCase:
    return ParityCase(
        name="tier_a_fiac_fixed_effects",
        fmrimod_pipeline=fmrimod_pipeline,
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "effect_sentence_effect": ParityTolerance(rtol=1e-5, atol=1e-7),
            "t_sentence_effect": ParityTolerance(rtol=1e-5, atol=1e-7),
            "effect_speaker_effect": ParityTolerance(rtol=1e-5, atol=1e-7),
            "t_speaker_effect": ParityTolerance(rtol=1e-5, atol=1e-7),
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
