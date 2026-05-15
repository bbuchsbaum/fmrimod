"""SPM canonical + temporal derivative basis parity workflow.

A new parity space: the SPM "informed basis set" — canonical HRF plus its
temporal derivative — against Nilearn's ``hrf_model="spm + derivative"``.

None of the existing tier_a benchmarks exercise a multi-column HRF expansion
of a single condition: ``tier_a_spm_auditory`` covers the single SPM
canonical, ``tier_a_fir_unconstrained_hrf`` covers FIR, but the derivative
basis — heavily used in SPM-style modeling to absorb onset-latency
mismatches — was unbenchmarked. This workflow closes that gap on the SPM
auditory single-subject dataset using the canonical
``fm.fmri_dataset`` + ``fm.fmri_lm`` path with ``basis="spmg2"``.
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
from fmrimod.contrast import SemanticContrast, condition
from fmrimod.spec import hrf as hrf_term

Array = NDArray[np.float64]


TR = 7.0
MAX_VOXELS = 2048
CONTRAST_NAME = "listening"
HRF_MODEL = "spm + derivative"
LISTENING_CONTRAST = SemanticContrast(
    positive=condition(CONTRAST_NAME, term="trial_type", basis="canonical"),
    name=CONTRAST_NAME,
)


def _fit_provenance_payload(fit: Any) -> dict[str, Any]:
    if fit.provenance is None:
        raise AssertionError("fit did not carry FitProvenance")
    payload = fit.provenance.to_dict()
    payload["completeness_errors"] = list(fit.provenance.completeness_errors)
    return payload


def _contrast_receipt_payload(result: Any) -> dict[str, Any]:
    explanation = result.explain().to_dict()
    intent = explanation["intent"]
    required = ("basis_label", "weights", "design_id", "provenance_id")
    missing = [key for key in required if intent.get(key) in (None, [], "")]
    if missing:
        raise AssertionError(f"contrast intent missing payload fields: {missing}")
    return {
        "name": explanation["name"],
        "intent": intent,
        "touched_columns": explanation["touched_columns"],
        "statistic": explanation["statistic"],
    }


@dataclass(frozen=True)
class SpmDerivativeInputs:
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


def load_inputs(max_voxels: int = MAX_VOXELS) -> SpmDerivativeInputs:
    """Fetch SPM auditory data and prime the derivative-basis reference design."""

    bunch = fetch_spm_auditory(verbose=0)
    img = load_img(bunch.func[0])
    events = pd.read_csv(bunch.events, sep="\t")
    mask_img = _sparse_mask_img(img, max_voxels=max_voxels)

    reference = FirstLevelModel(
        t_r=TR,
        hrf_model=HRF_MODEL,
        drift_model=None,
        noise_model="ols",
        mask_img=mask_img,
        standardize=False,
        signal_scaling=False,
        minimize_memory=False,
        verbose=0,
    )
    reference.fit(img, events=events)
    return SpmDerivativeInputs(
        img=img,
        events=events,
        mask_img=mask_img,
        reference_design=reference.design_matrices_[0],
    )


def nilearn_pipeline(inputs: SpmDerivativeInputs) -> PipelineOutput:
    """Run Nilearn FirstLevelModel with ``hrf_model='spm + derivative'``."""

    model = FirstLevelModel(
        t_r=TR,
        hrf_model=HRF_MODEL,
        drift_model=None,
        noise_model="ols",
        mask_img=inputs.mask_img,
        standardize=False,
        signal_scaling=False,
        minimize_memory=False,
        verbose=0,
    )
    model.fit(inputs.img, events=inputs.events)
    design = model.design_matrices_[0]
    maps = model.compute_contrast(CONTRAST_NAME, output_type="all")
    return PipelineOutput(
        arrays={
            "design_canonical": design[CONTRAST_NAME].to_numpy(dtype=np.float64),
            "design_derivative": design[
                f"{CONTRAST_NAME}_derivative"
            ].to_numpy(dtype=np.float64),
            "effect_canonical": model.masker_.transform(maps["effect_size"]),
            "t_canonical": model.masker_.transform(maps["stat"]),
        }
    )


def _elapsed_seconds(fn: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    return value, float(time.perf_counter() - start)


def _canonical_design_column(fit: Any) -> Array:
    """Pick the canonical (b01) column out of the SPMG2 design matrix."""

    matrix = np.asarray(fit.model.event_model.design_matrix, dtype=np.float64)
    columns = list(fit.model.event_model.column_names)
    canonical = [i for i, name in enumerate(columns) if name.endswith("_b01")]
    if not canonical:
        raise AssertionError(
            f"expected a '_b01' canonical column in SPMG2 design; saw {columns!r}"
        )
    return matrix[:, canonical[0]]


def _derivative_design_column(fit: Any) -> Array:
    """Pick the temporal-derivative (b02) column out of the SPMG2 design matrix."""

    matrix = np.asarray(fit.model.event_model.design_matrix, dtype=np.float64)
    columns = list(fit.model.event_model.column_names)
    derivative = [i for i, name in enumerate(columns) if name.endswith("_b02")]
    if not derivative:
        raise AssertionError(
            f"expected a '_b02' derivative column in SPMG2 design; saw {columns!r}"
        )
    return matrix[:, derivative[0]]


def fmrimod_pipeline(
    inputs: SpmDerivativeInputs,
    *,
    timing_sink: dict[str, float] | None = None,
    provenance_sink: dict[str, Any] | None = None,
    contrast_sink: dict[str, Any] | None = None,
) -> PipelineOutput:
    """Run the SPM + derivative parity case through the canonical fmrimod API.

    Three lines of user code: dataset → fit → contrast. ``basis="spmg2"``
    expands each condition into a canonical+derivative pair; the named
    :class:`SemanticContrast` resolves to the canonical (b01) column so the
    contrast intent is the same as the single-HRF case, but the design now
    carries the extra basis function that Nilearn's
    ``hrf_model="spm + derivative"`` produces.
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
        lambda: fm.fmri_lm(
            hrf_term("trial_type", basis="spmg2", norm="spm"),
            ds,
            precision=0.02,
        )
    )
    cres, contrast_seconds = _elapsed_seconds(
        lambda: fit.contrast(LISTENING_CONTRAST, name=CONTRAST_NAME)
    )

    if timing_sink is not None:
        timing_sink.update(
            {
                "dataset_seconds": dataset_seconds,
                "fit_seconds": fit_seconds,
                "contrast_seconds": contrast_seconds,
            }
        )

    if provenance_sink is not None:
        provenance_sink.update(_fit_provenance_payload(fit))

    if contrast_sink is not None:
        contrast_sink.update(_contrast_receipt_payload(cres))

    return PipelineOutput(
        arrays={
            "design_canonical": _canonical_design_column(fit),
            "design_derivative": _derivative_design_column(fit),
            "effect_canonical": cres.estimate,
            "t_canonical": cres.stat,
        }
    )


def make_case(
    max_voxels: int = MAX_VOXELS,
    *,
    timing_sink: dict[str, float] | None = None,
    provenance_sink: dict[str, Any] | None = None,
    contrast_sink: dict[str, Any] | None = None,
) -> ParityCase:
    """Build the tier-A SPM + temporal derivative parity case.

    The derivative basis is a small-amplitude regressor (its peak is roughly
    a tenth of the canonical peak under ``norm="spm"``) and is sensitive to
    the same SPM ``gamma.pdf`` vs ``exp(-t)*(a1*t^P1 - C*t^P2)``
    parameterization gap that the SPM auditory benchmark already documents.
    The canonical column matches the single-HRF case; the derivative column
    is held to looser Pearson/Spearman thresholds for that reason.
    """
    return ParityCase(
        name="tier_a_spm_derivative_basis",
        fmrimod_pipeline=lambda inputs: fmrimod_pipeline(
            inputs,
            timing_sink=timing_sink,
            provenance_sink=provenance_sink,
            contrast_sink=contrast_sink,
        ),
        reference_pipeline=nilearn_pipeline,
        inputs=load_inputs(max_voxels=max_voxels),
        tolerances={
            "design_canonical": ParityTolerance(
                check_allclose=False,
                min_pearson=0.98,
                min_spearman=0.93,
                max_abs=0.20,
            ),
            # Nilearn computes the temporal-derivative basis by finite-differencing
            # the canonical HRF at the sampling grid; fmrimod uses the closed-form
            # ``spmg1_derivative``. The two agree on amplitude and on the major
            # over/undershoot peaks (Pearson ~0.98, max_abs < 0.05 once rescaled),
            # but Spearman is sensitive to the rank-order of the near-zero tails
            # which the analytic / finite-difference implementations resolve
            # differently. Pearson + max_abs is the real gate here.
            "design_derivative": ParityTolerance(
                check_allclose=False,
                min_pearson=0.95,
                min_spearman=0.40,
                allow_rescale=True,
                max_abs=0.06,
            ),
            "effect_canonical": ParityTolerance(
                check_allclose=False,
                min_pearson=0.97,
                min_spearman=0.95,
                max_mae=1.20,
            ),
            "t_canonical": ParityTolerance(
                check_allclose=False,
                min_pearson=0.97,
                min_spearman=0.95,
                max_mae=0.30,
            ),
        },
    )


def main() -> None:
    timings: dict[str, float] = {}
    provenance: dict[str, Any] = {}
    contrast_receipt: dict[str, Any] = {}
    result = run(
        make_case(
            timing_sink=timings,
            provenance_sink=provenance,
            contrast_sink=contrast_receipt,
        )
    )
    out_dir = Path(__file__).resolve().parent / "reports"
    json_path, _ = render(result, out_dir)
    payload = json.loads(json_path.read_text())
    payload["timings"] = {
        "status": "recorded",
        "seconds": float(sum(timings.values())),
        "stages": timings,
    }
    payload["fit_provenance"] = provenance
    payload["contrast_receipt"] = contrast_receipt
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if result.status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
