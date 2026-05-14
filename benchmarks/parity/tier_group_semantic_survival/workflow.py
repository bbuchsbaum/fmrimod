"""Group-facing semantic-survival flagship demo.

This workflow starts from the public F-confound/drift seam and checks that a
typed first-level hypothesis remains named and inspectable when materialized as
a group-ready object. The point is not a new numerical parity oracle; it is the
artifact contract behind the README-grade group flagship.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import fmrimod as fm
from benchmarks.parity.tier_a_f_confound_drift.public_workflow import (
    TR,
    PublicFContrastInputs,
    load_inputs,
)
from fmrimod.group import SampleLabelSpace, group_dataset, group_model, ols_voxelwise


@dataclass(frozen=True)
class SemanticSurvivalResult:
    """Outputs from the semantic-survival group demo."""

    group: Any
    group_result: Any
    explanations: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def _fit_provenance_payload(fit: Any) -> dict[str, Any]:
    if fit.provenance is None:
        raise AssertionError("fit did not carry FitProvenance")
    payload = fit.provenance.to_dict()
    payload["completeness_errors"] = list(fit.provenance.completeness_errors)
    return payload


def _subject_data(
    inputs: PublicFContrastInputs,
    *,
    subject_index: int,
    n_subjects: int,
) -> NDArray[np.float64]:
    """Generate one deterministic synthetic subject from the public design."""
    design = np.asarray(inputs.design, dtype=np.float64)
    n_voxels = inputs.data.shape[1]
    n_coefficients = design.shape[1]
    rng = np.random.default_rng(20260513 + subject_index)

    a_col = inputs.design_columns.where(
        term="trial_type",
        level="condition_a",
    ).one().index
    b_col = inputs.design_columns.where(
        term="trial_type",
        level="condition_b",
    ).one().index

    subject_shift = (subject_index - (n_subjects - 1) / 2.0) * 0.08
    betas = rng.normal(scale=0.05, size=(n_coefficients, n_voxels))
    betas[a_col, :] = np.linspace(0.4, 1.0, n_voxels) + subject_shift
    betas[b_col, :] = np.linspace(-0.2, 0.3, n_voxels) - subject_shift / 2.0
    return design @ betas + rng.normal(scale=0.15, size=design.shape[0:1] + (n_voxels,))


def run_demo(
    *,
    n_subjects: int = 5,
    max_voxels: int = 32,
) -> SemanticSurvivalResult:
    """Run the group-facing semantic-survival demo."""
    timings: dict[str, float] = {}
    start = time.perf_counter()
    inputs = load_inputs(max_voxels=max_voxels)
    timings["load_inputs_seconds"] = float(time.perf_counter() - start)
    subjects = tuple(f"sub-{idx + 1:02d}" for idx in range(n_subjects))
    stats = []
    explanations: list[dict[str, Any]] = []
    fit_provenance: dict[str, Any] | None = None

    first_level_start = time.perf_counter()
    for subject_index, _subject in enumerate(subjects):
        data = _subject_data(
            inputs,
            subject_index=subject_index,
            n_subjects=n_subjects,
        )
        dataset = fm.fmri_dataset(data, tr=TR, events=inputs.events)
        fit = fm.fmri_lm(inputs.spec, dataset)
        if fit_provenance is None:
            fit_provenance = _fit_provenance_payload(fit)
        contrast = fit.contrast(inputs.omnibus)
        stats.append(np.asarray(contrast.stat, dtype=np.float64))
        explanations.append(contrast.summary())
    timings["first_level_seconds"] = float(time.perf_counter() - first_level_start)

    group_build_start = time.perf_counter()
    group_values = np.stack(stats, axis=1)[:, :, np.newaxis]
    first_explanation = explanations[0]
    contrast_name = inputs.omnibus.display_name
    contrast_data = pd.DataFrame(
        {
            "intent_kind": [first_explanation["intent"]["kind"]],
            "intent_term": [first_explanation["intent"]["term"]],
            "intent_levels": [
                ",".join(first_explanation["intent"]["levels"]),
            ],
            "statistic_family": [first_explanation["statistic"]["family"]],
            "touched_columns": [
                "|".join(first_explanation["touched_columns"]),
            ],
        },
        index=pd.Index([contrast_name], name="contrast"),
    )
    group = group_dataset(
        {"beta": group_values},
        space=SampleLabelSpace([f"v{idx:03d}" for idx in range(group_values.shape[0])]),
        subjects=subjects,
        contrasts=[contrast_name],
        row_data=pd.DataFrame(
            {"feature": [f"v{idx:03d}" for idx in range(group_values.shape[0])]}
        ),
        contrast_data=contrast_data,
        metadata={
            "source_workflow": "tier_a_f_confound_drift",
            "semantic_survival": True,
            "contrast_explanation": first_explanation,
        },
    )
    timings["group_dataset_seconds"] = float(time.perf_counter() - group_build_start)
    group_fit_start = time.perf_counter()
    group_result = ols_voxelwise(group, model=group_model())
    timings["group_fit_seconds"] = float(time.perf_counter() - group_fit_start)

    report = {
        "name": "tier_group_semantic_survival",
        "status": "pass",
        "caveats": [],
        "checks": {
            "typed_intent_kind": first_explanation["intent"]["kind"],
            "typed_intent_term": first_explanation["intent"]["term"],
            "typed_intent_levels": first_explanation["intent"]["levels"],
            "statistic_family": first_explanation["statistic"]["family"],
            "contrast_name_survives": list(group.contrasts),
            "group_assays": group.assay_names(),
            "group_result_assays": group_result.assay_names(),
        },
        "group": {
            "n_subjects": group.n_subjects,
            "n_samples": group.n_samples,
            "contrasts": list(group.contrasts),
            "contrast_data": contrast_data.reset_index().to_dict(orient="records"),
        },
        "timings": {
            "status": "recorded",
            "seconds": float(sum(timings.values())),
            "stages": timings,
        },
        "fit_provenance": fit_provenance,
    }
    return SemanticSurvivalResult(
        group=group,
        group_result=group_result,
        explanations=tuple(explanations),
        report=report,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="Directory where the semantic-survival report should be written.",
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = run_demo()
    report_path = args.out_dir / "semantic_survival_report.json"
    report_path.write_text(json.dumps(result.report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
