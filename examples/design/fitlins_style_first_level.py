"""
FitLins-Style First-Level Model in fmrimod
=========================================

This example mirrors the small first-level model at the center of the
FitLins ds000003 notebook, but keeps the workflow in ordinary Python:
events become an HRF-convolved design, fMRIPrep-style confounds become a
baseline/nuisance model, the GLM is fitted directly, and t/F contrasts are
explicit weights over named design columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pprint import pprint

import numpy as np
import pandas as pd

import fmrimod as fm
from fmrimod.dataset import FmriDataset
from fmrimod.dataset.adapters import NumpyAdapter
from fmrimod.model import FmriModel

FITLINS_STYLE_MODEL = {
    "Name": "synthetic_ds003_first_level",
    "BIDSModelVersion": "1.0.0",
    "Nodes": [
        {
            "Level": "run",
            "Transformations": [
                {"Name": "Factor", "Input": ["trial_type"]},
                {
                    "Name": "Convolve",
                    "Input": ["trial_type.word", "trial_type.pseudoword"],
                    "Model": "spm",
                },
            ],
            "Model": {
                "X": [
                    "trial_type.word",
                    "trial_type.pseudoword",
                    "framewise_displacement",
                    "trans_x",
                    1,
                ]
            },
            "Contrasts": [
                {
                    "Name": "word_gt_pseudoword",
                    "ConditionList": [
                        "trial_type.word",
                        "trial_type.pseudoword",
                    ],
                    "Weights": [1, -1],
                    "Test": "t",
                },
                {
                    "Name": "task_vs_baseline",
                    "ConditionList": [
                        "trial_type.word",
                        "trial_type.pseudoword",
                    ],
                    "Weights": [0.5, 0.5],
                    "Test": "t",
                },
                {
                    "Name": "task_omnibus",
                    "ConditionList": [
                        "trial_type.word",
                        "trial_type.pseudoword",
                    ],
                    "Weights": [[1, 0], [0, 1]],
                    "Test": "F",
                },
            ],
        }
    ],
}


@dataclass
class ExampleResult:
    events: pd.DataFrame
    fitlins_style_model: dict
    fmrimod_model: FmriModel
    fit: object
    column_names: list[str]
    contrast_summary: pd.DataFrame
    comparison: pd.DataFrame


def make_events(n_scans: int = 120, tr: float = 2.0) -> pd.DataFrame:
    """Create a tiny ds000003-like word/pseudoword event table."""
    duration = n_scans * tr
    onsets = np.arange(8.0, duration - 24.0, 12.0)
    trial_type = np.where(np.arange(len(onsets)) % 2 == 0, "word", "pseudoword")
    return pd.DataFrame(
        {
            "run": 1,
            "onset": onsets,
            "duration": 1.2,
            "trial_type": trial_type,
        }
    )


def make_confounds(n_scans: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Create two fMRIPrep-style nuisance regressors for one run."""
    framewise_displacement = np.linspace(-1.0, 1.0, n_scans)
    trans_x = rng.normal(0.0, 0.25, size=n_scans)
    return [np.column_stack([framewise_displacement, trans_x])]


def contrast_vector(
    column_names: list[str],
    weights: dict[str, float],
) -> np.ndarray:
    """Build a contrast vector from readable column names."""
    vector = np.zeros(len(column_names), dtype=float)
    for name, weight in weights.items():
        vector[column_names.index(name)] = weight
    return vector


def build_synthetic_fmrimod_workflow(
    seed: int = 7,
) -> tuple[FmriModel, list[str], pd.DataFrame]:
    """Construct the fmrimod model and synthetic response matrix."""
    rng = np.random.default_rng(seed)
    n_scans = 120
    n_voxels = 32
    tr = 2.0
    sampling_frame = fm.SamplingFrame(blocklens=[n_scans], tr=tr)
    events = make_events(n_scans=n_scans, tr=tr)
    confounds = make_confounds(n_scans=n_scans, rng=rng)

    event_design = fm.event_model(
        "hrf(trial_type)",
        data=events,
        sampling_frame=sampling_frame,
        block="run",
        durations="duration",
    )
    baseline = fm.baseline_model(
        basis="poly",
        degree=2,
        sframe=sampling_frame,
        intercept="runwise",
        nuisance_list=confounds,
    )

    design = np.column_stack([event_design.design_matrix, baseline.design_matrix])
    column_names = event_design.column_names + baseline.column_names

    betas = np.zeros((len(column_names), n_voxels), dtype=float)
    word_idx = column_names.index("trial_type_trial_type.word")
    pseudo_idx = column_names.index("trial_type_trial_type.pseudoword")
    betas[word_idx, : n_voxels // 2] = 0.75
    betas[pseudo_idx, n_voxels // 2 :] = 0.55
    betas[column_names.index("nuis_run1_c1"), :] = rng.normal(0.0, 0.10, n_voxels)
    betas[column_names.index("nuis_run1_c2"), :] = rng.normal(0.0, 0.10, n_voxels)

    signal = design @ betas
    bold = signal + rng.normal(0.0, 0.55, size=signal.shape)
    dataset = FmriDataset(
        NumpyAdapter(bold, sampling_frame),
        event_table=events,
    )
    return FmriModel(event_design, baseline, dataset), column_names, events


def summarize_contrasts(fit: object, column_names: list[str]) -> pd.DataFrame:
    """Compute the two t-contrasts from the FitLins-style model spec."""
    contrasts = {
        "word_gt_pseudoword": {
            "trial_type_trial_type.word": 1.0,
            "trial_type_trial_type.pseudoword": -1.0,
        },
        "task_vs_baseline": {
            "trial_type_trial_type.word": 0.5,
            "trial_type_trial_type.pseudoword": 0.5,
        },
    }

    rows = []
    for name, weights in contrasts.items():
        result = fit.contrast(contrast_vector(column_names, weights), name=name)
        rows.append(
            {
                "contrast": name,
                "test": result.stat_type,
                "mean_estimate": float(np.mean(result.estimate)),
                "max_abs_t": float(np.max(np.abs(result.stat))),
                "min_p": float(np.min(result.p_value)),
            }
        )
    return pd.DataFrame(rows)


def comparison_table() -> pd.DataFrame:
    """Show the FitLins-style contract beside the fmrimod expression."""
    return pd.DataFrame(
        [
            {
                "step": "model declaration",
                "fitlins_style": "BIDS-Stats-Model JSON node",
                "fmrimod": "ordinary Python objects",
            },
            {
                "step": "task regressors",
                "fitlins_style": "Factor + Convolve instructions",
                "fmrimod": "fm.event_model('hrf(trial_type)', data=events, ...)",
            },
            {
                "step": "nuisance regressors",
                "fitlins_style": "confound names inside Model.X",
                "fmrimod": "fm.baseline_model(..., nuisance_list=confounds)",
            },
            {
                "step": "fit",
                "fitlins_style": "fitlins bids_dir out_dir run --model model.json",
                "fmrimod": "fit = fm.fmri_lm(FmriModel(...))",
            },
            {
                "step": "contrasts",
                "fitlins_style": "ConditionList + Weights in JSON",
                "fmrimod": "fit.contrast(vector, name='word_gt_pseudoword')",
            },
        ]
    )


def run_example(seed: int = 7) -> ExampleResult:
    """Run the complete example and return inspectable objects."""
    model, column_names, events = build_synthetic_fmrimod_workflow(seed=seed)
    fit = fm.fmri_lm(model)
    summary = summarize_contrasts(fit, column_names)
    return ExampleResult(
        events=events,
        fitlins_style_model=FITLINS_STYLE_MODEL,
        fmrimod_model=model,
        fit=fit,
        column_names=column_names,
        contrast_summary=summary,
        comparison=comparison_table(),
    )


def main() -> None:
    result = run_example()

    print("FitLins-style model spec:")
    pprint(result.fitlins_style_model)

    print("\nfmrimod design columns:")
    pprint(result.column_names)

    print("\nContrast summary:")
    print(result.contrast_summary.to_string(index=False))

    print("\nWhere the same model lives in fmrimod:")
    print(result.comparison.to_string(index=False))


if __name__ == "__main__":
    main()
