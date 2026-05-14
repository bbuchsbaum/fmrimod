"""Tests for the constrained BIDS Stats Model translator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.bids import StatsModelContrast, translate_run_node
from fmrimod.spec import Confounds, HrfTerm, Intercept, Spec


def _stats_model() -> dict:
    return {
        "Name": "synthetic_fitlins_style",
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
                        "Name": "task_omnibus",
                        "ConditionList": [
                            "trial_type.word",
                            "trial_type.pseudoword",
                        ],
                        "Weights": [[1, 0], [0, 1]],
                        "Test": "F",
                    }
                ],
            }
        ],
    }


def test_translate_run_node_builds_design_and_contrasts():
    n_scans = 40
    sampling_frame = fm.SamplingFrame(blocklens=[n_scans], tr=2.0)
    events = pd.DataFrame(
        {
            "run": 1,
            "onset": [4.0, 12.0, 20.0, 28.0],
            "duration": [1.0, 1.0, 1.0, 1.0],
            "trial_type": ["word", "pseudoword", "word", "pseudoword"],
        }
    )
    confounds = pd.DataFrame(
        {
            "framewise_displacement": np.linspace(0.0, 1.0, n_scans),
            "trans_x": np.linspace(1.0, 0.0, n_scans),
        }
    )

    translated = translate_run_node(
        _stats_model(),
        events=events,
        sampling_frame=sampling_frame,
        confounds=confounds,
    )

    design = np.column_stack(
        [
            translated.event_model.design_matrix,
            translated.baseline_model.design_matrix,
        ]
    )
    assert design.shape[0] == n_scans
    assert "word_gt_pseudoword" in translated.contrast_vectors
    assert translated.contrast_vectors["word_gt_pseudoword"].shape == (
        len(translated.column_names),
    )
    assert any(name.endswith(".word") for name in translated.column_names)
    assert "nuis_run1_c1" in translated.column_names


def test_translate_run_node_exposes_typed_model_and_contrast_artifacts():
    n_scans = 40
    sampling_frame = fm.SamplingFrame(blocklens=[n_scans], tr=2.0)
    events = pd.DataFrame(
        {
            "run": 1,
            "onset": [4.0, 12.0, 20.0, 28.0],
            "duration": [1.0, 1.0, 1.0, 1.0],
            "trial_type": ["word", "pseudoword", "word", "pseudoword"],
        }
    )
    confounds = pd.DataFrame(
        {
            "framewise_displacement": np.linspace(0.0, 1.0, n_scans),
            "trans_x": np.linspace(1.0, 0.0, n_scans),
        }
    )

    translated = translate_run_node(
        _stats_model(),
        events=events,
        sampling_frame=sampling_frame,
        confounds=confounds,
    )

    assert isinstance(translated.model_spec, Spec)
    assert len(translated.model_spec.events) == 1
    hrf_term = translated.model_spec.events[0]
    assert isinstance(hrf_term, HrfTerm)
    assert hrf_term.variables == ("trial_type",)
    assert hrf_term.hrf == "spm"
    assert hrf_term.durations == "duration"
    assert any(
        isinstance(term, Intercept) and term.per == "global"
        for term in translated.model_spec.baseline
    )
    assert any(
        isinstance(term, Confounds)
        and term.columns == ("framewise_displacement", "trans_x")
        for term in translated.model_spec.baseline
    )

    t_spec = translated.contrast_specs["word_gt_pseudoword"]
    assert isinstance(t_spec, StatsModelContrast)
    assert t_spec.test == "t"
    assert t_spec.conditions == ("trial_type.word", "trial_type.pseudoword")
    np.testing.assert_array_equal(
        t_spec.weights,
        translated.contrast_vectors["word_gt_pseudoword"],
    )

    f_spec = translated.contrast_specs["task_omnibus"]
    assert f_spec.test == "F"
    assert translated.contrast_matrices["task_omnibus"].shape == (
        2,
        len(translated.column_names),
    )
    assert "task_omnibus" not in translated.contrast_vectors


def test_translate_run_node_rejects_ambiguous_flat_f_weights():
    model = _stats_model()
    model["Nodes"][0]["Contrasts"] = [
        {
            "Name": "ambiguous_f",
            "ConditionList": [
                "trial_type.word",
                "trial_type.pseudoword",
            ],
            "Weights": [1, -1],
            "Test": "F",
        }
    ]
    n_scans = 40
    sampling_frame = fm.SamplingFrame(blocklens=[n_scans], tr=2.0)
    events = pd.DataFrame(
        {
            "run": 1,
            "onset": [4.0, 12.0, 20.0, 28.0],
            "duration": [1.0, 1.0, 1.0, 1.0],
            "trial_type": ["word", "pseudoword", "word", "pseudoword"],
        }
    )
    confounds = pd.DataFrame(
        {
            "framewise_displacement": np.linspace(0.0, 1.0, n_scans),
            "trans_x": np.linspace(1.0, 0.0, n_scans),
        }
    )

    with pytest.raises(ValueError, match="F contrast Weights must be a 2-D matrix"):
        translate_run_node(
            model,
            events=events,
            sampling_frame=sampling_frame,
            confounds=confounds,
        )
