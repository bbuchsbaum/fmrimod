"""Tests for the constrained BIDS Stats Model translator."""

from __future__ import annotations

import numpy as np
import pandas as pd

import fmrimod as fm
from fmrimod.bids import translate_run_node


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
