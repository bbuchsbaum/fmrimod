"""Tests for the constrained BIDS Stats Model translator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.bids import StatsModelContrast, translate_run_node
from fmrimod.spec import Confounds, HrfTerm, Intercept, Spec


def _intent_field(intent: object, key: str) -> object:
    if isinstance(intent, dict):
        return intent[key]
    return getattr(intent, key)


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


def _modulated_stats_model() -> dict:
    model = _stats_model()
    node = model["Nodes"][0]
    node["Transformations"] = [
        {"Name": "Factor", "Input": ["trial_type"]},
        {"Name": "Scale", "Input": ["rt"], "Output": "rt_z"},
        {
            "Name": "Product",
            "Input": ["trial_type.word", "trial_type.pseudoword", "rt_z"],
            "Output": "trial_type_rt_z",
        },
        {
            "Name": "Convolve",
            "Input": ["trial_type.word", "trial_type.pseudoword"],
            "Model": "spm",
        },
    ]
    node["Model"] = {
        "X": [
            "trial_type.word",
            "trial_type.pseudoword",
            "rt_z",
            "framewise_displacement",
            "trans_x",
            1,
        ]
    }
    return model


def _threshold_or_stats_model() -> dict:
    model = _stats_model()
    node = model["Nodes"][0]
    node["Transformations"] = [
        {"Name": "Factor", "Input": ["trial_type"]},
        {"Name": "Scale", "Input": ["rt"], "Output": "rt_z"},
        {
            "Name": "Threshold",
            "Input": ["rt_z"],
            "Output": "fast_rt",
            "Threshold": 0.0,
            "Binarize": True,
            "Above": False,
        },
        {
            "Name": "Or",
            "Input": ["fast_rt", "accuracy_error"],
            "Output": "salient_trial",
        },
        {
            "Name": "Product",
            "Input": ["trial_type.word", "trial_type.pseudoword", "salient_trial"],
            "Output": "trial_type_salient_trial",
        },
        {
            "Name": "Convolve",
            "Input": ["trial_type.word", "trial_type.pseudoword"],
            "Model": "spm",
        },
    ]
    node["Model"] = {
        "X": [
            "trial_type.word",
            "trial_type.pseudoword",
            "salient_trial",
            "framewise_displacement",
            "trans_x",
            1,
        ]
    }
    return model


def _derivative_stats_model(hrf_model: str) -> dict:
    model = _stats_model()
    node = model["Nodes"][0]
    node["Transformations"][1]["Model"] = hrf_model
    return model


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
            "physio": np.cos(np.linspace(0.0, 1.0, n_scans)),
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
    assert t_spec.semantic_spec() is not None
    np.testing.assert_array_equal(
        t_spec.weights,
        translated.contrast_vectors["word_gt_pseudoword"],
    )

    f_spec = translated.contrast_specs["task_omnibus"]
    assert f_spec.test == "F"
    assert f_spec.semantic_spec() is not None
    assert translated.contrast_matrices["task_omnibus"].shape == (
        2,
        len(translated.column_names),
    )
    assert "task_omnibus" not in translated.contrast_vectors


def test_translate_run_node_realises_scale_product_parametric_modulator():
    n_scans = 40
    sampling_frame = fm.SamplingFrame(blocklens=[n_scans], tr=2.0)
    events = pd.DataFrame(
        {
            "run": 1,
            "onset": [4.0, 12.0, 20.0, 28.0],
            "duration": [1.0, 1.0, 1.0, 1.0],
            "trial_type": ["word", "pseudoword", "word", "pseudoword"],
            "rt": [0.70, 1.10, 0.85, 1.25],
        }
    )
    confounds = pd.DataFrame(
        {
            "framewise_displacement": np.linspace(0.0, 1.0, n_scans),
            "trans_x": np.linspace(1.0, 0.0, n_scans),
        }
    )

    translated = translate_run_node(
        _modulated_stats_model(),
        events=events,
        sampling_frame=sampling_frame,
        confounds=confounds,
    )

    assert isinstance(translated.model_spec, Spec)
    hrf_term = translated.model_spec.events[0]
    assert isinstance(hrf_term, HrfTerm)
    assert hrf_term.modulators == ("rt_z",)
    assert "rt_z" in translated.event_table
    assert "rt_z" not in events
    expected_rt_z = (
        events["rt"].to_numpy(dtype=np.float64)
        - float(np.nanmean(events["rt"].to_numpy(dtype=np.float64)))
    ) / float(np.nanstd(events["rt"].to_numpy(dtype=np.float64), ddof=0))
    np.testing.assert_allclose(translated.event_table["rt_z"], expected_rt_z)
    assert len(translated.event_model.column_names) == 4
    assert any("rt_z" in name for name in translated.event_model.column_names)
    assert "word_gt_pseudoword" in translated.contrast_vectors


def test_translate_run_node_realises_threshold_or_parametric_modulator():
    n_scans = 40
    sampling_frame = fm.SamplingFrame(blocklens=[n_scans], tr=2.0)
    events = pd.DataFrame(
        {
            "run": 1,
            "onset": [4.0, 12.0, 20.0, 28.0],
            "duration": [1.0, 1.0, 1.0, 1.0],
            "trial_type": ["word", "pseudoword", "word", "pseudoword"],
            "rt": [0.70, 1.10, 0.85, 1.25],
            "accuracy_error": [0, 1, 0, 0],
        }
    )
    confounds = pd.DataFrame(
        {
            "framewise_displacement": np.linspace(0.0, 1.0, n_scans),
            "trans_x": np.linspace(1.0, 0.0, n_scans),
        }
    )

    translated = translate_run_node(
        _threshold_or_stats_model(),
        events=events,
        sampling_frame=sampling_frame,
        confounds=confounds,
    )

    hrf_term = translated.model_spec.events[0]
    assert isinstance(hrf_term, HrfTerm)
    assert hrf_term.modulators == ("salient_trial",)
    assert "fast_rt" not in events
    assert "salient_trial" not in events
    expected_rt_z = (
        events["rt"].to_numpy(dtype=np.float64)
        - float(np.nanmean(events["rt"].to_numpy(dtype=np.float64)))
    ) / float(np.nanstd(events["rt"].to_numpy(dtype=np.float64), ddof=0))
    expected_fast_rt = (expected_rt_z <= 0.0).astype(float)
    expected_salient = np.logical_or(
        expected_fast_rt.astype(bool),
        events["accuracy_error"].astype(bool).to_numpy(),
    ).astype(int)

    np.testing.assert_allclose(translated.event_table["rt_z"], expected_rt_z)
    np.testing.assert_array_equal(translated.event_table["fast_rt"], expected_fast_rt)
    np.testing.assert_array_equal(
        translated.event_table["salient_trial"],
        expected_salient,
    )
    assert any(
        "salient_trial" in name
        for name in translated.event_model.column_names
    )


@pytest.mark.parametrize(
    ("model_label", "basis", "n_basis"),
    [
        ("spm + derivative", "spmg2", 2),
        ("spm + derivative + dispersion", "spmg3", 3),
    ],
)
def test_translate_run_node_normalizes_convolve_derivative_hrf_models(
    model_label: str,
    basis: str,
    n_basis: int,
):
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
            "trans_x": np.sin(np.linspace(0.0, 1.0, n_scans)),
        }
    )

    translated = translate_run_node(
        _derivative_stats_model(model_label),
        events=events,
        sampling_frame=sampling_frame,
        confounds=confounds,
    )

    hrf_term = translated.model_spec.events[0]
    assert isinstance(hrf_term, HrfTerm)
    assert hrf_term.hrf == basis
    assert len(translated.event_model.column_names) == 2 * n_basis
    assert any(name.endswith("_b01") for name in translated.event_model.column_names)
    assert any(
        name.endswith(f"_b0{n_basis}")
        for name in translated.event_model.column_names
    )

    contrast = translated.contrast_vectors["word_gt_pseudoword"]
    nonzero = np.flatnonzero(contrast)
    assert len(nonzero) == 2
    assert translated.column_names[int(nonzero[0])].endswith("pseudoword_b01")
    assert translated.column_names[int(nonzero[1])].endswith("word_b01")
    assert contrast[int(nonzero[0])] == -1.0
    assert contrast[int(nonzero[1])] == 1.0


def test_derivative_bids_contrast_uses_explicit_vector_bridge_fallback():
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
            "trans_x": np.sin(np.linspace(0.0, 1.0, n_scans)),
        }
    )
    translated = translate_run_node(
        _derivative_stats_model("spm + derivative"),
        events=events,
        sampling_frame=sampling_frame,
        confounds=confounds,
    )
    legacy_design = np.column_stack(
        [
            translated.event_model.design_matrix,
            translated.baseline_model.design_matrix,
        ]
    )
    beta = np.zeros((legacy_design.shape[1], 2), dtype=np.float64)
    beta[translated.column_names.index("trial_type_trial_type.word_b01"), :] = 0.6
    beta[
        translated.column_names.index("trial_type_trial_type.pseudoword_b01"),
        :,
    ] = -0.1
    rng = np.random.default_rng(41)
    bold = legacy_design @ beta + rng.normal(0.0, 0.03, size=(n_scans, 2))

    fit = translated.fit(fm.fmri_dataset(bold, tr=2.0, events=events))
    result = translated.contrast(fit, "word_gt_pseudoword")

    assert result.stat_type == "t"
    assert _intent_field(result.intent, "kind") == "bids_vector_bridge"


def test_translate_run_node_rejects_unknown_convolve_hrf_model():
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
            "trans_x": np.sin(np.linspace(0.0, 1.0, n_scans)),
        }
    )

    with pytest.raises(NotImplementedError, match="Unsupported Convolve HRF model"):
        translate_run_node(
            _derivative_stats_model("fir"),
            events=events,
            sampling_frame=sampling_frame,
            confounds=confounds,
        )


def test_modulated_bids_translation_fit_uses_transformed_event_table():
    n_scans = 40
    sampling_frame = fm.SamplingFrame(blocklens=[n_scans], tr=2.0)
    events = pd.DataFrame(
        {
            "run": 1,
            "onset": [4.0, 12.0, 20.0, 28.0],
            "duration": [1.0, 1.0, 1.0, 1.0],
            "trial_type": ["word", "pseudoword", "word", "pseudoword"],
            "rt": [0.70, 1.10, 0.85, 1.25],
        }
    )
    confounds = pd.DataFrame(
        {
            "framewise_displacement": np.linspace(0.0, 1.0, n_scans),
            "trans_x": np.sin(np.linspace(0.0, 1.0, n_scans)),
            "physio": np.cos(np.linspace(0.0, 1.0, n_scans)),
        }
    )
    translated = translate_run_node(
        _modulated_stats_model(),
        events=events,
        sampling_frame=sampling_frame,
        confounds=confounds,
    )
    legacy_design = np.column_stack(
        [
            translated.event_model.design_matrix,
            translated.baseline_model.design_matrix,
        ]
    )
    beta = np.zeros((legacy_design.shape[1], 3), dtype=np.float64)
    beta[translated.column_names.index("trial_type_trial_type.word"), :] = 0.6
    beta[translated.column_names.index("trial_type_trial_type.pseudoword"), :] = -0.1
    modulated_word_ix = next(
        idx
        for idx, name in enumerate(translated.column_names)
        if "rt_z" in name
    )
    beta[modulated_word_ix, :] = 0.3
    rng = np.random.default_rng(23)
    bold = legacy_design @ beta + rng.normal(0.0, 0.03, size=(n_scans, 3))

    dataset = fm.fmri_dataset(bold, tr=2.0, events=events)
    fit = translated.fit(dataset)

    np.testing.assert_allclose(
        fit.model.design_matrix_array(),
        legacy_design,
        rtol=1e-12,
        atol=1e-12,
    )
    result = translated.contrast(fit, "word_gt_pseudoword")
    assert result.stat_type == "t"
    assert _intent_field(result.intent, "kind") == "semantic_linear_contrast"


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


def test_translated_bids_artifacts_drive_public_lm_contrast_path():
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
            "trans_x": np.sin(np.linspace(0.0, 1.0, n_scans)),
        }
    )
    translated = translate_run_node(
        _stats_model(),
        events=events,
        sampling_frame=sampling_frame,
        confounds=confounds,
    )
    legacy_design = np.column_stack(
        [
            translated.event_model.design_matrix,
            translated.baseline_model.design_matrix,
        ]
    )
    beta = np.zeros((legacy_design.shape[1], 4), dtype=np.float64)
    beta[translated.column_names.index("trial_type_trial_type.word"), :] = 0.8
    beta[translated.column_names.index("trial_type_trial_type.pseudoword"), :] = -0.2
    rng = np.random.default_rng(17)
    bold = legacy_design @ beta + rng.normal(0.0, 0.05, size=(n_scans, 4))

    dataset = fm.fmri_dataset(bold, tr=2.0, events=events)
    fit = translated.fit(dataset)

    np.testing.assert_allclose(
        fit.model.design_matrix_array(),
        legacy_design,
        rtol=1e-12,
        atol=1e-12,
    )
    results = translated.compute_contrasts(fit)

    assert set(results) == {
        "word_gt_pseudoword",
        "task_omnibus",
    }
    assert results["word_gt_pseudoword"].stat_type == "t"
    assert _intent_field(
        results["word_gt_pseudoword"].intent,
        "kind",
    ) == "semantic_linear_contrast"
    assert _intent_field(results["word_gt_pseudoword"].intent, "term") == "trial_type"
    assert tuple(
        _intent_field(results["word_gt_pseudoword"].intent, "levels")
    ) == ("word", "pseudoword")
    assert results["task_omnibus"].stat_type == "F"
    assert _intent_field(results["task_omnibus"].intent, "kind") == "omnibus"

    manual_t = fit.contrast(
        translated.contrast_vectors["word_gt_pseudoword"],
        name="manual_word_gt_pseudoword",
    )
    manual_f = fit.contrast(
        translated.contrast_matrices["task_omnibus"],
        name="manual_task_omnibus",
    )
    np.testing.assert_allclose(results["word_gt_pseudoword"].stat, manual_t.stat)
    np.testing.assert_allclose(results["task_omnibus"].stat, manual_f.stat)


def test_scale_product_modulator_fits_through_public_seam_with_transformed_events():
    n_scans = 40
    sampling_frame = fm.SamplingFrame(blocklens=[n_scans], tr=2.0)
    events = pd.DataFrame(
        {
            "run": 1,
            "onset": [4.0, 12.0, 20.0, 28.0],
            "duration": [1.0, 1.0, 1.0, 1.0],
            "trial_type": ["word", "pseudoword", "word", "pseudoword"],
            "rt": [0.5, 0.7, 0.6, 1.2],
        }
    )
    confounds = pd.DataFrame(
        {
            "framewise_displacement": np.linspace(0.0, 1.0, n_scans),
            "trans_x": np.sin(np.linspace(0.0, 1.0, n_scans)),
        }
    )

    translated = translate_run_node(
        _modulated_stats_model(),
        events=events,
        sampling_frame=sampling_frame,
        confounds=confounds,
    )

    hrf_term = translated.model_spec.events[0]
    assert hrf_term.modulators == ("rt_z",)
    assert "rt_z" not in events.columns
    assert "rt_z" in translated.event_table.columns
    assert any(
        isinstance(term, Confounds)
        and term.columns == ("framewise_displacement", "trans_x")
        for term in translated.model_spec.baseline
    )

    legacy_design = np.column_stack(
        [
            translated.event_model.design_matrix,
            translated.baseline_model.design_matrix,
        ]
    )
    beta = np.zeros((legacy_design.shape[1], 3), dtype=np.float64)
    beta[0, :] = -0.25
    beta[1, :] = 0.7
    beta[2, :] = 0.2
    rng = np.random.default_rng(23)
    bold = legacy_design @ beta + rng.normal(0.0, 0.05, size=(n_scans, 3))
    dataset = fm.fmri_dataset(bold, tr=2.0, events=events)

    fit = translated.fit(dataset)
    np.testing.assert_allclose(
        fit.model.design_matrix_array(),
        legacy_design,
        rtol=1e-12,
        atol=1e-12,
    )
    result = translated.contrast(fit, "word_gt_pseudoword")

    assert _intent_field(result.intent, "kind") == "semantic_linear_contrast"
    assert result.stat_type == "t"
