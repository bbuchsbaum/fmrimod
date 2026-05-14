"""Executable artifact for ``docs/contracts/parametric_contrast_sugar_v1.md``."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


def _fit_with_modulator(seed: int = 14):
    import fmrimod as fm
    from fmrimod.spec import hrf

    rng = np.random.default_rng(seed)
    n_scans = 72
    events = pd.DataFrame(
        {
            "onset": [6.0, 14.0, 24.0, 34.0, 44.0, 54.0],
            "duration": np.zeros(6),
            "trial_type": ["word", "pseudoword"] * 3,
            "rt_z": [-0.8, -0.4, 0.0, 0.2, 0.6, 0.9],
        }
    )
    dataset = fm.fmri_dataset(rng.standard_normal((n_scans, 5)), tr=2.0, events=events)
    return fm.fmri_lm(hrf("trial_type", modulators=["rt_z"]), dataset)


def test_modulator_v1_call_shape_compiles() -> None:
    """v1 t-contrast spelling: modulator(...).within(...).slope(level)."""

    from fmrimod.contrast import modulator

    rt = modulator("rt_z").within("trial_type")
    spec = rt.slope("word") - rt.slope("pseudoword")
    assert spec is not None


def test_modulator_rejects_missing_modulator_with_actionable_message() -> None:
    """Error policy from Q3: headline names the missing identifier."""

    from fmrimod.contrast import DesignProvenanceError, modulator

    fit = _fit_with_modulator()
    spec = modulator("not_a_real_column").within("trial_type").slope("word")
    with pytest.raises(DesignProvenanceError) as excinfo:
        fit.contrast(spec)
    message = str(excinfo.value)
    assert "not_a_real_column" in message
    assert "Available modulators" in message


def test_modulator_f_contrast_v1_raises_with_contract_pointer() -> None:
    """v1 ships t-only; F-contrast deferred to v2 with a clear error."""

    from fmrimod.contrast import modulator

    rt = modulator("rt_z").within("trial_type")
    with pytest.raises(NotImplementedError):
        rt.slopes("word", "pseudoword", "neutral")
    with pytest.raises(NotImplementedError):
        rt.omnibus("word", "pseudoword", "neutral")


def test_group_dataset_from_contrasts_v1_call_shape_compiles() -> None:
    """v1 single-contrast-per-subject dict constructor."""

    from fmrimod.contrast import group_dataset_from_contrasts

    assert callable(group_dataset_from_contrasts)


def test_group_dataset_from_contrasts_preserves_provenance_fields() -> None:
    """Q5 metadata contract: required ContrastResult fields survive lowering."""

    from fmrimod.contrast import group_dataset_from_contrasts, modulator

    fit_a = _fit_with_modulator(seed=1)
    fit_b = _fit_with_modulator(seed=2)
    rt = modulator("rt_z").within("trial_type")
    spec = rt.slope("word") - rt.slope("pseudoword")
    a = fit_a.contrast(spec, name="word_rt_slope_gt_pseudoword_rt_slope")
    b = fit_b.contrast(spec, name="word_rt_slope_gt_pseudoword_rt_slope")

    group = group_dataset_from_contrasts(
        {"sub-01": a, "sub-02": b},
        covariates=pd.DataFrame(
            {"subject": ["sub-02", "sub-01"], "behavior": [0.5, -0.5]}
        ),
    )

    assert group.metadata["source_format"] == "contrast_results"
    assert group.subjects == ("sub-01", "sub-02")
    assert group.contrasts == ("word_rt_slope_gt_pseudoword_rt_slope",)
    assert group.col_data is not None
    assert group.col_data["behavior"].tolist() == [-0.5, 0.5]
    np.testing.assert_allclose(group.assay("beta")[:, 0, 0], a.estimate)
    np.testing.assert_allclose(group.assay("beta")[:, 1, 0], b.estimate)
    np.testing.assert_allclose(group.assay("se")[:, 0, 0], a.se)
    assert group.contrast_data is not None
    row = group.contrast_data.loc["word_rt_slope_gt_pseudoword_rt_slope"]
    payload = json.loads(str(row["contrast_intent"]))
    assert payload["kind"] in {"semantic_contrast", "semantic_linear_contrast"}
    assert json.loads(str(row["touched_columns"])) == list(a.touched_columns)
    assert json.loads(str(row["touched_column_details"])) == list(
        a.touched_column_details
    )


def test_low_level_condition_escape_hatch_remains_valid() -> None:
    """The sugar must be additive: condition(..., term=...) still works."""

    from fmrimod.contrast import condition

    word = condition("word", term="trial_type:rt_z")
    pseudoword = condition("pseudoword", term="trial_type:rt_z")
    spec = word - pseudoword
    assert spec is not None


def test_modulator_sugar_matches_low_level_condition_weights() -> None:
    """The v1 sugar must lower to the same columns as condition(term=...)."""

    from fmrimod.contrast import condition, modulator

    fit = _fit_with_modulator()
    rt = modulator("rt_z").within("trial_type")
    sugar = rt.slope("word") - rt.slope("pseudoword")
    low_level = condition("word", term="trial_type:rt_z") - condition(
        "pseudoword",
        term="trial_type:rt_z",
    )

    np.testing.assert_allclose(
        sugar.resolve(fit.design_columns()),
        low_level.resolve(fit.design_columns()),
    )
    sugar_result = fit.contrast(sugar, name="sugar")
    low_level_result = fit.contrast(low_level, name="low_level")
    np.testing.assert_allclose(sugar_result.estimate, low_level_result.estimate)
    assert sugar_result.touched_columns == low_level_result.touched_columns
