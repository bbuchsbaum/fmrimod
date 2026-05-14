"""Executable contracts for the fmridesign design/formula typed seam."""

import numpy as np
import pandas as pd
import pytest

from fmrimod.baseline import (
    CleanedNuisance,
    NuisanceCheck,
    baseline_model,
    check_nuisance,
    clean_nuisance,
)
from fmrimod.design.event_model import event_model
from fmrimod.formula import dsl_hrf, event, hrf, parse_formula, term
from fmrimod.sampling import SamplingFrame


def _single_condition_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "onset": [2.0, 10.0, 18.0],
            "condition": ["A", "A", "A"],
            "duration": [1.0, 1.0, 1.0],
        }
    )


def _mixed_timing_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "onset": [40.0, 45.0, 50.0, 55.0],
            "stim_onset": [2.0, 6.0, 10.0, 14.0],
            "term_onset": [3.0, 7.0, 11.0, 15.0],
            "condition": ["A", "B", "A", "B"],
            "duration": [1.0, 1.0, 1.0, 1.0],
            "stim_duration": [0.5, 1.5, 0.75, 2.0],
            "run": ["run_b", "run_a", "run_b", "run_a"],
        }
    )


def _bad_nuisance_block() -> pd.DataFrame:
    dvars = np.arange(1, 7, dtype=float)
    return pd.DataFrame(
        {
            "dvars": dvars,
            "std_dvars": 10.0 * dvars,
            "zero_col": np.zeros_like(dvars),
        }
    )


def _two_block_nuisance() -> list[pd.DataFrame]:
    return [
        _bad_nuisance_block(),
        pd.DataFrame(
            {
                "motion_x": [-2, -1, 0, 1, 2, 3],
                "motion_y": [1, -1, 1, -1, 1, -1],
            }
        ),
    ]


def test_parse_formula_event_mode_preserves_hrf_options():
    terms = parse_formula(
        (
            "onset ~ hrf(condition, basis='spmg1', normalize=True, "
            "summate=False, id='stim_term', prefix='stim', lag=1.5, "
            "nbasis=3, onsets='stim_onset', durations='duration', "
            "subset='condition == \"A\"')"
        ),
        for_event_model=True,
    )

    assert len(terms) == 1
    parsed = terms[0]
    assert parsed.events == ["condition"]
    assert parsed.hrf == "spmg1"
    assert parsed.name == "stim_term"
    assert parsed.normalize is True
    assert parsed.summate is False
    assert parsed.kwargs["prefix"] == "stim"
    assert parsed.kwargs["lag"] == 1.5
    assert parsed.kwargs["nbasis"] == 3
    assert parsed.kwargs["onsets"] == "stim_onset"
    assert parsed.kwargs["durations"] == "duration"
    assert parsed.kwargs["subset"] == 'condition == "A"'


def test_functional_hrf_options_are_hoisted_for_event_model():
    df = _single_condition_events()

    model = event_model(
        [term("condition") | hrf("spmg1", normalize=True, summate=False)],
        data=df,
        tr=1.0,
        n_scans=40,
    )

    assert model.terms[0].normalize is True
    assert model.terms[0].summate is False
    assert np.max(np.abs(model.design_matrix)) == pytest.approx(1.0)


def test_formula_lhs_sets_event_model_onset_column():
    df = _mixed_timing_events()

    model = event_model(
        "stim_onset ~ hrf(condition, basis='spmg1')",
        data=df,
        tr=1.0,
        n_scans=30,
    )

    np.testing.assert_allclose(model.events["condition"].onsets, df["stim_onset"])
    assert not np.array_equal(model.events["condition"].onsets, df["onset"])
    assert np.max(np.abs(model.design_matrix)) > 0


def test_hrf_term_subset_and_timing_options_create_term_local_event():
    df = _mixed_timing_events()

    model = event_model(
        (
            "onset ~ hrf(condition, basis='spmg1', "
            "subset='condition == \"A\"', onsets='term_onset', "
            "durations='stim_duration')"
        ),
        data=df,
        tr=1.0,
        n_scans=30,
    )

    assert model.terms[0].events == ["condition"]
    event_key = model.terms[0]._event_overrides[0]
    timed_event = model.events[event_key]
    assert event_key != "condition"
    assert timed_event.name == "condition"
    assert list(np.asarray(timed_event.values).astype(str)) == ["A", "A"]
    np.testing.assert_allclose(timed_event.onsets, [3.0, 11.0])
    np.testing.assert_allclose(timed_event.durations, [0.5, 0.75])
    assert np.max(np.abs(model.design_matrix)) > 0


def test_functional_hrf_timing_options_create_term_local_event():
    df = _mixed_timing_events()

    model = event_model(
        [
            term("condition")
            | hrf(
                "spmg1",
                subset='condition == "A"',
                onsets="term_onset",
                durations="stim_duration",
            )
        ],
        data=df,
        tr=1.0,
        n_scans=30,
    )

    assert model.terms[0].events == ["condition"]
    event_key = model.terms[0]._event_overrides[0]
    timed_event = model.events[event_key]
    np.testing.assert_allclose(timed_event.onsets, [3.0, 11.0])
    np.testing.assert_allclose(timed_event.durations, [0.5, 0.75])


def test_term_subset_accepts_mapping_selectors_from_typed_spec_lowering():
    df = _mixed_timing_events()

    model = event_model(
        [term("condition") | hrf("spmg1", subset={"condition": "A"})],
        data=df,
        tr=1.0,
        n_scans=30,
    )

    assert model.terms[0].events == ["condition"]
    event_key = model.terms[0]._event_overrides[0]
    timed_event = model.events[event_key]
    assert list(np.asarray(timed_event.values).astype(str)) == ["A", "A"]
    np.testing.assert_allclose(timed_event.onsets, [40.0, 50.0])


def test_block_ids_preserve_first_appearance_order():
    df = _mixed_timing_events()

    model = event_model(
        "condition",
        data=df,
        block="run",
        tr=1.0,
        n_scans=30,
    )

    assert model.blockids.tolist() == [1, 2, 1, 2]


def test_block_ids_validate_length_against_data_rows():
    df = _mixed_timing_events()

    with pytest.raises(ValueError, match="Block vector length"):
        event_model(
            "condition",
            data=df,
            block=["run_b", "run_a"],
            tr=1.0,
            n_scans=30,
        )


def test_dsl_spm_canonical_alias_is_resolvable_by_event_model():
    df = _single_condition_events()
    condition = event("condition")

    model = event_model(
        [condition @ dsl_hrf.spm_canonical],
        data=df,
        tr=1.0,
        n_scans=40,
    )

    assert model.terms[0].hrf == "spm_canonical"
    assert model.design_matrix.shape == (40, 1)
    assert np.max(np.abs(model.design_matrix)) > 0


def test_check_nuisance_reports_zero_variance_duplicate_and_aliasing():
    sframe = SamplingFrame(blocklens=[6, 6], TR=1.0)

    report = check_nuisance(_two_block_nuisance(), sframe, basis="constant")

    assert isinstance(report, NuisanceCheck)
    assert report.ok is False
    assert {"zero_variance", "duplicate", "rank_deficient_with_baseline"}.issubset(
        set(report.problems["issue"])
    )
    assert report.by_block[0].zero_variance == ("zero_col",)
    assert report.by_block[0].aliased_columns == ("std_dvars",)
    assert report.by_block[0].duplicate_pairs[0].column == "std_dvars"
    assert report.by_block[0].duplicate_pairs[0].duplicates == "dvars"


def test_baseline_model_default_warns_on_nuisance_rank_problems():
    sframe = SamplingFrame(blocklens=[6], TR=1.0)

    with pytest.warns(UserWarning, match="Zero-variance columns: zero_col"):
        model = baseline_model(
            basis="constant",
            sframe=sframe,
            nuisance_list=[_bad_nuisance_block()],
        )

    assert isinstance(model.nuisance_check, NuisanceCheck)
    assert model.nuisance_check.ok is False


def test_baseline_model_can_error_on_nuisance_rank_problems():
    sframe = SamplingFrame(blocklens=[6], TR=1.0)
    nuisance = [_bad_nuisance_block().drop(columns=["zero_col"])]

    with pytest.raises(ValueError, match="Duplicate or near-duplicate columns"):
        baseline_model(
            basis="constant",
            sframe=sframe,
            nuisance_list=nuisance,
            nuisance_check="error",
        )


def test_baseline_model_can_drop_rank_useless_nuisance_columns():
    sframe = SamplingFrame(blocklens=[6, 6], TR=1.0)

    with pytest.warns(
        UserWarning,
        match="Dropped non-finite, zero-variance, and rank-aliased nuisance columns",
    ):
        model = baseline_model(
            basis="constant",
            sframe=sframe,
            nuisance_list=_two_block_nuisance(),
            nuisance_check="drop",
        )

    nuisance_term = model.terms["nuisance"]
    assert nuisance_term.design_matrix.shape == (12, 3)
    assert np.linalg.matrix_rank(model.design_matrix) == model.design_matrix.shape[1]


def test_clean_nuisance_returns_cleaned_matrices_and_audit_report():
    sframe = SamplingFrame(blocklens=[6], TR=1.0)

    cleaned = clean_nuisance(
        [_bad_nuisance_block()],
        sframe,
        basis="constant",
    )

    assert isinstance(cleaned, CleanedNuisance)
    assert isinstance(cleaned.report, NuisanceCheck)
    assert list(cleaned.nuisance_list[0].columns) == ["dvars"]
