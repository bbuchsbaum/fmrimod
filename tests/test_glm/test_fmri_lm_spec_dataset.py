"""Tests for the canonical ``fmri_lm(spec, dataset, ...)`` entry point.

The legacy ``fmri_lm(model, config)`` signature is exercised in
``test_fmri_lm_api.py``; this file covers the spec+dataset realignment and the
dispatch boundary between the two forms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.contrast import condition
from fmrimod.glm.engine import ChunkwiseEngineOptions
from fmrimod.glm.fmri_lm import FmriLm, _is_fmri_model_like, fmri_lm
from fmrimod.model.config import FmriLmConfig
from fmrimod.spec import covariate


@pytest.fixture
def synthetic_run():
    """Single-run synthetic block design with a known listening contrast."""
    rng = np.random.default_rng(2024)
    tr = 2.0
    n_scans = 60
    duration = 12.0

    onsets = np.arange(0.0, n_scans * tr, duration * 2)
    n_events = len(onsets)
    half = n_events // 2
    trial_types = np.array(
        ["listening"] * half + ["rest"] * (n_events - half),
        dtype=object,
    )
    rng.shuffle(trial_types)
    events = pd.DataFrame(
        {
            "onset": onsets,
            "trial_type": trial_types,
            "duration": np.full(n_events, duration),
            "run": np.ones(n_events, dtype=int),
        }
    )

    Y = rng.normal(loc=100.0, scale=1.0, size=(n_scans, 8)).astype(np.float64)
    return events, Y, tr


def test_fmri_lm_canonical_spec_dataset_call(synthetic_run):
    """fmri_lm(spec, dataset) builds an FmriModel and fits it end-to-end."""
    events, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr, events=events)

    fit = fmri_lm("hrf(trial_type)", ds)

    assert isinstance(fit, FmriLm)
    # Event model contributes 2 columns (listening, rest) + 1 runwise intercept.
    assert fit.n_coefficients >= 2
    assert fit.n_voxels == Y.shape[1]
    assert fit.betas.shape == (fit.n_coefficients, fit.n_voxels)


def test_fmri_lm_spec_dataset_accepts_authored_semantic_contrast(synthetic_run):
    """Authored condition intent resolves through the public dataset seam."""
    events, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr, events=events)

    fit = fmri_lm("hrf(trial_type)", ds)
    semantic = fit.contrast(
        condition("listening", term="trial_type")
        - condition("rest", term="trial_type"),
    )
    columns = semantic.explain().to_dict()["design_columns"]

    assert semantic.intent["kind"] == "semantic_contrast"
    assert semantic.touched_columns == (
        "trial_type_trial_type.listening",
        "trial_type_trial_type.rest",
    )
    assert [column["level"] for column in columns] == ["listening", "rest"]
    assert {column["provenance"]["level"] for column in columns} == {"declared"}
    assert semantic.estimate.shape == (Y.shape[1],)


def test_fmri_lm_semantic_contrast_survives_categorical_ordering() -> None:
    """Semantic contrasts follow event levels, not realized design positions."""

    rng = np.random.default_rng(20260513)
    tr = 2.0
    n_scans = 84
    labels = np.array(["gain", "loss"] * 5, dtype=object)
    y = rng.normal(size=(n_scans, 5)).astype(np.float64)

    def fit_with_order(order: tuple[str, str]) -> FmriLm:
        events = pd.DataFrame(
            {
                "onset": np.arange(labels.size, dtype=np.float64) * 12.0,
                "duration": np.full(labels.size, 6.0),
                "trial_type": pd.Categorical(
                    labels,
                    categories=list(order),
                    ordered=True,
                ),
                "run": np.ones(labels.size, dtype=int),
            }
        )
        dataset = fm.fmri_dataset(y, tr=tr, events=events)
        return fmri_lm("hrf(trial_type)", dataset)

    canonical = fit_with_order(("gain", "loss"))
    reversed_levels = fit_with_order(("loss", "gain"))
    contrast = condition("gain", term="trial_type") - condition(
        "loss",
        term="trial_type",
    )

    canonical_names = canonical.design_columns().names
    reversed_names = reversed_levels.design_columns().names
    canonical_result = canonical.contrast(contrast)
    reversed_result = reversed_levels.contrast(contrast)

    assert canonical_names != reversed_names
    assert canonical_result.touched_columns == (
        "trial_type_trial_type.gain",
        "trial_type_trial_type.loss",
    )
    assert reversed_result.touched_columns == (
        "trial_type_trial_type.loss",
        "trial_type_trial_type.gain",
    )
    assert np.allclose(canonical_result.estimate, reversed_result.estimate, atol=1e-10)
    assert np.allclose(canonical_result.stat, reversed_result.stat, atol=1e-7)
    assert canonical_result.intent["kind"] == "semantic_contrast"
    assert canonical_result.intent["positive"]["level"] == "gain"
    assert canonical_result.intent["negative"]["level"] == "loss"
    assert reversed_result.intent["kind"] == "semantic_contrast"
    assert reversed_result.intent["positive"]["level"] == "gain"
    assert reversed_result.intent["negative"]["level"] == "loss"


def test_fmri_lm_hrf_formula_routes_through_typed_spec(synthetic_run, monkeypatch):
    """Convertible HRF formulas should reach event_model as typed terms."""
    events, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr, events=events)

    from fmrimod.design import event_model as event_model_module

    original_event_model = event_model_module.event_model

    def guard_no_string_formula(formula, *args, **kwargs):
        if isinstance(formula, str):
            raise AssertionError("fmri_lm should adapt HRF formulas to typed Spec")
        return original_event_model(formula, *args, **kwargs)

    monkeypatch.setattr(event_model_module, "event_model", guard_no_string_formula)

    fit = fmri_lm(
        "hrf(trial_type, basis='spmg1', normalize=True, summate=False)",
        ds,
    )

    assert isinstance(fit, FmriLm)
    term = fit.model.event_model.terms[0]
    assert term.normalize is True
    assert term.summate is False


def test_fmri_lm_functional_formula_routes_through_typed_spec(synthetic_run):
    """Functional/list HRF formulas should adapt into the typed Spec path."""
    events, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr, events=events)

    from fmrimod.formula import hrf as formula_hrf
    from fmrimod.formula import term as formula_term

    fit = fmri_lm(
        [
            formula_term("trial_type")
            | formula_hrf("spmg1", normalize=True, summate=False)
        ],
        ds,
    )

    assert isinstance(fit, FmriLm)
    term = fit.model.event_model.terms[0]
    assert term.normalize is True
    assert term.summate is False


def test_fmri_lm_accepts_identity_hrf_covariate_term(synthetic_run) -> None:
    """Sampled covariates lower as event-model identity-HRF regressors."""
    events, _Y, tr = synthetic_run
    n_scans = 60
    seed = np.linspace(-1.0, 1.0, n_scans)
    source = pd.DataFrame({"seed": seed})
    y = (2.0 * seed[:, None] + 0.1).astype(np.float64)
    dataset = fm.fmri_dataset(y, tr=tr, events=events)

    fit = fmri_lm(covariate("seed", source=source), dataset)
    names = fit.design_columns().names
    seed_index = names.index("seed")

    assert "seed" in names
    assert fit.model.event_model.terms[0].hrf is None
    np.testing.assert_allclose(fit.coef()[seed_index, 0], 2.0, atol=1e-10)


def test_fmri_lm_spec_dataset_with_config_kwarg(synthetic_run):
    events, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr, events=events)

    fit = fmri_lm("hrf(trial_type)", ds, config=FmriLmConfig())
    assert isinstance(fit, FmriLm)


def test_fmri_lm_spec_dataset_with_explicit_baseline(synthetic_run):
    events, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr, events=events)

    bm = fm.baseline_model(basis="poly", degree=2, sframe=ds.sampling_frame)
    fit = fmri_lm("hrf(trial_type)", ds, baseline=bm)
    # Poly degree-2 + intercept → at least 3 baseline columns.
    assert fit.n_coefficients >= 4


def test_fmri_lm_legacy_model_only_path_still_works():
    """Pre-built FmriModel-shaped object continues to work."""

    class _DummyDataset:
        def get_data(self, run):
            if run != 0:
                raise IndexError
            return self._y

        def get_censor(self, run):
            return None

        def __init__(self, y):
            self._y = y

    class _DummyModel:
        def __init__(self, x, y):
            self._x = x
            self.dataset = _DummyDataset(y)
            self.n_runs = 1

        def design_matrix_array(self, run):
            if run != 0:
                raise IndexError
            return self._x

        def contrast_weights(self):
            return {}

    rng = np.random.default_rng(0)
    n, p, v = 40, 3, 4
    x = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    y = x @ np.array([[1.0], [2.0], [0.5]]) + rng.standard_normal((n, v)) * 0.3
    fit = fmri_lm(_DummyModel(x, y))
    assert fit.n_coefficients == p
    assert fit.n_voxels == v


def test_fmri_lm_legacy_model_with_config_positional_still_works():
    """``fmri_lm(model, FmriLmConfig())`` legacy positional signature."""

    class _DummyDataset:
        def get_data(self, run):
            return self._y

        def get_censor(self, run):
            return None

        def __init__(self, y):
            self._y = y

    class _DummyModel:
        def __init__(self, x, y):
            self._x = x
            self.dataset = _DummyDataset(y)
            self.n_runs = 1

        def design_matrix_array(self, run):
            return self._x

        def contrast_weights(self):
            return {}

    rng = np.random.default_rng(1)
    x = np.column_stack([np.ones(30), rng.standard_normal((30, 1))])
    y = rng.standard_normal((30, 2))
    fit = fmri_lm(_DummyModel(x, y), FmriLmConfig())
    assert isinstance(fit, FmriLm)


def test_fmri_lm_mixing_model_and_dataset_raises():
    """Cannot pass both an FmriModel-like object and a dataset."""

    class _DummyModel:
        design_matrix_array = lambda self, run: np.zeros((1, 1))
        dataset = object()
        n_runs = 1

    ds = fm.matrix_dataset(np.zeros((4, 1)), tr=2.0)
    with pytest.raises(ValueError, match="pre-built model and a dataset"):
        fmri_lm(_DummyModel(), ds)


def test_fmri_lm_spec_without_dataset_raises():
    with pytest.raises(ValueError, match="requires an FmriDataset"):
        fmri_lm("hrf(trial_type)")


def test_fmri_lm_spec_with_dataset_missing_events_raises(synthetic_run):
    _, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr)  # no events
    with pytest.raises(ValueError, match="event table"):
        fmri_lm("hrf(trial_type)", ds)


def test_fmri_lm_auto_detects_run_and_duration_columns(synthetic_run):
    events, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr, events=events)
    # No explicit block= or durations= arguments — must auto-detect.
    fit = fmri_lm("hrf(trial_type)", ds)
    assert isinstance(fit, FmriLm)


def test_fmri_lm_falls_back_to_single_block_when_no_run_column(synthetic_run):
    events, Y, tr = synthetic_run
    events_no_run = events.drop(columns=["run"])
    ds = fm.fmri_dataset(Y, tr=tr, events=events_no_run)
    fit = fmri_lm("hrf(trial_type)", ds)
    assert isinstance(fit, FmriLm)


def test_fmri_lm_double_config_raises(synthetic_run):
    _, Y, _ = synthetic_run
    cfg = FmriLmConfig()

    class _DummyDataset:
        def get_data(self, run):
            return Y

        def get_censor(self, run):
            return None

    class _DummyModel:
        design_matrix_array = lambda self, run: np.eye(Y.shape[0])
        dataset = _DummyDataset()
        n_runs = 1

        def contrast_weights(self):
            return {}

    with pytest.raises(ValueError, match="both positionally and as a kwarg"):
        fmri_lm(_DummyModel(), cfg, config=FmriLmConfig())


def test_fmri_lm_rejects_typed_engine_options_with_legacy_kwargs(synthetic_run):
    events, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr, events=events)

    with pytest.raises(ValueError, match="typed engine options"):
        fmri_lm("hrf(trial_type)", ds, engine=ChunkwiseEngineOptions(), chunk_size=13)


def test_is_fmri_model_like_helper():
    """Direct unit on the duck-type discriminator."""

    class Yes:
        design_matrix_array = lambda self, run: None
        dataset = None
        n_runs = 1

    class No:
        pass

    assert _is_fmri_model_like(Yes())
    assert not _is_fmri_model_like(No())
    assert not _is_fmri_model_like("hrf(trial_type)")
    assert not _is_fmri_model_like(["hrf(x)"])
