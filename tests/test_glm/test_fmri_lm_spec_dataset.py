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
from fmrimod.glm.fmri_lm import FmriLm, _is_fmri_model_like, fmri_lm
from fmrimod.model.config import FmriLmConfig


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
    events, Y, tr = synthetic_run
    ds = fm.fmri_dataset(Y, tr=tr, events=events)
    # The positional second arg here is a dataset, so it should NOT trigger
    # the "double config" error.  Only triggers when a config sits in both
    # positions.
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
