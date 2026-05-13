"""Tests for estimate_single_trial_from_dataset (Slice 1 of bd-01KRGQCT34QWSYKQ38BVFHD51E).

Verifies the public-seam wrapper consumes a typed FmriDataset + trialwise
spec, builds X internally, and produces a SingleTrialResult numerically
equivalent to the matrix-first estimate_single_trial(Y, X, ...) call with
the same realized design.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.single import (
    SingleTrialResult,
    estimate_single_trial,
    estimate_single_trial_from_dataset,
)


def _build_dataset_and_spec():
    """Construct a small FmriDataset with a single-run trialwise event table.

    The spec is passed as the string form ``"trialwise()"``. Compiling a bare
    :func:`fmrimod.trialwise.trialwise` Term object through the typed
    ``fmrimod.spec`` path is a separate follow-on (the placeholder event is
    not resolved by the typed compiler today); the string form is the
    established working path for trialwise designs and is what this Slice 1
    wrapper exercises.
    """
    rng = np.random.default_rng(2026)
    n_time, n_voxels = 80, 6
    tr = 2.0

    onsets = np.array([10.0, 30.0, 50.0, 70.0])
    events = pd.DataFrame(
        {
            "onset": onsets,
            "duration": np.zeros_like(onsets),
            "run": [1, 1, 1, 1],
        }
    )

    bold = rng.standard_normal((n_time, n_voxels)).astype(np.float64)
    ds = fm.fmri_dataset(bold, tr=tr, events=events)
    spec = "trialwise()"
    return ds, spec, events


def test_wrapper_returns_single_trial_result_with_trial_labels():
    """Wrapper returns SingleTrialResult carrying one label per trial."""
    ds, spec, events = _build_dataset_and_spec()

    result = estimate_single_trial_from_dataset(ds, spec, method="lss")

    assert isinstance(result, SingleTrialResult)
    assert result.betas.shape[1] == 6  # n_voxels
    assert result.betas.shape[0] == len(events)
    assert result.trial_labels is not None
    assert len(result.trial_labels) == len(events)


def test_wrapper_is_top_level_public_entry():
    """The dataset wrapper is reachable from the public fmrimod namespace."""
    ds, spec, events = _build_dataset_and_spec()

    result = fm.estimate_single_trial_from_dataset(ds, spec, method="lss")

    assert isinstance(result, SingleTrialResult)
    assert result.betas.shape[0] == len(events)


def test_wrapper_equivalent_to_matrix_estimate_on_same_realised_design():
    """For identical (Y, X), wrapper matches the matrix-first dispatcher.

    Construct the realised design via the same event_model() path the
    wrapper uses internally, then compare the wrapper output to a
    direct estimate_single_trial(Y, X) call. Numerical equality proves
    the wrapper is a transparent typed-entry alternative.
    """
    from fmrimod.design.event_model import event_model as _build_event_model

    ds, spec, _ = _build_dataset_and_spec()

    em = _build_event_model(
        formula=spec,
        data=ds.event_table,
        block="run",
        sampling_frame=ds.get_sampling_frame(),
    )
    X = np.ascontiguousarray(np.asarray(em.design_matrix, dtype=np.float64))
    Y = np.asarray(ds.get_data(), dtype=np.float64)

    matrix_result = estimate_single_trial(Y, X, method="lss")
    wrapper_result = estimate_single_trial_from_dataset(ds, spec, method="lss")

    np.testing.assert_array_equal(wrapper_result.betas, matrix_result.betas)


def test_wrapper_rejects_spec_without_trialwise_term():
    """A spec containing no trialwise() must raise a clear error."""
    rng = np.random.default_rng(0)
    n_time, n_voxels = 40, 3
    bold = rng.standard_normal((n_time, n_voxels)).astype(np.float64)
    events = pd.DataFrame({
        "onset": [4.0, 20.0],
        "duration": [0.0, 0.0],
        "trial_type": ["A", "B"],
        "run": [1, 1],
    })
    ds = fm.fmri_dataset(bold, tr=2.0, events=events)
    spec = "hrf(trial_type)"

    with pytest.raises(ValueError, match="trialwise"):
        estimate_single_trial_from_dataset(ds, spec, method="lss")


def test_wrapper_method_dispatch_matches_string_and_enum():
    """method='lsa' is accepted and produces a result of the same shape."""
    ds, spec, events = _build_dataset_and_spec()

    result_lsa = estimate_single_trial_from_dataset(ds, spec, method="lsa")
    assert isinstance(result_lsa, SingleTrialResult)
    assert result_lsa.betas.shape[0] == len(events)
