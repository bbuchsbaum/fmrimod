"""Regression tests for the cosine drift basis.

Wiring history: ``drift(basis="cosine", cutoff=...)`` was declared on
the typed Spec for a long time but was a dead end — the lowering
pipeline rejected ``"cosine"`` at fit time
(``ValueError: Invalid basis: cosine``). The cosine basis is now
implemented in ``BaselineSpec._cosine_basis`` (a DCT-II matching
SPM's and Nilearn's ``create_cosine_drift``), and the typed compile
step translates ``cutoff`` (high-pass period in seconds) to the
``degree`` argument (number of basis functions) using
``floor(2 * T * 1/cutoff)`` against the longest block.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nilearn.glm.first_level.design_matrix import create_cosine_drift

import fmrimod as fm
from fmrimod.dataset.constructors import matrix_dataset
from fmrimod.spec import drift, hrf, intercept


def _events_and_dataset(n_scans: int = 180, tr: float = 2.0) -> tuple[pd.DataFrame, object]:
    events = pd.DataFrame(
        {
            "onset": np.linspace(8.0, 96.0, 8),
            "duration": 0.0,
            "trial_type": ["A", "B"] * 4,
            "run": 1,
        }
    )
    ds = fm.fmri_dataset(np.zeros((n_scans, 1)), tr=tr, events=events)
    return events, ds


def test_cosine_drift_matches_nilearn_bitwise() -> None:
    """fmrimod's cosine drift columns are bitwise-equal to Nilearn's."""
    events, ds = _events_and_dataset(n_scans=180, tr=2.0)
    spec = (
        hrf("trial_type", basis="spm", norm="spm")
        + drift("cosine", cutoff=128.0)
        + intercept(per="run")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(spec, ds)
    X_fm = fit.model.design_matrix_array(run=None)
    drift_indices = [
        c.index
        for c in fit.design_columns().columns
        if c.role == "drift" and "cosine" in (c.name or "")
    ]
    fm_drift = X_fm[:, drift_indices]

    frame_times = np.arange(180) * 2.0
    nl_full = create_cosine_drift(high_pass=1.0 / 128.0, frame_times=frame_times)
    nl_drift = nl_full[:, :-1]  # drop trailing constant (intercept is separate)

    assert fm_drift.shape == nl_drift.shape
    np.testing.assert_array_equal(fm_drift, nl_drift)


def test_cosine_drift_degree_from_cutoff() -> None:
    """SPM convention: ``n_basis = floor(2 * T / cutoff)``."""
    events, ds = _events_and_dataset(n_scans=200, tr=2.0)
    # T = 200 * 2.0 = 400 s; cutoff = 100 -> n = floor(2 * 400 / 100) = 8
    spec = (
        hrf("trial_type", basis="spm", norm="spm")
        + drift("cosine", cutoff=100.0)
        + intercept(per="run")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(spec, ds)
    cosine_cols = [
        c for c in fit.design_columns().columns
        if c.role == "drift" and "cosine" in (c.name or "")
    ]
    assert len(cosine_cols) == 8


def test_cosine_drift_without_cutoff_errors() -> None:
    """``drift("cosine")`` without ``cutoff=`` must fail clearly."""
    events, ds = _events_and_dataset()
    with pytest.raises(ValueError, match="cutoff"):
        fm.fmri_lm(
            hrf("trial_type", basis="spm", norm="spm") + drift("cosine"),
            ds,
        )


def test_cutoff_with_non_cosine_basis_warns() -> None:
    """A ``cutoff=`` on a non-cosine basis is ignored with a warning."""
    events, ds = _events_and_dataset()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fm.fmri_lm(
            hrf("trial_type", basis="spm", norm="spm")
            + drift("poly", degree=2, cutoff=128.0),
            ds,
        )
    matches = [
        w for w in captured
        if issubclass(w.category, UserWarning)
        and "cutoff" in str(w.message) and "cosine" in str(w.message)
    ]
    assert matches, "expected a warning about cutoff being ignored"


def test_cosine_drift_per_run_in_multirun_dataset() -> None:
    """In a multi-run dataset the cosine basis is block-diagonal per run."""
    events_rows = []
    for run_id in (1, 2):
        for k, onset in enumerate(np.linspace(8.0, 96.0, 6)):
            events_rows.append(
                {
                    "onset": float(onset),
                    "duration": 0.0,
                    "trial_type": "A" if k % 2 == 0 else "B",
                    "run": run_id,
                }
            )
    events = pd.DataFrame(events_rows)
    Y = np.zeros((200, 1))
    ds = matrix_dataset(Y, tr=2.0, run_length=100, event_table=events)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = fm.fmri_lm(
            hrf("trial_type", basis="spm", norm="spm")
            + drift("cosine", cutoff=64.0)
            + intercept(per="run"),
            ds,
        )
    X = fit.model.design_matrix_array(run=None)
    cosine_block_1 = [
        c for c in fit.design_columns().columns
        if c.role == "drift" and "cosine" in (c.name or "")
        and "block_1" in (c.name or "")
    ]
    cosine_block_2 = [
        c for c in fit.design_columns().columns
        if c.role == "drift" and "cosine" in (c.name or "")
        and "block_2" in (c.name or "")
    ]
    assert cosine_block_1, "expected per-run cosine columns for block 1"
    assert cosine_block_2, "expected per-run cosine columns for block 2"
    # Block-diagonal: each block's cosine cols are zero in the other block.
    for c in cosine_block_1:
        assert np.allclose(X[100:, c.index], 0.0), (
            f"{c.name}: leaked into block 2"
        )
    for c in cosine_block_2:
        assert np.allclose(X[:100, c.index], 0.0), (
            f"{c.name}: leaked into block 1"
        )
