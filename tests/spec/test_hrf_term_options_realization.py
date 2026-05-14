"""Realized-design checks for non-timing HRF term options."""

from __future__ import annotations

import numpy as np
import pandas as pd

import fmrimod as fm
from fmrimod.spec import hrf

TR = 2.0
N_SCANS = 40


def _single_trial_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "onset": [10.0],
            "duration": [0.0],
            "trial_type": ["A"],
            "run": [1],
        }
    )


def _two_level_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "onset": [8.0, 18.0, 28.0, 38.0],
            "duration": [0.0, 0.0, 0.0, 0.0],
            "trial_type": ["A", "B", "A", "B"],
            "run": [1, 1, 1, 1],
        }
    )


def _zero_dataset(events: pd.DataFrame) -> fm.FmriDataset:
    return fm.fmri_dataset(np.zeros((N_SCANS, 1)), tr=TR, events=events)


def test_typed_hrf_lag_shifts_realized_fir_columns() -> None:
    """``lag`` changes realised timing, not only lowered metadata."""

    events = _single_trial_events()
    ds = _zero_dataset(events)
    base = fm.fmri_lm(hrf("trial_type", basis="fir", nbasis=4), ds)
    lagged = fm.fmri_lm(hrf("trial_type", basis="fir", nbasis=4, lag=TR), ds)

    base_x = np.asarray(base.model.event_model.design_matrix)
    lagged_x = np.asarray(lagged.model.event_model.design_matrix)
    assert base_x.shape == lagged_x.shape == (N_SCANS, 4)
    for basis_col in range(base_x.shape[1]):
        base_rows = np.flatnonzero(base_x[:, basis_col])
        lagged_rows = np.flatnonzero(lagged_x[:, basis_col])
        assert lagged_rows[0] - base_rows[0] == 1
        assert not np.array_equal(lagged_rows, base_rows)


def test_typed_hrf_nbasis_controls_realized_basis_width() -> None:
    """``nbasis`` on generator-backed HRFs controls realised columns."""

    fit = fm.fmri_lm(
        hrf("trial_type", basis="fir", nbasis=4),
        _zero_dataset(_two_level_events()),
    )

    task = [c for c in fit.design_columns().columns if c.role == "task"]
    assert len(task) == 8
    assert sorted({c.level for c in task}) == ["A", "B"]
    assert {c.basis_ix for c in task} == {1, 2, 3, 4}
    assert {c.basis_total for c in task} == {4}


def test_typed_hrf_fun_realizes_custom_multibasis_columns() -> None:
    """``hrf_fun`` is used as the realised HRF, with declared basis width."""

    def custom_two_basis(t):
        t = np.asarray(t, dtype=np.float64)
        return np.column_stack(
            [
                np.exp(-0.5 * np.maximum(t, 0.0)),
                t * np.exp(-0.5 * np.maximum(t, 0.0)),
            ]
        )

    fit = fm.fmri_lm(
        hrf("trial_type", hrf_fun=custom_two_basis, nbasis=2),
        _zero_dataset(_two_level_events()),
    )

    x = np.asarray(fit.model.event_model.design_matrix)
    task = [c for c in fit.design_columns().columns if c.role == "task"]
    assert x.shape == (N_SCANS, 4)
    assert {c.basis_ix for c in task} == {1, 2}
    assert {c.basis_total for c in task} == {2}
    assert np.max(np.abs(x)) > 0.0


def test_typed_hrf_prefix_controls_realized_column_names() -> None:
    """``prefix`` affects user-facing design column names."""

    fit = fm.fmri_lm(
        hrf("trial_type", basis="spm", prefix="stim"),
        _zero_dataset(_two_level_events()),
    )

    task_names = [
        c.name for c in fit.design_columns().columns if c.role == "task"
    ]
    assert task_names
    assert all(name.startswith("stim_") for name in task_names)
