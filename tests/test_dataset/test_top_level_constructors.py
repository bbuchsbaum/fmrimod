"""Tests for top-level constructor parity wrappers."""

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.dataset.fmri_dataset import FmriDataset
from fmrimod.events import EventFactor, EventMatrix, EventTerm, EventVariable


def test_top_level_event_constructor_wrappers():
    ef = fm.event_factor(
        name="condition",
        onsets=[1.0, 2.0, 3.0],
        values=["A", "B", "A"],
    )
    ev = fm.event_variable(
        name="rating",
        onsets=[1.0, 2.0, 3.0],
        values=[0.1, 0.2, 0.3],
    )
    em = fm.event_matrix(
        name="motion",
        onsets=[1.0, 2.0, 3.0],
        values=np.array([[1.0, 0.0], [0.5, 0.1], [0.2, -0.2]]),
    )
    et = fm.event_term(ef, ev, name="cond_x_rating", interaction=True)

    assert isinstance(ef, EventFactor)
    assert isinstance(ev, EventVariable)
    assert isinstance(em, EventMatrix)
    assert isinstance(et, EventTerm)
    assert et.name == "cond_x_rating"
    assert et.interaction is True


def test_matrix_dataset_single_run_construction():
    y = np.arange(60, dtype=float).reshape(20, 3)
    ds = fm.matrix_dataset(y, tr=2.0)
    assert isinstance(ds, FmriDataset)
    assert ds.n_runs == 1
    assert ds.n_timepoints == [20]
    assert ds.get_all_data().shape == (20, 3)
    assert np.isclose(ds.sampling_frame.TR, 2.0)


def test_matrix_dataset_split_by_run_length_int():
    y = np.arange(90, dtype=float).reshape(30, 3)
    ds = fm.matrix_dataset(y, tr=1.5, run_length=10)
    assert ds.n_runs == 3
    assert ds.n_timepoints == [10, 10, 10]
    assert ds.get_data(0).shape == (10, 3)
    assert ds.get_data(2).shape == (10, 3)


def test_matrix_dataset_split_by_run_length_list_and_event_table():
    y = np.arange(72, dtype=float).reshape(24, 3)
    events = pd.DataFrame({"onset": [2.0, 8.0], "condition": ["A", "B"]})
    ds = fm.matrix_dataset(y, tr=[1.0, 2.0], run_length=[8, 16], event_table=events)
    assert ds.n_runs == 2
    assert ds.n_timepoints == [8, 16]
    assert ds.event_table is events


def test_matrix_dataset_validates_input_shape():
    bad = np.arange(8, dtype=float).reshape(2, 2, 2)
    with pytest.raises(ValueError, match="2-D matrix data"):
        fm.matrix_dataset(bad, tr=2.0)
