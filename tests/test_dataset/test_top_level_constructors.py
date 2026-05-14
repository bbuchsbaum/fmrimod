"""Tests for top-level constructor parity wrappers."""

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
import fmrimod.dataset as dataset
from fmrimod.dataset import MatrixBackend
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
    assert isinstance(dataset.matrix_dataset(y, tr=2.0), FmriDataset)
    assert isinstance(ds.storage_backend, MatrixBackend)
    assert ds.n_runs == 1
    assert ds.n_timepoints == 20
    assert ds.run_lengths == [20]
    assert ds.get_all_data().shape == (20, 3)
    assert np.isclose(ds.sampling_frame.TR, 2.0)


def test_matrix_dataset_split_by_run_length_int():
    y = np.arange(90, dtype=float).reshape(30, 3)
    ds = fm.matrix_dataset(y, tr=1.5, run_length=10)
    assert ds.n_runs == 3
    assert ds.n_timepoints == 30
    assert ds.run_lengths == [10, 10, 10]
    assert ds.get_data(0).shape == (10, 3)
    assert ds.get_data(2).shape == (10, 3)


def test_matrix_dataset_split_by_run_length_list_and_event_table():
    y = np.arange(72, dtype=float).reshape(24, 3)
    events = pd.DataFrame({"onset": [2.0, 8.0], "condition": ["A", "B"]})
    ds = fm.matrix_dataset(y, tr=[1.0, 2.0], run_length=[8, 16], event_table=events)
    assert ds.n_runs == 2
    assert ds.n_timepoints == 24
    assert ds.run_lengths == [8, 16]
    assert ds.event_table is events


def test_fmri_dataset_matrix_input_accepts_run_length():
    y = np.arange(72, dtype=float).reshape(24, 3)
    events = pd.DataFrame({"onset": [1.0], "condition": ["A"]})

    ds = fm.fmri_dataset(y, tr=2.0, run_length=[8, 16], events=events)

    assert isinstance(ds, FmriDataset)
    assert isinstance(ds.storage_backend, MatrixBackend)
    assert ds.n_runs == 2
    assert ds.run_lengths == [8, 16]
    assert ds.get_data(0).shape == (8, 3)
    assert ds.get_data(1).shape == (16, 3)
    assert ds.event_table is events


def test_fmri_dataset_run_length_rejects_non_matrix_inputs():
    with pytest.raises(ValueError, match="run_length.*2-D ndarray"):
        fm.fmri_dataset(np.zeros((2, 2, 2, 4)), tr=2.0, run_length=2)

    with pytest.raises(ValueError, match="run_length.*2-D ndarray"):
        fm.fmri_dataset([np.zeros((2, 2, 2, 4))], tr=2.0, run_length=2)


def test_matrix_dataset_accepts_TR_compatibility_spelling():
    y = np.arange(60, dtype=float).reshape(20, 3)
    ds = dataset.matrix_dataset(y, TR=2.0)

    assert ds.n_timepoints == 20
    assert np.isclose(ds.sampling_frame.TR, 2.0)

    with pytest.raises(ValueError, match="must agree"):
        dataset.matrix_dataset(y, tr=1.0, TR=2.0)


def test_matrix_dataset_validates_input_shape():
    bad = np.arange(8, dtype=float).reshape(2, 2, 2)
    with pytest.raises(ValueError, match="2-D matrix data"):
        fm.matrix_dataset(bad, tr=2.0)
