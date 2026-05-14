"""Executable contracts for the fmridataset consolidation PRD.

These tests encode the single-source-of-truth decisions in
``docs/contracts/fmridataset_consolidation_plan_v1.md``. Strict xfails mark
contracts that are intentionally not implemented until the migration starts;
an XPASS means the implementation changed and the gate should be reviewed.
"""

from __future__ import annotations

import numpy as np
import pytest

import fmrimod as fm
import fmrimod.dataset as dataset
from fmrimod.sampling import SamplingFrame


def test_sampling_frame_has_single_owner() -> None:
    assert fm.SamplingFrame is SamplingFrame
    if hasattr(dataset, "SamplingFrame"):
        assert dataset.SamplingFrame is SamplingFrame


def test_sampling_frame_samples_are_acquisition_times() -> None:
    sf = SamplingFrame(blocklens=[3, 2], tr=2.0)
    np.testing.assert_allclose(sf.samples, np.array([1.0, 3.0, 5.0, 7.0, 9.0]))
    assert not np.array_equal(sf.samples, np.arange(1, 6))


def test_integer_sample_indices_have_explicit_helper() -> None:
    assert hasattr(dataset, "sample_indices")
    sf = SamplingFrame(blocklens=[3, 2], tr=2.0)
    np.testing.assert_array_equal(dataset.sample_indices(sf), np.arange(1, 6))


def test_one_based_run_ids_are_explicit_opt_in() -> None:
    assert hasattr(SamplingFrame, "run_ids")
    sf = SamplingFrame(blocklens=[2, 3], tr=1.0)
    np.testing.assert_array_equal(
        sf.run_ids(one_based=False),
        np.array([0, 0, 1, 1, 1]),
    )
    np.testing.assert_array_equal(
        sf.run_ids(one_based=True),
        np.array([1, 1, 2, 2, 2]),
    )


def test_matrix_dataset_returns_canonical_dataset_class() -> None:
    mat = np.arange(20, dtype=np.float64).reshape(10, 2)
    assert fm.matrix_dataset is not dataset.matrix_dataset
    ds = dataset.matrix_dataset(mat, tr=2.0)
    assert isinstance(ds, dataset.FmriDataset)
    assert isinstance(fm.matrix_dataset(mat, tr=2.0), dataset.FmriDataset)


def test_n_timepoints_is_total_count() -> None:
    mat = np.arange(40, dtype=np.float64).reshape(20, 2)
    ds = fm.matrix_dataset(mat, tr=2.0, run_length=[8, 12])
    assert ds.n_timepoints == 20


def test_per_run_lengths_are_not_n_timepoints() -> None:
    mat = np.arange(40, dtype=np.float64).reshape(20, 2)
    ds = fm.matrix_dataset(mat, tr=2.0, run_length=[8, 12])
    lengths = getattr(ds, "run_lengths", getattr(ds, "blocklens", None))
    assert tuple(lengths) == (8, 12)
    assert ds.n_timepoints == 20


def test_get_data_uses_rows_cols_matrix_slicing() -> None:
    mat = np.arange(40, dtype=np.float64).reshape(20, 2)
    ds = fm.matrix_dataset(mat, tr=2.0, run_length=[8, 12])
    np.testing.assert_array_equal(
        ds.get_data(rows=np.array([0, 9]), cols=np.array([1])),
        mat[[0, 9]][:, [1]],
    )
    np.testing.assert_array_equal(
        dataset.get_data(ds, rows=np.array([0, 9]), cols=np.array([1])),
        mat[[0, 9]][:, [1]],
    )
    np.testing.assert_array_equal(
        dataset.get_data_matrix(ds, rows=np.array([0, 9]), cols=np.array([1])),
        mat[[0, 9]][:, [1]],
    )


def test_run_access_uses_explicit_method() -> None:
    mat = np.arange(40, dtype=np.float64).reshape(20, 2)
    ds = fm.matrix_dataset(mat, tr=2.0, run_length=[8, 12])
    assert hasattr(ds, "get_run_data")
    np.testing.assert_array_equal(ds.get_run_data(1), mat[8:20])
    np.testing.assert_array_equal(dataset.get_run_data(ds, 1), mat[8:20])


def test_dataset_metadata_accessors_use_canonical_meanings() -> None:
    mat = np.arange(40, dtype=np.float64).reshape(20, 2)
    ds = fm.matrix_dataset(mat, tr=2.0, run_length=[8, 12])

    assert dataset.get_TR(ds) == 2.0
    assert dataset.get_run_lengths(ds) == (8, 12)
    assert dataset.blocklens(ds) == (8, 12)
    assert dataset.n_runs(ds) == 2
    assert dataset.n_timepoints(ds) == 20
    assert dataset.get_total_duration(ds) == 40.0
    np.testing.assert_array_equal(
        dataset.get_run_duration(ds),
        np.array([16.0, 24.0]),
    )
    np.testing.assert_array_equal(dataset.all_timepoints(ds), np.arange(20))
    np.testing.assert_array_equal(
        dataset.blockids(ds),
        np.array([0] * 8 + [1] * 12, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        dataset.blockids(ds, one_based=True),
        np.array([1] * 8 + [2] * 12, dtype=np.int32),
    )
    np.testing.assert_allclose(dataset.samples(ds), ds.sampling_frame.samples)


def test_fmridataset_facade_identity_when_available() -> None:
    fmridataset = pytest.importorskip("fmridataset")
    assert fmridataset.FmriDataset is dataset.FmriDataset
    assert fmridataset.matrix_dataset is dataset.matrix_dataset
    assert fmridataset.SamplingFrame is SamplingFrame
