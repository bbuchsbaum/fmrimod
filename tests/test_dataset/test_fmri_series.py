"""Tests for canonical FmriSeries helpers."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

import fmridataset
import fmrimod
import fmrimod.dataset as dataset


def test_fmri_series_extracts_with_selector_and_timepoints() -> None:
    mat = np.arange(40, dtype=float).reshape(10, 4)
    ds = fmrimod.matrix_dataset(mat, tr=2.0, run_length=[4, 6])
    chunks = list(dataset.data_chunks(ds, nchunks=2))

    series = dataset.fmri_series(
        ds,
        selector=dataset.index_selector(chunks[0].voxel_ind),
        timepoints=chunks[0].row_ind,
    )

    assert isinstance(series, dataset.FmriSeries)
    np.testing.assert_array_equal(series.data, chunks[0].data)
    np.testing.assert_array_equal(dataset.as_matrix(series), chunks[0].data)
    assert series.shape == chunks[0].data.shape
    assert series.voxel_info["voxel"].tolist() == chunks[0].voxel_ind.tolist()
    assert series.temporal_info["run_id"].tolist() == [0] * 4 + [1] * 6
    np.testing.assert_allclose(
        series.temporal_info["sample_time"].to_numpy(),
        ds.sampling_frame.samples,
    )


def test_fmri_series_dataframe_and_matrix_outputs() -> None:
    mat = np.arange(24, dtype=float).reshape(6, 4)
    ds = fmrimod.matrix_dataset(mat, tr=1.5)
    selector = np.array([1, 3], dtype=np.intp)
    timepoints = np.array([0, 2, 5], dtype=np.intp)

    dense = dataset.fmri_series(
        ds,
        selector=selector,
        timepoints=timepoints,
        output="matrix",
    )
    np.testing.assert_array_equal(dense, mat[timepoints][:, selector])

    series = dataset.fmri_series(ds, selector=selector, timepoints=timepoints)
    frame = dataset.to_dataframe(series)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 6
    assert set(["run_id", "timepoint", "sample_time", "voxel", "signal"]).issubset(
        frame.columns
    )
    np.testing.assert_array_equal(
        frame["signal"].to_numpy(),
        mat[timepoints][:, selector].ravel(),
    )
    with pytest.warns(DeprecationWarning):
        alias = dataset.series(ds, selector=selector, timepoints=timepoints)
    assert dataset.is_fmri_series(alias)


def test_timepoint_resolution_validates_shape_and_bounds() -> None:
    ds = fmrimod.matrix_dataset(np.ones((5, 2)), tr=2.0)
    np.testing.assert_array_equal(
        dataset.resolve_timepoints(ds, np.array([True, False, True, False, False])),
        np.array([0, 2]),
    )
    with pytest.raises(ValueError, match="Boolean timepoints length"):
        dataset.resolve_timepoints(ds, np.array([True, False]))
    with pytest.raises(IndexError, match="out of range"):
        dataset.resolve_timepoints(ds, np.array([5]))


def test_fmridataset_series_facade_reexports_canonical_objects() -> None:
    assert fmridataset.FmriSeries is dataset.FmriSeries
    assert fmridataset.fmri_series is dataset.fmri_series
    assert fmridataset.new_fmri_series is dataset.new_fmri_series
    assert fmridataset.is_fmri_series is dataset.is_fmri_series
    assert fmridataset.as_matrix is dataset.as_matrix
    assert fmridataset.as_tibble is dataset.as_tibble
    assert fmridataset.resolve_selector is dataset.resolve_selector
    assert fmridataset.resolve_timepoints is dataset.resolve_timepoints

    facade_fmri_series = importlib.import_module("fmridataset.fmri_series")
    assert facade_fmri_series.FmriSeries is dataset.FmriSeries
    assert facade_fmri_series.fmri_series is dataset.fmri_series
