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


def test_boolean_selectors_act_as_masks() -> None:
    """Logical row/column selectors are masks, not 0/1 positions.

    R ``ec6efd3`` (ADR-010 selection algebra, ``R/axis-selection.R``): a
    logical selector must match the axis length and act as a mask; numeric
    positions must be whole.
    """
    mat = np.arange(40, dtype=np.float64).reshape(8, 5)
    ds = dataset.matrix_dataset(mat, tr=2.0)
    keep_rows = np.zeros(8, dtype=bool)
    keep_rows[[5, 7]] = True
    keep_cols = np.array([False, True, False, False, True])

    np.testing.assert_array_equal(
        ds.get_data(rows=keep_rows), ds.get_data(rows=[5, 7])
    )
    np.testing.assert_array_equal(
        ds.get_data(cols=keep_cols), ds.get_data(cols=[1, 4])
    )
    backend = ds.storage_backend
    np.testing.assert_array_equal(
        backend.get_data(rows=keep_rows), mat[[5, 7]]
    )
    np.testing.assert_array_equal(
        backend.get_data(cols=keep_cols), mat[:, [1, 4]]
    )
    with pytest.raises(ValueError, match="boolean mask length"):
        ds.get_data(rows=keep_rows[:4])
    with pytest.raises(ValueError, match="integers"):
        ds.get_data(rows=np.array([2.7]))
    with pytest.raises(ValueError, match="integers"):
        backend.get_data(rows=np.array([2.7]))
    # Python's negative-index and repeat semantics are kept.
    np.testing.assert_array_equal(ds.get_data(rows=[-1, -1]), mat[[7, 7]])


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


@pytest.mark.parametrize(
    "kwargs",
    [{"tr": float("nan")}, {"tr": 2.0, "start_time": float("nan")}, {"tr": float("inf")}],
)
def test_timing_must_be_finite(kwargs) -> None:
    # fmridataset 99d19fe (ADR-004): TR positive and finite, start_time finite.
    # Inventory v2 DS-08. All three entry points, so a constructor-only check
    # cannot pass.
    x = np.zeros((6, 2))
    with pytest.raises(ValueError):
        SamplingFrame(blocklens=[6], **kwargs)
    with pytest.raises(ValueError):
        dataset.matrix_dataset(x, **kwargs)
    with pytest.raises(ValueError):
        dataset.fmri_dataset(x, run_length=6, **kwargs)


def test_dataset_start_time_must_be_non_negative() -> None:
    # R splits a permissive fmrihrf sampling_frame from a strict fmridataset:
    # a negative first-volume offset is refused at the dataset constructors.
    x = np.zeros((6, 2))
    SamplingFrame(blocklens=[6], tr=2.0, start_time=-1.0)  # still allowed
    with pytest.raises(ValueError, match="start_time"):
        dataset.matrix_dataset(x, tr=2.0, start_time=-1.0)
    with pytest.raises(ValueError, match="start_time"):
        dataset.fmri_dataset(x, tr=2.0, run_length=6, start_time=-1.0)


def test_tr_spellings_agree_within_float_tolerance() -> None:
    # fmridataset cce6a44: seconds compared with rtol sqrt(eps). Inventory DS-09.
    x = np.zeros((6, 2))
    ds = dataset.matrix_dataset(x, tr=0.1 * 3, TR=0.3)
    np.testing.assert_allclose(ds.sampling_frame.tr, [0.3])
    with pytest.raises(ValueError, match="agree"):
        dataset.matrix_dataset(x, tr=0.3, TR=0.31)
    sf = dataset.matrix_dataset(x, tr=2.0, start_time=0.1 * 3, slice_timing_offset=0.3)
    np.testing.assert_allclose(sf.sampling_frame.start_time, [0.3])
