"""Tests for the neuroim-backed dataset adapter and fmri_dataset dispatch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.dataset import fmri_dataset
from fmrimod.dataset.adapters import NeuroVecAdapter
from fmrimod.dataset.fmri_dataset import FmriDataset

neuroim = pytest.importorskip("neuroim")


def _make_neurovec(shape4d=(4, 4, 3, 20), *, seed: int = 7) -> "neuroim.DenseNeuroVec":
    rng = np.random.default_rng(seed)
    data = rng.normal(size=shape4d).astype(np.float64)
    # Force a zero plane so the default mask drops some voxels deterministically.
    data[0, :, :, :] = 0.0
    # 4-D NeuroSpace needs spacing/origin to match dim count (3 spatial + 1 time).
    space = neuroim.NeuroSpace(
        dim=shape4d,
        spacing=(2.0, 2.0, 2.0, 1.0),
        origin=(0.0, 0.0, 0.0, 0.0),
    )
    return neuroim.DenseNeuroVec(data, space)


def _make_logical_mask(shape3d=(4, 4, 3), *, seed: int = 11) -> "neuroim.LogicalNeuroVol":
    rng = np.random.default_rng(seed)
    arr = rng.uniform(size=shape3d) > 0.5
    arr[0, 0, 0] = True  # guarantee at least one True voxel
    space = neuroim.NeuroSpace(
        dim=shape3d, spacing=(2.0, 2.0, 2.0), origin=(0.0, 0.0, 0.0)
    )
    return neuroim.LogicalNeuroVol(arr, space)


def test_neurovec_adapter_default_mask_drops_zero_voxels():
    vec = _make_neurovec()
    adapter = NeuroVecAdapter(vec, tr=2.0)
    mask = adapter.get_mask()
    assert mask.shape == (4, 4, 3)
    # The forced zero plane must be False in the auto mask.
    assert not mask[0].any()
    # Other voxels with random nonzero data should be True.
    assert mask[1:].all()
    assert adapter.n_voxels == int(mask.sum())
    assert adapter.n_runs == 1
    assert adapter.n_timepoints == [20]
    data = adapter.get_data(0)
    assert data.shape == (20, adapter.n_voxels)
    assert data.dtype == np.float64
    # SamplingFrame round-trip
    sf = adapter.get_sampling_frame()
    assert sf.blocklens == [20]
    assert np.isclose(sf.TR, 2.0)


def test_neurovec_adapter_explicit_logical_mask():
    vec = _make_neurovec()
    mask = _make_logical_mask()
    adapter = NeuroVecAdapter(vec, mask=mask, tr=2.0)
    np.testing.assert_array_equal(adapter.get_mask(), np.asarray(mask.data, dtype=bool))
    assert adapter.n_voxels == int(np.asarray(mask.data).sum())


def test_neurovec_adapter_multi_run_consistency():
    v1 = _make_neurovec(shape4d=(4, 4, 3, 10), seed=1)
    v2 = _make_neurovec(shape4d=(4, 4, 3, 15), seed=2)
    adapter = NeuroVecAdapter([v1, v2], tr=2.0)
    assert adapter.n_runs == 2
    assert adapter.n_timepoints == [10, 15]
    assert adapter.get_data(0).shape == (10, adapter.n_voxels)
    assert adapter.get_data(1).shape == (15, adapter.n_voxels)


def test_neurovec_adapter_mismatched_spatial_shape_rejects():
    v1 = _make_neurovec(shape4d=(4, 4, 3, 10), seed=1)
    v2 = _make_neurovec(shape4d=(4, 4, 4, 10), seed=2)
    with pytest.raises(ValueError, match="spatial shape"):
        NeuroVecAdapter([v1, v2], tr=2.0)


def test_neurovec_adapter_mask_shape_rejects():
    vec = _make_neurovec(shape4d=(4, 4, 3, 10))
    bad_mask = np.ones((5, 5, 3), dtype=bool)
    with pytest.raises(ValueError, match="Mask shape"):
        NeuroVecAdapter(vec, mask=bad_mask, tr=2.0)


def test_neurovec_adapter_from_array_round_trip():
    rng = np.random.default_rng(3)
    arr = rng.normal(size=(3, 3, 2, 12)).astype(np.float64)
    adapter = NeuroVecAdapter.from_array(arr, spacing=(2.0, 2.0, 3.0), tr=1.5)
    assert adapter.n_runs == 1
    assert adapter.n_timepoints == [12]
    affine = adapter.get_affine()
    np.testing.assert_allclose(np.diag(affine)[:3], (2.0, 2.0, 3.0))


def test_fmri_dataset_accepts_neurovec():
    vec = _make_neurovec()
    events = pd.DataFrame({"onset": [0.0, 10.0], "trial_type": ["a", "b"]})
    ds = fmri_dataset(vec, tr=2.0, events=events)
    assert isinstance(ds, FmriDataset)
    assert ds.n_runs == 1
    assert ds.n_timepoints == 20
    assert ds.event_table is events


def test_fmri_dataset_accepts_4d_ndarray():
    rng = np.random.default_rng(5)
    arr = rng.normal(size=(3, 3, 2, 12)).astype(np.float64)
    ds = fm.fmri_dataset(arr, tr=1.5)
    assert isinstance(ds, FmriDataset)
    assert ds.n_runs == 1
    assert ds.n_timepoints == 12


def test_fmri_dataset_accepts_2d_ndarray():
    rng = np.random.default_rng(6)
    Y = rng.normal(size=(20, 5)).astype(np.float64)
    ds = fm.fmri_dataset(Y, tr=2.0)
    assert isinstance(ds, FmriDataset)
    assert ds.n_runs == 1
    assert ds.get_all_data().shape == (20, 5)


def test_fmri_dataset_accepts_sequence_of_neurovec():
    v1 = _make_neurovec(shape4d=(4, 4, 3, 10), seed=1)
    v2 = _make_neurovec(shape4d=(4, 4, 3, 15), seed=2)
    ds = fm.fmri_dataset([v1, v2], tr=2.0)
    assert ds.n_runs == 2
    assert ds.run_lengths == [10, 15]


def test_fmri_dataset_legacy_data_source_path_still_works():
    rng = np.random.default_rng(7)
    Y = rng.normal(size=(20, 5)).astype(np.float64)
    legacy = fm.matrix_dataset(Y, tr=2.0)  # produces FmriDataset with a BackendAdapter
    # Pass through the legacy `data_source` keyword.
    ds = fmri_dataset(data_source=legacy._source)  # type: ignore[attr-defined]
    assert ds.n_runs == 1
    assert ds.get_all_data().shape == (20, 5)


def test_fmri_dataset_rejects_unknown_input_type():
    with pytest.raises(TypeError, match="cannot build adapter"):
        fmri_dataset({"not": "a real image"}, tr=2.0)


def test_fmri_dataset_requires_tr_for_neurovec():
    vec = _make_neurovec()
    with pytest.raises(ValueError, match="`tr` is required"):
        fmri_dataset(vec)
