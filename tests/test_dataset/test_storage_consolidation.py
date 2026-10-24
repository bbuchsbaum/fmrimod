"""One storage contract under FmriDataset (consolidation plan; drift v2 D1).

Every dataset reads through one ``StorageBackend`` paired with a
``SamplingFrame``; the public adapters are thin facades over a backend, and
``rows``/``cols`` reads reach the backend instead of materialising all data.
"""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.dataset import FmriDataset, StorageBackend, matrix_dataset
from fmrimod.dataset.adapters import NumpyAdapter
from fmrimod.dataset.backends import ConcatBackend, MatrixBackend
from fmrimod.dataset.space import VolumeSpace
from fmrimod.sampling import SamplingFrame


class _CountingBackend(MatrixBackend):
    """MatrixBackend that records every read."""

    def __init__(self, data, **kw):
        super().__init__(data, **kw)
        self.reads: list[tuple[object, object]] = []

    def get_data(self, rows=None, cols=None):
        self.reads.append((None if rows is None else np.asarray(rows).tolist(),
                           None if cols is None else np.asarray(cols).tolist()))
        return super().get_data(rows=rows, cols=cols)


def test_slices_and_runs_are_read_from_the_backend_not_materialised():
    data = np.arange(40, dtype=float).reshape(10, 4)
    be = _CountingBackend(data)
    ds = FmriDataset(be, sampling_frame=SamplingFrame(blocklens=[6, 4], tr=2.0))
    np.testing.assert_array_equal(ds.get_data(rows=[7, 2], cols=[3]), data[[7, 2]][:, [3]])
    np.testing.assert_array_equal(ds.get_run_data(1), data[6:])
    assert be.reads == [([7, 2], [3]), ([6, 7, 8, 9], None)]


def test_constructors_and_adapters_resolve_to_one_backend():
    sf = SamplingFrame(blocklens=[3, 2], tr=2.0)
    runs = [np.ones((3, 4)), np.full((2, 4), 2.0)]
    via_adapter = FmriDataset(NumpyAdapter(runs, sf))
    via_matrix = matrix_dataset(runs, tr=2.0)
    for ds in (via_adapter, via_matrix):
        assert isinstance(ds.storage_backend, StorageBackend)
        np.testing.assert_array_equal(ds.get_all_data(), np.vstack(runs))
        np.testing.assert_array_equal(ds.get_run_data(1), runs[1])


def _write_nifti(path, arr):
    nib = pytest.importorskip("nibabel")
    nib.save(nib.Nifti1Image(arr, np.eye(4)), path)
    return str(path)


@pytest.mark.parametrize("path", ["matrix", "numpy_adapter", "nifti_preload", "nifti_lazy"])
def test_reads_are_fresh_arrays(path, tmp_path):
    # Writing into a read (e.g. centring before analysis) must never change
    # the dataset; a backend may cache, but it must not hand the cache out.
    from fmrimod.dataset import nifti_backend

    sf = SamplingFrame(blocklens=[2, 2], tr=2.0)
    if path == "matrix":
        ds = matrix_dataset(np.zeros((4, 3)), tr=2.0, run_length=[2, 2])
    elif path == "numpy_adapter":
        ds = FmriDataset(NumpyAdapter([np.zeros((2, 3)), np.zeros((2, 3))], sf))
    else:
        img = _write_nifti(tmp_path / "b.nii", np.zeros((3, 1, 1, 4)))
        mask = _write_nifti(tmp_path / "m.nii", np.ones((3, 1, 1), "uint8"))
        be = nifti_backend(img, mask, preload=path == "nifti_preload")
        be.open()
        ds = FmriDataset(be, sampling_frame=sf)
    for read in (ds.get_all_data, lambda: ds.get_run_data(1),
                 lambda: ds.get_data(rows=[0, 3]), lambda: ds.get_data(cols=[1])):
        out = read()
        out += 1.0
    np.testing.assert_array_equal(ds.get_all_data(), np.zeros((4, 3)))


def test_concat_backend_routes_rows_and_refuses_a_part_in_another_space():
    a = MatrixBackend.from_masked(np.arange(12.0).reshape(4, 3), np.ones((3, 1, 1), bool),
                                  affine=np.eye(4))
    b = MatrixBackend.from_masked(100 + np.arange(6.0).reshape(2, 3),
                                  np.ones((3, 1, 1), bool), affine=np.eye(4))
    cat = ConcatBackend([a, b])
    full = np.vstack([a.get_data(), b.get_data()])
    rows = np.array([5, 0, 4, 2])
    np.testing.assert_array_equal(cat.get_data(rows=rows, cols=[2, 0]), full[rows][:, [2, 0]])
    assert cat.source_run_lengths == (4, 2)

    flipped = np.diag([-1.0, 1.0, 1.0, 1.0])
    c = MatrixBackend.from_masked(np.zeros((2, 3)), np.ones((3, 1, 1), bool), affine=flipped)
    with pytest.raises(ValueError, match="Run 1 is not in the same world space"):
        ConcatBackend([a, c])
    d = MatrixBackend.from_masked(np.zeros((2, 2)), np.array([True, True, False]).reshape(3, 1, 1),
                                  affine=np.eye(4))
    with pytest.raises(ValueError, match="Run 1 has a different mask"):
        ConcatBackend([a, d])


def test_user_written_runwise_sources_still_work():
    class Custom:
        def __init__(self):
            self.runs = [np.ones((3, 2)), 2 * np.ones((2, 2))]

        def get_data(self, run):
            return self.runs[run]

        def get_mask(self):
            return np.ones((2, 1, 1), dtype=bool)

        def get_affine(self):
            return np.diag([2.0, 2.0, 2.0, 1.0])

        def get_sampling_frame(self):
            return SamplingFrame(blocklens=[3, 2], tr=2.0)

        n_runs = 2
        n_timepoints = [3, 2]
        n_voxels = 2

    ds = FmriDataset(Custom())
    np.testing.assert_array_equal(ds.get_data(rows=[4, 0]), [[2.0, 2.0], [1.0, 1.0]])
    np.testing.assert_array_equal(ds.get_affine(), np.diag([2.0, 2.0, 2.0, 1.0]))
    assert ds.get_mask().shape == (2, 1, 1)


def test_volume_space_identity_rules():
    mask = np.array([True, False, True, True])
    known = VolumeSpace((2, 2, 1), mask, affine=np.eye(4), mask_shape=(2, 2, 1))
    unknown = VolumeSpace((2, 2, 1), mask)
    # Geometry is never assumed: unknown only matches unknown.
    assert not known.same_grid(unknown)
    assert unknown.same_grid(VolumeSpace((2, 2, 1), ~mask))
    assert known.mask_array().shape == (2, 2, 1)
    assert unknown.mask_array().shape == (4,)
    np.testing.assert_array_equal(known.voxel_ids, [0, 2, 3])
    vs = known.to_voxel_space()
    assert vs.n_samples == 3
    np.testing.assert_array_equal(vs.mask_idx, [0, 2, 3])
    with pytest.raises(ValueError, match="cover"):
        VolumeSpace((2, 2, 2), mask)
