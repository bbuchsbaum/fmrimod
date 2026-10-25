"""Contracts for the optional NIfTI storage backend."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.dataset import BackendIOError, NiftiBackend, create_backend, nifti_backend

nib = pytest.importorskip("nibabel")


@pytest.fixture()
def nifti_full_mask(tmp_path):
    rng = np.random.default_rng(20260514)
    data = rng.standard_normal((3, 3, 3, 10)).astype(np.float32)
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    data_path = tmp_path / "data_full.nii.gz"
    mask_path = tmp_path / "mask_full.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(data_path))
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))
    return data_path, mask_path, data


@pytest.fixture()
def nifti_partial_mask(tmp_path):
    rng = np.random.default_rng(20260515)
    data = rng.standard_normal((3, 3, 3, 10)).astype(np.float32)
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    mask[0, 0, 0] = 0
    data_path = tmp_path / "data_partial.nii.gz"
    mask_path = tmp_path / "mask_partial.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(data_path))
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))
    return data_path, mask_path, data, mask.astype(bool)


def test_nifti_backend_open_dims_and_metadata(nifti_full_mask) -> None:
    data_path, mask_path, _data = nifti_full_mask
    backend = NiftiBackend(source=data_path, mask_source=mask_path)
    backend.open()

    dims = backend.get_dims()
    assert dims.spatial == (3, 3, 3)
    assert dims.time == 10
    assert backend.get_metadata()["format"] == "nifti"
    assert backend.get_metadata()["source"] == [str(data_path)]
    backend.close()


def test_nifti_backend_lazy_subset_read(nifti_full_mask) -> None:
    data_path, mask_path, data = nifti_full_mask
    backend = nifti_backend(source=data_path, mask_source=mask_path, preload=False)
    backend.open()
    assert backend._data is None

    rows = np.array([2, 0], dtype=np.intp)
    cols = np.array([0, 2], dtype=np.intp)
    observed = backend.get_data(rows=rows, cols=cols)
    expected = data.reshape(-1, data.shape[-1]).T[rows][:, cols]

    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)
    assert backend._data is None
    backend.close()


def test_nifti_backend_preload_subset_read_matches_lazy(nifti_full_mask) -> None:
    data_path, mask_path, data = nifti_full_mask
    backend = nifti_backend(source=data_path, mask_source=mask_path, preload=True)
    backend.open()
    assert backend._data is not None

    rows = np.array([2, 0], dtype=np.intp)
    cols = np.array([0, 2], dtype=np.intp)
    observed = backend.get_data(rows=rows, cols=cols)
    expected = data.reshape(-1, data.shape[-1]).T[rows][:, cols]

    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)
    backend.close()


def test_nifti_backend_partial_mask(nifti_partial_mask) -> None:
    data_path, mask_path, data, mask = nifti_partial_mask
    backend = NiftiBackend(source=data_path, mask_source=mask_path)
    backend.open()

    observed = backend.get_data()
    expected = data.reshape(-1, data.shape[-1]).T[:, mask.reshape(-1)]

    assert observed.shape == (10, int(mask.sum()))
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(backend.get_mask(), mask.reshape(-1))
    backend.close()


def test_nifti_backend_multiple_sources_stack_time(tmp_path) -> None:
    first = np.arange(2 * 2 * 1 * 3, dtype=np.float32).reshape(2, 2, 1, 3)
    second = first + 100
    mask = np.ones((2, 2, 1), dtype=np.uint8)
    p1 = tmp_path / "run1.nii.gz"
    p2 = tmp_path / "run2.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(first, np.eye(4)), str(p1))
    nib.save(nib.Nifti1Image(second, np.eye(4)), str(p2))
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))

    backend = NiftiBackend(source=[p1, p2], mask_source=mask_path)
    backend.open()

    expected = np.vstack(
        [
            first.reshape(-1, first.shape[-1]).T,
            second.reshape(-1, second.shape[-1]).T,
        ]
    )
    np.testing.assert_array_equal(backend.get_data(), expected)
    assert backend.get_dims().time == 6
    backend.close()


def test_nifti_backend_errors_for_missing_or_mismatched_inputs(
    tmp_path,
    nifti_full_mask,
) -> None:
    data_path, mask_path, _data = nifti_full_mask
    missing = NiftiBackend(source=tmp_path / "missing.nii.gz", mask_source=mask_path)
    with pytest.raises(BackendIOError, match="Source file not found"):
        missing.open()

    bad_data = tmp_path / "bad_shape.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 4)), np.eye(4)), str(bad_data))
    mismatched = NiftiBackend(source=bad_data, mask_source=mask_path)
    with pytest.raises(BackendIOError, match="does not match mask"):
        mismatched.open()

    missing_mask = NiftiBackend(
        source=data_path,
        mask_source=tmp_path / "missing_mask.nii.gz",
    )
    with pytest.raises(BackendIOError, match="Mask file not found"):
        missing_mask.open()


def test_nifti_backend_registry_constructs_closed_backend(nifti_full_mask) -> None:
    data_path, mask_path, _data = nifti_full_mask
    backend = create_backend(
        "nifti",
        validate=False,
        source=data_path,
        mask_source=mask_path,
    )

    assert isinstance(backend, NiftiBackend)
    backend.open()
    assert backend.validate()
    backend.close()


def test_fmridataset_nifti_backend_facade_is_identity_alias() -> None:
    import fmridataset.backends.nifti_backend as facade
    from fmrimod.dataset.backends.nifti_backend import (
        nifti_backend as backend_nifti_backend,
    )

    assert facade.NiftiBackend is NiftiBackend
    assert facade.nifti_backend is backend_nifti_backend


def _write_nifti(path, arr, affine):
    nib = pytest.importorskip("nibabel")
    nib.save(nib.Nifti1Image(arr, affine), path)
    return str(path)


def test_backend_dataset_checks_run_lengths_per_file(tmp_path) -> None:
    # fmridataset 631cf0d (#81): run_length is checked against each file and
    # the error names the run. Inventory v2 DS-10. The fixture's totals agree
    # (13 == 13), so only a per-file comparison fails it. Runs are 0-based in
    # fmrimod messages, so the first mismatching run is "Run 0".
    from fmrimod.dataset import FmriDataset
    from fmrimod.dataset.adapters import BackendAdapter
    from fmrimod.sampling import SamplingFrame

    aff = np.eye(4)
    paths = [_write_nifti(tmp_path / f"run{i}.nii", np.full((2, 2, 2, n), float(i)), aff)
             for i, n in enumerate((6, 7))]
    mask = _write_nifti(tmp_path / "mask.nii", np.ones((2, 2, 2), "uint8"), aff)
    be = nifti_backend(paths, mask)
    be.open()
    assert be.source_run_lengths == (6, 7)
    with pytest.raises(ValueError, match="Run 0 has 7"):
        BackendAdapter(be, SamplingFrame(blocklens=[7, 6], tr=2.0))
    with pytest.raises(ValueError, match="Run 0 has 7"):
        FmriDataset(be, sampling_frame=SamplingFrame(blocklens=[7, 6], tr=2.0))
    ds = FmriDataset(be, sampling_frame=SamplingFrame(blocklens=[6, 7], tr=2.0))
    assert ds.get_run_data(1)[:, 0].tolist() == [1.0] * 7
    # One run spanning both files is a different split, not a per-file one.
    FmriDataset(be, sampling_frame=SamplingFrame(blocklens=[13], tr=2.0))


def test_backend_dataset_keeps_the_nifti_space(tmp_path) -> None:
    # fmridataset 7dad6c0: a NIfTI-backed dataset keeps the backend's space,
    # so results can go back to the grid. Inventory v2 DS-11. The header
    # affine flips x, so an identity affine cannot pass.
    from fmrimod.dataset import FmriDataset
    from fmrimod.glm.spatial import SpatialContext
    from fmrimod.sampling import SamplingFrame

    aff = np.array([[-2.0, 0, 0, 90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]])
    p = _write_nifti(tmp_path / "b.nii", np.random.default_rng(0).normal(size=(3, 3, 2, 5)), aff)
    m = _write_nifti(tmp_path / "m.nii", np.ones((3, 3, 2), "uint8"), aff)
    be = nifti_backend(p, m)
    be.open()
    ds = FmriDataset(be, sampling_frame=SamplingFrame(blocklens=[5], tr=2.0))
    np.testing.assert_allclose(ds.get_affine(), aff)
    assert ds.get_mask().shape == (3, 3, 2)
    ctx = SpatialContext.from_dataset(ds)
    assert ctx is not None
    np.testing.assert_allclose(ctx.affine, aff)


def test_nifti_backend_refuses_a_source_in_another_space(tmp_path) -> None:
    # Spatial identity (DS-07) on the NIfTI path: same shape, flipped affine.
    aff = np.eye(4) * 2.0
    aff[3, 3] = 1.0
    flipped = aff.copy()
    flipped[0, 0] = -2.0
    p = _write_nifti(tmp_path / "b.nii", np.ones((2, 2, 2, 3)), flipped)
    m = _write_nifti(tmp_path / "m.nii", np.ones((2, 2, 2), "uint8"), aff)
    with pytest.raises(BackendIOError, match="space"):
        nifti_backend(p, m).open()
