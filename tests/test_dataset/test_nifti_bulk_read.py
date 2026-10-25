"""The NIfTI backend's bulk run read returns exactly what the per-volume read does."""

from __future__ import annotations

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

import fmrimod as fm  # noqa: E402


@pytest.fixture()
def runs(tmp_path):
    rng = np.random.default_rng(0)
    shape = (7, 6, 5)
    mask = np.zeros(shape, dtype=bool)
    mask[1:6, 1:5, 1:4] = True
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    imgs, paths = [], []
    for i, n in enumerate((30, 24)):
        data = rng.normal(size=shape + (n,)).astype(np.float32)
        img = nib.Nifti1Image(data, affine)
        path = tmp_path / f"run{i}.nii.gz"
        nib.save(img, path)
        imgs.append(img)
        paths.append(str(path))
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nib.save(mask_img, tmp_path / "mask.nii.gz")
    reference = np.vstack([np.asarray(i.get_fdata())[mask].T for i in imgs])
    return imgs, paths, mask_img, str(tmp_path / "mask.nii.gz"), reference


@pytest.mark.parametrize("on_disk", [False, True])
def test_full_partial_and_column_reads_match(runs, on_disk) -> None:
    imgs, paths, mask_img, mask_path, reference = runs
    ds = fm.fmri_dataset(
        paths if on_disk else imgs, mask=mask_path if on_disk else mask_img, tr=2.0
    )
    np.testing.assert_array_equal(ds.get_all_data(), reference)
    rows = np.r_[np.arange(2, 28), np.arange(31, 50)][::-1]  # bulk path, reordered
    np.testing.assert_array_equal(ds.get_data(rows=rows), reference[rows])
    few = np.array([0, 29, 30, 53])  # per-volume path
    np.testing.assert_array_equal(ds.get_data(rows=few), reference[few])
    cols = np.arange(0, reference.shape[1], 3)
    np.testing.assert_array_equal(ds.get_data(cols=cols), reference[:, cols])
