"""Tests for canonical dataset mask helpers."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.dataset import mask_to_logical, mask_to_volume


def test_mask_to_logical_flattens_to_bool() -> None:
    mask = np.array([[1, 0], [2, 0]])

    np.testing.assert_array_equal(
        mask_to_logical(mask),
        np.array([True, False, True, False]),
    )


def test_mask_to_volume_validates_size() -> None:
    mask = np.array([True, False, True, False])

    np.testing.assert_array_equal(
        mask_to_volume(mask, (2, 2, 1)),
        np.array([[[True], [False]], [[True], [False]]]),
    )
    with pytest.raises(ValueError, match="Mask length"):
        mask_to_volume(mask, (2, 2, 2))


# R binarises masks with ``which(as.logical(x))`` (fmridataset
# read-bids-bold.R:373, nifti-array-source.R:192): every non-zero value is in,
# including negatives, and NaN is out. One fixture holds all four cases, so a
# ``> 0`` rule (drops -1) and an ``astype(bool)`` / ``!= 0`` rule (keeps NaN)
# both fail it.
_VALUES = np.array([0.0, 2.0, -1.0, np.nan, 0.5, 0.0], dtype=np.float64)
_EXPECTED = np.array([False, True, True, False, True, False])


def test_mask_to_logical_follows_r_as_logical() -> None:
    np.testing.assert_array_equal(mask_to_logical(_VALUES), _EXPECTED)
    np.testing.assert_array_equal(mask_to_logical(_EXPECTED), _EXPECTED)


def test_every_mask_input_path_uses_the_same_rule(tmp_path) -> None:
    from fmrimod.dataset import fmri_dataset, matrix_dataset, nifti_backend
    from fmrimod.dataset.adapters import NumpyAdapter
    from fmrimod.sampling import SamplingFrame

    n_in = int(_EXPECTED.sum())
    paths = {
        "matrix_dataset": lambda: matrix_dataset(np.ones((4, 6)), tr=2.0, mask=_VALUES),
        "fmri_dataset": lambda: fmri_dataset(np.ones((4, 6)), tr=2.0, mask=_VALUES),
        "NumpyAdapter": lambda: NumpyAdapter(
            np.ones((4, n_in)), SamplingFrame(blocklens=[4], tr=2.0),
            mask=_VALUES.reshape(6, 1, 1),
        ),
    }
    nib = pytest.importorskip("nibabel")
    vol = _VALUES.reshape(6, 1, 1)
    nib.save(nib.Nifti1Image(np.ones((6, 1, 1, 4)), np.eye(4)), tmp_path / "b.nii")
    nib.save(nib.Nifti1Image(vol, np.eye(4)), tmp_path / "m.nii")

    def nifti():
        be = nifti_backend(str(tmp_path / "b.nii"), str(tmp_path / "m.nii"))
        be.open()
        return be

    paths["nifti_backend"] = nifti
    neuroim = pytest.importorskip("neuroim")
    space = neuroim.NeuroSpace(dim=(6, 1, 1), spacing=(1, 1, 1), origin=(0, 0, 0))
    vec = neuroim.DenseNeuroVec(
        np.ones((6, 1, 1, 4)),
        neuroim.NeuroSpace(dim=(6, 1, 1, 4), spacing=(1, 1, 1, 1), origin=(0, 0, 0, 0)),
    )
    paths["NeuroVol mask"] = lambda: fmri_dataset(vec, mask=neuroim.DenseNeuroVol(vol, space), tr=2.0)
    paths["array mask on NeuroVec"] = lambda: fmri_dataset(vec, mask=vol, tr=2.0)

    for name, build in paths.items():
        source = build()
        mask = np.asarray(source.get_mask()).reshape(-1)
        np.testing.assert_array_equal(mask, _EXPECTED, err_msg=name)
