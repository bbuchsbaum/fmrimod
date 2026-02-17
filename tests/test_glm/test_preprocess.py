"""Parity-focused tests for GLM preprocessing helpers."""

import numpy as np
import pytest

from fmrimod.glm.preprocess import (
    compute_dvars,
    dvars_weights,
    extract_nuisance_timeseries,
)


def test_compute_dvars_matches_fmrireg_formula():
    # Two-voxel toy data with deterministic temporal differences.
    y = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [3.0, 3.0],
        ]
    )
    got = compute_dvars(y, normalize=True)

    # fmrireg formula:
    # dvars_raw = sqrt(rowMeans(diff(Y)^2))
    # dvars = c(median(dvars_raw), dvars_raw)
    # dvars = dvars / median(dvars)
    dvars_raw = np.array([1.0, 2.0])
    expected = np.array([1.5, 1.0, 2.0]) / 1.5
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_compute_dvars_requires_at_least_two_timepoints():
    y = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="at least 2 timepoints"):
        compute_dvars(y)


def test_dvars_weights_inverse_squared_matches_fmrireg_formula():
    dvars = np.array([1.0, 2.0, 3.0])
    got = dvars_weights(dvars, method="inverse_squared")
    expected = 1.0 / (1.0 + dvars**2)
    expected = np.clip(expected, 0.0, 1.0)
    expected = expected / np.mean(expected)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_dvars_weights_soft_threshold_matches_fmrireg_formula():
    dvars = np.array([0.5, 1.5, 3.0])
    threshold = 1.5
    steepness = 2.0
    got = dvars_weights(
        dvars,
        method="soft_threshold",
        threshold=threshold,
        steepness=steepness,
    )
    expected = 1.0 / (
        1.0 + ((np.maximum(dvars, threshold) - threshold) / threshold) ** steepness
    )
    expected = np.clip(expected, 0.0, 1.0)
    expected = expected / np.mean(expected)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_dvars_weights_tukey_matches_fmrireg_formula():
    dvars = np.array([0.5, 2.0, 4.0])
    threshold = 1.5
    got = dvars_weights(dvars, method="tukey", threshold=threshold)
    c_tukey = threshold * 2.0
    u = dvars / c_tukey
    expected = np.where(np.abs(u) <= 1.0, (1.0 - u**2) ** 2, 0.0)
    expected = np.clip(expected, 0.0, 1.0)
    expected = expected / np.mean(expected)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_dvars_weights_rejects_negative_dvars():
    with pytest.raises(ValueError, match="DVARS values must be non-negative"):
        dvars_weights(np.array([0.0, -0.1, 1.0]))


def test_extract_nuisance_timeseries_accepts_pathlike_mask_file(tmp_path):
    nib = pytest.importorskip("nibabel")

    y = np.arange(20, dtype=np.float64).reshape(5, 4)
    mask = np.array([[[1]], [[0]], [[1]], [[0]]], dtype=np.uint8)
    mask_path = tmp_path / "nuisance_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine=np.eye(4)), str(mask_path))

    nuisance = extract_nuisance_timeseries(y, mask_path)
    np.testing.assert_array_equal(nuisance, y[:, [0, 2]])


def test_extract_nuisance_timeseries_path_mask_maps_via_dataset_mask(tmp_path):
    nib = pytest.importorskip("nibabel")

    y = np.arange(15, dtype=np.float64).reshape(5, 3)
    # Full-space nuisance mask (4 voxels), then mapped through dataset_mask.
    nuisance_full = np.array([[[1]], [[0]], [[1]], [[0]]], dtype=np.uint8)
    dataset_mask = np.array([[[1]], [[0]], [[1]], [[1]]], dtype=bool)
    mask_path = tmp_path / "nuisance_full_mask.nii.gz"
    nib.save(nib.Nifti1Image(nuisance_full, affine=np.eye(4)), str(mask_path))

    nuisance = extract_nuisance_timeseries(y, mask_path, dataset_mask=dataset_mask)
    np.testing.assert_array_equal(nuisance, y[:, [0, 1]])
