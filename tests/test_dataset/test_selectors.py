"""Tests for canonical dataset spatial selectors."""

from __future__ import annotations

import numpy as np
import pytest

import fmridataset
import fmridataset.selectors as facade_selectors
import fmrimod
import fmrimod.dataset as dataset


def test_index_all_and_mask_selectors_resolve_masked_columns() -> None:
    mat = np.arange(40, dtype=float).reshape(10, 4)
    ds = fmrimod.matrix_dataset(mat, tr=2.0, mask=np.array([True, False, True, True]))

    np.testing.assert_array_equal(
        dataset.index_selector([0, 2]).resolve_indices(ds),
        np.array([0, 2]),
    )
    np.testing.assert_array_equal(
        dataset.all_selector().resolve_indices(ds),
        np.array([0, 1, 2]),
    )
    np.testing.assert_array_equal(
        dataset.mask_selector(np.array([True, False, True])).resolve_indices(ds),
        np.array([0, 2]),
    )
    with pytest.raises(IndexError, match="out of range"):
        dataset.index_selector([3]).resolve_indices(ds)


def test_full_volume_roi_and_voxel_selectors_intersect_dataset_mask() -> None:
    mat = np.arange(40, dtype=float).reshape(10, 4)
    ds = fmrimod.matrix_dataset(mat, tr=2.0, mask=np.array([True, False, True, True]))

    np.testing.assert_array_equal(
        dataset.roi_selector(np.array([False, True, True, True])).resolve_indices(ds),
        np.array([1, 2]),
    )
    np.testing.assert_array_equal(
        dataset.voxel_selector([[1, 1, 1], [3, 1, 1]]).resolve_indices(ds),
        np.array([0, 1]),
    )
    np.testing.assert_array_equal(
        dataset.sphere_selector([1, 1, 1], radius=0.1).resolve_indices(ds),
        np.array([0]),
    )


def test_selector_facade_reexports_canonical_objects() -> None:
    assert fmridataset.SeriesSelector is dataset.SeriesSelector
    assert fmridataset.IndexSelector is dataset.IndexSelector
    assert fmridataset.index_selector is dataset.index_selector
    assert fmridataset.resolve_indices is dataset.resolve_indices
    assert facade_selectors.MaskSelector is dataset.MaskSelector
    assert facade_selectors.sphere_selector is dataset.sphere_selector
