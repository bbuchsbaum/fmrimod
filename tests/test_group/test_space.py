"""Tests for native group-analysis space descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.group import (
    BasisSpace,
    GroupSpaceError,
    ParcelSpace,
    SampleLabelSpace,
    SurfaceSpace,
    VoxelSpace,
    assert_compatible_spaces,
    common_mask,
)


def test_sample_label_space_subsets_by_integer_and_logical_indices() -> None:
    space = SampleLabelSpace(["a", "b", "c"])

    assert space.n_samples == 3
    assert space.subset([0, 2]).labels == ("a", "c")
    assert space.subset(np.array([True, False, True])).labels == ("a", "c")


def test_label_spaces_require_unique_non_empty_labels() -> None:
    with pytest.raises(GroupSpaceError, match="unique"):
        ParcelSpace(["roi", "roi"])
    with pytest.raises(GroupSpaceError, match="non-empty"):
        SampleLabelSpace([])


def test_voxel_space_dense_and_packed_sample_counts() -> None:
    dense = VoxelSpace((2, 3, 4), affine=np.eye(4))
    packed = VoxelSpace((2, 3, 4), mask_idx=[0, 5, 7], storage="packed")

    assert dense.n_samples == 24
    assert packed.n_samples == 3
    np.testing.assert_array_equal(packed.mask_idx, np.array([0, 5, 7]))


def test_voxel_space_validates_affine_storage_and_mask_indices() -> None:
    with pytest.raises(GroupSpaceError, match="4x4"):
        VoxelSpace((2, 2, 2), affine=np.eye(3))
    with pytest.raises(GroupSpaceError, match="packed"):
        VoxelSpace((2, 2, 2), storage="packed")
    with pytest.raises(GroupSpaceError, match="out of range"):
        VoxelSpace((2, 2, 2), mask_idx=[8], storage="packed")
    with pytest.raises(GroupSpaceError, match="unique"):
        VoxelSpace((2, 2, 2), mask_idx=[1, 1], storage="packed")


def test_voxel_space_subset_packs_dense_space_with_zero_based_indices() -> None:
    dense = VoxelSpace((2, 2, 1))
    subset = dense.subset([0, 3])

    assert subset.storage == "packed"
    np.testing.assert_array_equal(subset.mask_idx, np.array([0, 3]))


def test_surface_space_subsets_vertices_and_reindexes_faces() -> None:
    surface = SurfaceSpace(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        hemi="L",
    )
    subset = surface.subset([0, 1, 2])

    assert subset.n_samples == 3
    np.testing.assert_array_equal(subset.faces, np.array([[0, 1, 2]]))


def test_basis_space_validates_component_count() -> None:
    assert BasisSpace(3, basis_name="ICA").n_samples == 3
    with pytest.raises(GroupSpaceError, match="positive"):
        BasisSpace(0)


def test_assert_compatible_spaces_checks_kind_shape_and_affine() -> None:
    left = VoxelSpace((2, 2, 1), affine=np.eye(4))
    right = VoxelSpace((2, 2, 1), affine=np.eye(4))
    assert_compatible_spaces(left, right)

    with pytest.raises(GroupSpaceError, match="kinds differ"):
        assert_compatible_spaces(left, SampleLabelSpace(["a", "b", "c", "d"]))
    with pytest.raises(GroupSpaceError, match="shapes differ"):
        assert_compatible_spaces(left, VoxelSpace((2, 2, 2), affine=np.eye(4)))


def test_common_mask_returns_sample_indices_and_common_space() -> None:
    left = VoxelSpace((3, 1, 1), mask_idx=[0, 2], storage="packed")
    right = VoxelSpace((3, 1, 1), mask_idx=[1, 2], storage="packed")

    left_idx, right_idx, space = common_mask(left, right)
    np.testing.assert_array_equal(left_idx, np.array([1]))
    np.testing.assert_array_equal(right_idx, np.array([1]))
    np.testing.assert_array_equal(space.mask_idx, np.array([2]))

    left_idx, right_idx, union_space = common_mask(left, right, rule="union")
    np.testing.assert_array_equal(left_idx, np.array([0, 1]))
    np.testing.assert_array_equal(right_idx, np.array([0, 1]))
    np.testing.assert_array_equal(union_space.mask_idx, np.array([0, 1, 2]))

