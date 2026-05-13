"""Tests for native group-analysis HDF5 I/O."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.group import (
    GDS_H5_VERSION,
    SampleLabelSpace,
    UnsupportedGroupFeatureError,
    VoxelSpace,
    group_dataset,
    read_hdf5,
    write_hdf5,
    write_out,
)

h5py = pytest.importorskip("h5py")


def test_hdf5_roundtrip_sample_label_space(tmp_path) -> None:
    ds = group_dataset(
        {
            "beta": np.array([[[1.0], [2.0]], [[3.0], [4.0]]]),
            "se": np.ones((2, 2, 1)),
        },
        space=SampleLabelSpace(["r1", "r2"]),
        subjects=["s1", "s2"],
        contrasts=["c1"],
        metadata={"analysis": "demo"},
    )
    path = tmp_path / "group.h5"

    write_hdf5(ds, path)
    got = read_hdf5(path)

    assert got.metadata["schema_version"] == GDS_H5_VERSION
    assert got.metadata["analysis"] == "demo"
    assert got.subjects == ("s1", "s2")
    assert got.contrasts == ("c1",)
    assert isinstance(got.space, SampleLabelSpace)
    assert got.space.labels == ("r1", "r2")
    np.testing.assert_allclose(got.assay("beta"), ds.assay("beta"))
    np.testing.assert_allclose(got.assay("se"), ds.assay("se"))


def test_write_out_dispatches_to_hdf5(tmp_path) -> None:
    ds = group_dataset(
        {"beta": np.ones((1, 1, 1))},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1"],
        contrasts=["c1"],
    )
    path = tmp_path / "out.h5"

    written = write_out(ds, path)

    assert written == path
    assert read_hdf5(path).shape == (1, 1, 1)


def test_hdf5_roundtrip_voxel_space_zero_based_mask(tmp_path) -> None:
    ds = group_dataset(
        {"beta": np.ones((2, 1, 1))},
        space=VoxelSpace((2, 2, 1), mask_idx=[0, 3], storage="packed"),
        subjects=["s1"],
        contrasts=["c1"],
    )
    path = tmp_path / "vox.h5"

    write_hdf5(ds, path)
    got = read_hdf5(path)

    assert isinstance(got.space, VoxelSpace)
    np.testing.assert_array_equal(got.space.mask_idx, np.array([0, 3]))
    assert got.space.storage == "packed"


def test_hdf5_reader_rejects_r_serialized_alignments(tmp_path) -> None:
    ds = group_dataset(
        {"beta": np.ones((1, 1, 1))},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1"],
        contrasts=["c1"],
    )
    path = tmp_path / "alignment.h5"
    write_hdf5(ds, path)
    with h5py.File(path, "a") as h5:
        family = h5["gds"].create_group("alignments").create_group("native")
        family.create_dataset("serialized", data="A\n1\n262153\n")

    with pytest.raises(UnsupportedGroupFeatureError, match="map families"):
        read_hdf5(path)
