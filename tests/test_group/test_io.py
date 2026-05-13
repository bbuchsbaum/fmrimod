"""Tests for native group-analysis HDF5 I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmrimod.group import (
    GDS_H5_VERSION,
    GroupSchemaError,
    SampleLabelSpace,
    SurfaceSpace,
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


def test_hdf5_roundtrip_axis_metadata_and_provenance(tmp_path) -> None:
    ds = group_dataset(
        {"beta": np.ones((2, 2, 1))},
        space=SampleLabelSpace(["r1", "r2"]),
        subjects=["s1", "s2"],
        contrasts=["task"],
        col_data=pd.DataFrame(
            {"age": [20, 30]},
            index=pd.Index(["s1", "s2"], name="subject"),
        ),
        row_data=pd.DataFrame(
            {"region": ["a", "b"]},
            index=pd.Index(["r1", "r2"], name="sample"),
        ),
        contrast_data=pd.DataFrame(
            {"kind": ["task"]},
            index=pd.Index(["task"], name="contrast"),
        ),
    )
    path = tmp_path / "axis_data.h5"

    write_hdf5(ds, path)
    with h5py.File(path, "r") as h5:
        assert "axis_data/subjects/json" in h5["gds"]
        assert "axis_data/samples/json" in h5["gds"]
        assert "axis_data/contrasts/json" in h5["gds"]
        assert "provenance/json" in h5["gds"]
    got = read_hdf5(path)

    assert got.col_data is not None
    assert got.row_data is not None
    assert got.contrast_data is not None
    assert got.col_data.index.name == "subject"
    assert got.row_data.index.name == "sample"
    assert got.contrast_data.index.name == "contrast"
    assert got.col_data["age"].tolist() == [20, 30]
    assert got.row_data["region"].tolist() == ["a", "b"]
    assert got.contrast_data["kind"].tolist() == ["task"]
    assert got.metadata["hdf5_provenance"]["writer"] == "fmrimod.group"
    assert got.metadata["hdf5_provenance"]["schema_version"] == GDS_H5_VERSION
    assert got.metadata["hdf5_provenance"]["assays"] == ["beta"]


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


def test_hdf5_roundtrip_surface_space(tmp_path) -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2]], dtype=np.intp)
    ds = group_dataset(
        {"beta": np.ones((3, 1, 1))},
        space=SurfaceSpace(vertices=vertices, faces=faces, hemi="L", template_id="fsaverage"),
        subjects=["s1"],
        contrasts=["c1"],
    )
    path = tmp_path / "surface.h5"

    write_hdf5(ds, path)
    got = read_hdf5(path)

    assert isinstance(got.space, SurfaceSpace)
    assert got.space.hemi == "L"
    assert got.space.template_id == "fsaverage"
    np.testing.assert_allclose(got.space.vertices, vertices)
    np.testing.assert_array_equal(got.space.faces, faces)


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


def test_hdf5_reader_can_preserve_r_serialized_alignments_as_opaque(tmp_path) -> None:
    ds = group_dataset(
        {"beta": np.ones((1, 1, 1))},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1"],
        contrasts=["c1"],
    )
    path = tmp_path / "opaque_alignment.h5"
    write_hdf5(ds, path)
    with h5py.File(path, "a") as h5:
        family = h5["gds"].create_group("alignments").create_group("native")
        family.create_dataset("serialized", data="A\n1\n262153\n")

    got = read_hdf5(path, allow_opaque_alignments=True)

    opaque = got.metadata["opaque_alignment_families"]["native"]
    assert opaque["format"] == "r-serialized-opaque"
    assert opaque["serialized"]["encoding"] == "utf-8"
    assert opaque["serialized"]["payload"].startswith("A\n1\n")


def test_hdf5_roundtrips_language_neutral_alignment_manifest(tmp_path) -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    ds = group_dataset(
        {"beta": np.ones((1, 1, 1))},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1"],
        contrasts=["c1"],
        metadata={
            "analysis": "alignment-demo",
            "alignment_families": {
                "native_to_mni": {
                    "description": "portable affine transform",
                    "entries": [
                        {
                            "name": "affine",
                            "kind": "affine",
                            "source_space": "native",
                            "target_space": "MNI152NLin2009cAsym",
                            "matrix": matrix,
                        }
                    ],
                }
            },
        },
    )
    path = tmp_path / "manifest_alignment.h5"

    write_hdf5(ds, path)
    with h5py.File(path, "r") as h5:
        assert h5["gds/version"][()].decode("utf-8") == "gds-h5/1.0"
        family = h5["gds/alignments/native_to_mni"]
        assert "manifest" in family
        assert "matrices/affine" in family
        np.testing.assert_allclose(family["matrices/affine"][()], matrix)
    got = read_hdf5(path)

    assert got.metadata["analysis"] == "alignment-demo"
    families = got.metadata["alignment_families"]
    family = families["native_to_mni"]
    assert family["format"] == "fmrimod.alignment_manifest"
    assert family["schema_version"] == "gds-h5/1.0-alignment"
    assert family["description"] == "portable affine transform"
    entry = family["entries"][0]
    assert entry["name"] == "affine"
    assert entry["kind"] == "affine"
    assert entry["source_space"] == "native"
    assert entry["target_space"] == "MNI152NLin2009cAsym"
    assert entry["matrix_dataset"] == "matrices/affine"
    np.testing.assert_allclose(entry["matrix"], matrix)


def test_hdf5_reader_accepts_scoped_legacy_v0_schema(tmp_path) -> None:
    ds = group_dataset(
        {"beta": np.ones((1, 1, 1))},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1"],
        contrasts=["c1"],
    )
    path = tmp_path / "legacy.h5"
    write_hdf5(ds, path)
    with h5py.File(path, "a") as h5:
        del h5["gds"]["version"]
        h5["gds"].create_dataset("version", data="gds-h5/0.1", dtype=h5py.string_dtype())

    got = read_hdf5(path)

    assert got.metadata["schema_version"] == "gds-h5/0.1"


def test_hdf5_reader_rejects_unsupported_schema_version(tmp_path) -> None:
    ds = group_dataset(
        {"beta": np.ones((1, 1, 1))},
        space=SampleLabelSpace(["r1"]),
        subjects=["s1"],
        contrasts=["c1"],
    )
    path = tmp_path / "future.h5"
    write_hdf5(ds, path)
    with h5py.File(path, "a") as h5:
        del h5["gds"]["version"]
        h5["gds"].create_dataset("version", data="gds-h5/2.0", dtype=h5py.string_dtype())

    with pytest.raises(GroupSchemaError, match="Unsupported GDS HDF5 schema version"):
        read_hdf5(path)
