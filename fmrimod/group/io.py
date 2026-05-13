"""HDF5 I/O for native group-analysis datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import h5py  # type: ignore[import-untyped]
import numpy as np

from .dataset import GroupDataset
from .errors import GroupSchemaError, UnsupportedGroupFeatureError
from .space import (
    BasisSpace,
    GroupSpace,
    ParcelSpace,
    SampleLabelSpace,
    StorageMode,
    VoxelSpace,
)

GDS_H5_VERSION = "gds-h5/0.1"


def _string_dtype() -> h5py.Datatype:
    return h5py.string_dtype(encoding="utf-8")


def _write_strings(group: h5py.Group, name: str, values: list[str]) -> None:
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=_string_dtype())


def _read_strings(group: h5py.Group, name: str) -> tuple[str, ...]:
    raw = group[name][()]
    return tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in np.atleast_1d(raw)
    )


def _write_space(group: h5py.Group, space: GroupSpace) -> None:
    group.attrs["type"] = space.kind
    if isinstance(space, VoxelSpace):
        voxel = group.create_group("voxel")
        voxel.create_dataset("dim", data=np.asarray(space.shape, dtype=np.int64))
        voxel.create_dataset("affine", data=np.asarray(space.affine, dtype=np.float64))
        voxel.attrs["storage"] = space.storage
        voxel.attrs["index_base"] = 0
        if space.template_id is not None:
            voxel.attrs["template_id"] = space.template_id
        if space.mask_idx is not None:
            voxel.create_dataset("mask_idx", data=np.asarray(space.mask_idx, dtype=np.int64))
    elif isinstance(space, ParcelSpace):
        parcels = group.create_group("parcels")
        _write_strings(parcels, "labels", list(space.labels))
    elif isinstance(space, SampleLabelSpace):
        sample_labels = group.create_group("sample_labels")
        _write_strings(sample_labels, "labels", list(space.labels))
    elif isinstance(space, BasisSpace):
        basis = group.create_group("basis")
        basis.attrs["k"] = int(space.n_components)
        if space.basis_name is not None:
            basis.attrs["basis_name"] = space.basis_name
    else:
        raise UnsupportedGroupFeatureError(
            f"HDF5 writer does not yet support space kind '{space.kind}'"
        )


def _read_space(group: h5py.Group) -> GroupSpace:
    kind_raw = group.attrs.get("type")
    kind = kind_raw.decode("utf-8") if isinstance(kind_raw, bytes) else str(kind_raw)
    if kind == "voxel":
        voxel = group["voxel"]
        index_base = int(voxel.attrs.get("index_base", 1))
        mask_idx = None
        if "mask_idx" in voxel:
            mask_idx = np.asarray(voxel["mask_idx"][()], dtype=np.intp)
            if index_base == 1:
                mask_idx = mask_idx - 1
        template_id = voxel.attrs.get("template_id")
        if isinstance(template_id, bytes):
            template_id = template_id.decode("utf-8")
        storage_raw = voxel.attrs.get("storage", "dense")
        storage = storage_raw.decode("utf-8") if isinstance(storage_raw, bytes) else str(storage_raw)
        if storage not in ("dense", "packed"):
            raise GroupSchemaError(f"Unsupported voxel storage mode: {storage}")
        return VoxelSpace(
            shape=tuple(int(x) for x in voxel["dim"][()]),
            affine=np.asarray(voxel["affine"][()], dtype=np.float64),
            mask_idx=mask_idx,
            storage=cast(StorageMode, storage),
            template_id=template_id,
        )
    if kind == "parcels":
        return ParcelSpace(_read_strings(group["parcels"], "labels"))
    if kind == "sample_labels":
        return SampleLabelSpace(_read_strings(group["sample_labels"], "labels"))
    if kind == "basis":
        basis = group["basis"]
        basis_name = basis.attrs.get("basis_name")
        if isinstance(basis_name, bytes):
            basis_name = basis_name.decode("utf-8")
        return BasisSpace(int(basis.attrs["k"]), basis_name=basis_name)
    raise GroupSchemaError(f"Unsupported HDF5 space type: {kind}")


def write_hdf5(dataset: GroupDataset, path: str | Path) -> Path:
    """Write a scoped ``/gds`` HDF5 file."""
    out = Path(path)
    with h5py.File(out, "w") as h5:
        gds = h5.create_group("gds")
        gds.create_dataset("version", data=GDS_H5_VERSION, dtype=_string_dtype())
        axes = gds.create_group("axes")
        _write_strings(axes, "subjects", list(dataset.subjects))
        _write_strings(axes, "contrasts", list(dataset.contrasts))

        space = gds.create_group("space")
        _write_space(space, dataset.space)

        assays = gds.create_group("assays")
        for name, arr in dataset.assays.items():
            assays.create_dataset(name, data=np.asarray(arr, dtype=np.float64))

        metadata = gds.create_group("metadata")
        metadata.create_dataset(
            "json",
            data=json.dumps(dict(dataset.metadata), sort_keys=True, default=str),
            dtype=_string_dtype(),
        )
    return out


def read_hdf5(path: str | Path, *, allow_opaque_alignments: bool = False) -> GroupDataset:
    """Read a scoped ``/gds`` HDF5 file."""
    with h5py.File(path, "r") as h5:
        if "gds" not in h5:
            raise GroupSchemaError("HDF5 file does not contain /gds")
        gds = h5["gds"]
        version_raw = gds["version"][()]
        version = version_raw.decode("utf-8") if isinstance(version_raw, bytes) else str(version_raw)
        if not version.startswith("gds-h5/0."):
            raise GroupSchemaError(f"Unsupported GDS HDF5 schema version: {version}")
        if (
            "alignments" in gds
            and len(gds["alignments"]) > 0
            and not allow_opaque_alignments
        ):
            raise UnsupportedGroupFeatureError(
                "R-serialized map families in /gds/alignments are not semantically supported"
            )

        subjects = _read_strings(gds["axes"], "subjects")
        contrasts = _read_strings(gds["axes"], "contrasts")
        space = _read_space(gds["space"])
        assays = {
            name: np.asarray(gds["assays"][name][()], dtype=np.float64)
            for name in gds["assays"]
        }
        metadata: dict[str, Any] = {}
        if "metadata" in gds and "json" in gds["metadata"]:
            raw = gds["metadata"]["json"][()]
            text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            metadata = json.loads(text)
        metadata["schema_version"] = version

    return GroupDataset(
        assays=assays,
        space=space,
        subjects=subjects,
        contrasts=contrasts,
        metadata=metadata,
    )
