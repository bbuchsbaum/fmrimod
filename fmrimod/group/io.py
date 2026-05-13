"""HDF5 I/O for native group-analysis datasets."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .dataset import GroupDataset
from .errors import GroupSchemaError, UnsupportedGroupFeatureError
from .space import (
    BasisSpace,
    GroupSpace,
    Hemisphere,
    ParcelSpace,
    SampleLabelSpace,
    StorageMode,
    SurfaceSpace,
    VoxelSpace,
)

GDS_H5_VERSION = "gds-h5/1.0"
GDS_H5_SUPPORTED_PREFIXES = ("gds-h5/0.", "gds-h5/1.")
ALIGNMENT_FAMILIES_METADATA_KEY = "alignment_families"
OPAQUE_ALIGNMENT_FAMILIES_METADATA_KEY = "opaque_alignment_families"
ALIGNMENT_MANIFEST_FORMAT = "fmrimod.alignment_manifest"
ALIGNMENT_MANIFEST_VERSION = "gds-h5/1.0-alignment"


def _import_h5py() -> Any:
    try:
        import h5py  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - optional dependency behavior
        raise UnsupportedGroupFeatureError(
            "native group HDF5 I/O requires optional dependency 'h5py'"
        ) from exc
    return h5py


def _string_dtype() -> Any:
    return _import_h5py().string_dtype(encoding="utf-8")


def _write_strings(group: Any, name: str, values: list[str]) -> None:
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=_string_dtype())


def _read_strings(group: Any, name: str) -> tuple[str, ...]:
    raw = group[name][()]
    return tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in np.atleast_1d(raw)
    )


def _read_text_dataset(group: Any, name: str) -> str:
    raw = group[name][()]
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _write_json_dataset(group: Any, name: str, value: Any) -> None:
    group.create_dataset(
        name,
        data=json.dumps(value, sort_keys=True, default=str),
        dtype=_string_dtype(),
    )


def _write_axis_frame(parent: Any, name: str, frame: pd.DataFrame | None) -> None:
    if frame is None:
        return
    group = parent.create_group(name)
    group.attrs["orient"] = "split"
    if frame.index.name is not None:
        group.attrs["index_name"] = str(frame.index.name)
    group.create_dataset(
        "json",
        data=frame.to_json(orient="split", date_format="iso"),
        dtype=_string_dtype(),
    )


def _read_axis_frame(parent: Any, name: str) -> pd.DataFrame | None:
    if name not in parent:
        return None
    group = parent[name]
    orient_raw = group.attrs.get("orient", "split")
    orient = orient_raw.decode("utf-8") if isinstance(orient_raw, bytes) else str(orient_raw)
    if orient != "split":
        raise GroupSchemaError(f"Unsupported axis_data JSON orient: {orient}")
    frame = pd.read_json(StringIO(_read_text_dataset(group, "json")), orient="split")
    index_name = group.attrs.get("index_name")
    if isinstance(index_name, bytes):
        index_name = index_name.decode("utf-8")
    if index_name is not None:
        frame.index.name = str(index_name)
    return frame


def _provenance_payload(dataset: GroupDataset) -> dict[str, Any]:
    return {
        "writer": "fmrimod.group",
        "schema_version": GDS_H5_VERSION,
        "assays": sorted(dataset.assays),
        "shape": list(dataset.shape),
        "space_kind": dataset.space.kind,
        "n_subjects": len(dataset.subjects),
        "n_contrasts": len(dataset.contrasts),
    }


def _read_schema_version(gds: Any) -> str:
    if "version" not in gds:
        raise GroupSchemaError("HDF5 /gds is missing schema version")
    version = _read_text_dataset(gds, "version")
    if not any(version.startswith(prefix) for prefix in GDS_H5_SUPPORTED_PREFIXES):
        raise GroupSchemaError(f"Unsupported GDS HDF5 schema version: {version}")
    return version


def _metadata_payload(dataset: GroupDataset) -> dict[str, Any]:
    metadata = dict(dataset.metadata)
    metadata.pop(ALIGNMENT_FAMILIES_METADATA_KEY, None)
    metadata.pop(OPAQUE_ALIGNMENT_FAMILIES_METADATA_KEY, None)
    return metadata


def _validate_hdf5_name(value: Any, *, kind: str) -> str:
    name = str(value)
    if not name or "/" in name:
        raise GroupSchemaError(f"{kind} name must be non-empty and must not contain '/'")
    return name


def _json_safe(value: Any, *, context: str) -> Any:
    try:
        json.dumps(value)
    except TypeError as exc:
        raise GroupSchemaError(f"{context} must be JSON-serializable") from exc
    return value


def _write_alignment_families(group: Any, dataset: GroupDataset) -> None:
    raw = dataset.metadata.get(ALIGNMENT_FAMILIES_METADATA_KEY)
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise GroupSchemaError("metadata['alignment_families'] must be a mapping")

    alignments = group.create_group("alignments")
    alignments.attrs["format"] = ALIGNMENT_MANIFEST_FORMAT
    alignments.attrs["schema_version"] = GDS_H5_VERSION
    for family_key, family_raw in raw.items():
        family_name = _validate_hdf5_name(family_key, kind="alignment family")
        if not isinstance(family_raw, Mapping):
            raise GroupSchemaError(f"alignment family '{family_name}' must be a mapping")

        entries_raw = family_raw.get("entries", ())
        if isinstance(entries_raw, (str, bytes)) or not isinstance(entries_raw, Sequence):
            raise GroupSchemaError(
                f"alignment family '{family_name}' entries must be a sequence"
            )

        family = alignments.create_group(family_name)
        family.attrs["format"] = ALIGNMENT_MANIFEST_FORMAT
        family.attrs["schema_version"] = ALIGNMENT_MANIFEST_VERSION
        matrices = family.create_group("matrices")
        manifest: dict[str, Any] = {
            "schema_version": ALIGNMENT_MANIFEST_VERSION,
            "format": ALIGNMENT_MANIFEST_FORMAT,
            "family": family_name,
            "entries": [],
        }
        for key, value in family_raw.items():
            key_str = str(key)
            if key_str in {"entries", "schema_version", "format", "family"}:
                continue
            manifest[key_str] = _json_safe(
                value,
                context=f"alignment family '{family_name}' field '{key_str}'",
            )

        seen_entries: set[str] = set()
        manifest_entries = cast(list[dict[str, Any]], manifest["entries"])
        for idx, entry_raw in enumerate(entries_raw):
            if not isinstance(entry_raw, Mapping):
                raise GroupSchemaError(
                    f"alignment family '{family_name}' entry {idx} must be a mapping"
                )
            if "matrix" not in entry_raw:
                raise GroupSchemaError(
                    f"alignment family '{family_name}' entry {idx} is missing matrix"
                )
            entry_name = _validate_hdf5_name(
                entry_raw.get("name", f"matrix_{idx}"),
                kind=f"alignment family '{family_name}' entry",
            )
            if entry_name in seen_entries:
                raise GroupSchemaError(
                    f"alignment family '{family_name}' has duplicate entry '{entry_name}'"
                )
            seen_entries.add(entry_name)

            matrix = np.asarray(entry_raw["matrix"], dtype=np.float64)
            if matrix.ndim != 2:
                raise GroupSchemaError(
                    f"alignment family '{family_name}' entry '{entry_name}' "
                    "matrix must be 2-D"
                )
            matrices.create_dataset(entry_name, data=matrix)

            manifest_entry: dict[str, Any] = {}
            for key, value in entry_raw.items():
                key_str = str(key)
                if key_str == "matrix":
                    continue
                manifest_entry[key_str] = _json_safe(
                    value,
                    context=(
                        f"alignment family '{family_name}' entry '{entry_name}' "
                        f"field '{key_str}'"
                    ),
                )
            manifest_entry["name"] = entry_name
            manifest_entry["matrix_dataset"] = f"matrices/{entry_name}"
            manifest_entry["shape"] = list(matrix.shape)
            manifest_entry["dtype"] = "float64"
            manifest_entries.append(manifest_entry)

        _write_json_dataset(family, "manifest", manifest)


def _read_opaque_dataset(dataset: Any) -> dict[str, Any]:
    raw = dataset[()]
    if isinstance(raw, bytes):
        try:
            return {"encoding": "utf-8", "payload": raw.decode("utf-8")}
        except UnicodeDecodeError:
            return {
                "encoding": "base64",
                "payload": base64.b64encode(raw).decode("ascii"),
            }
    if isinstance(raw, str):
        return {"encoding": "utf-8", "payload": raw}

    arr = np.asarray(raw)
    return {
        "encoding": "base64",
        "payload": base64.b64encode(arr.tobytes()).decode("ascii"),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def _read_alignment_families(
    group: Any,
    *,
    allow_opaque_alignments: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if "alignments" not in group or len(group["alignments"]) == 0:
        return None, None

    alignment_families: dict[str, Any] = {}
    opaque_families: dict[str, Any] = {}
    alignments = group["alignments"]
    for family_name in alignments:
        family = alignments[family_name]
        if "manifest" in family:
            manifest = json.loads(_read_text_dataset(family, "manifest"))
            if manifest.get("format") != ALIGNMENT_MANIFEST_FORMAT:
                raise GroupSchemaError(
                    f"Unsupported alignment manifest format for family '{family_name}'"
                )
            if manifest.get("schema_version") != ALIGNMENT_MANIFEST_VERSION:
                raise GroupSchemaError(
                    f"Unsupported alignment manifest version for family '{family_name}'"
                )
            entries_raw = manifest.get("entries", [])
            if isinstance(entries_raw, (str, bytes)) or not isinstance(
                entries_raw,
                Sequence,
            ):
                raise GroupSchemaError(
                    f"alignment manifest entries for family '{family_name}' "
                    "must be a sequence"
                )
            entries: list[dict[str, Any]] = []
            for idx, entry_raw in enumerate(entries_raw):
                if not isinstance(entry_raw, Mapping):
                    raise GroupSchemaError(
                        f"alignment manifest entry {idx} for family "
                        f"'{family_name}' must be a mapping"
                    )
                matrix_dataset = entry_raw.get("matrix_dataset")
                if not isinstance(matrix_dataset, str):
                    raise GroupSchemaError(
                        f"alignment manifest entry {idx} for family "
                        f"'{family_name}' is missing matrix_dataset"
                    )
                try:
                    matrix_raw = family[matrix_dataset][()]
                except KeyError as exc:
                    raise GroupSchemaError(
                        f"alignment matrix dataset '{matrix_dataset}' is missing "
                        f"for family '{family_name}'"
                    ) from exc
                entry = dict(entry_raw)
                entry["matrix"] = np.asarray(matrix_raw, dtype=np.float64)
                entries.append(entry)
            family_manifest = dict(manifest)
            family_manifest["entries"] = entries
            alignment_families[str(family_name)] = family_manifest
            continue

        if "serialized" in family:
            if not allow_opaque_alignments:
                raise UnsupportedGroupFeatureError(
                    "R-serialized map families in /gds/alignments are not "
                    "semantically supported; pass allow_opaque_alignments=True "
                    "to preserve the raw payload as metadata"
                )
            opaque_families[str(family_name)] = {
                "format": "r-serialized-opaque",
                "serialized": _read_opaque_dataset(family["serialized"]),
            }
            continue

        raise UnsupportedGroupFeatureError(
            f"Unsupported alignment family '{family_name}' in /gds/alignments"
        )

    return alignment_families or None, opaque_families or None


def _write_space(group: Any, space: GroupSpace) -> None:
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
    elif isinstance(space, SurfaceSpace):
        surface = group.create_group("surface")
        surface.create_dataset(
            "vertices",
            data=np.asarray(space.vertices, dtype=np.float64),
        )
        surface.create_dataset("faces", data=np.asarray(space.faces, dtype=np.int64))
        surface.attrs["hemi"] = space.hemi
        if space.template_id is not None:
            surface.attrs["template_id"] = space.template_id
    else:
        raise UnsupportedGroupFeatureError(
            f"HDF5 writer does not yet support space kind '{space.kind}'"
        )


def _read_space(group: Any) -> GroupSpace:
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
    if kind == "surface":
        surface = group["surface"]
        hemi = surface.attrs["hemi"]
        if isinstance(hemi, bytes):
            hemi = hemi.decode("utf-8")
        template_id = surface.attrs.get("template_id")
        if isinstance(template_id, bytes):
            template_id = template_id.decode("utf-8")
        return SurfaceSpace(
            vertices=np.asarray(surface["vertices"][()], dtype=np.float64),
            faces=np.asarray(surface["faces"][()], dtype=np.intp),
            hemi=cast(Hemisphere, str(hemi)),
            template_id=template_id,
        )
    raise GroupSchemaError(f"Unsupported HDF5 space type: {kind}")


def write_hdf5(dataset: GroupDataset, path: str | Path) -> Path:
    """Write a scoped ``/gds`` HDF5 file."""
    h5py = _import_h5py()
    out = Path(path)
    with h5py.File(out, "w") as h5:
        gds = h5.create_group("gds")
        gds.create_dataset("version", data=GDS_H5_VERSION, dtype=_string_dtype())
        axes = gds.create_group("axes")
        _write_strings(axes, "subjects", list(dataset.subjects))
        _write_strings(axes, "contrasts", list(dataset.contrasts))

        axis_data = gds.create_group("axis_data")
        _write_axis_frame(axis_data, "subjects", dataset.col_data)
        _write_axis_frame(axis_data, "samples", dataset.row_data)
        _write_axis_frame(axis_data, "contrasts", dataset.contrast_data)

        space = gds.create_group("space")
        _write_space(space, dataset.space)

        assays = gds.create_group("assays")
        for name, arr in dataset.assays.items():
            assays.create_dataset(name, data=np.asarray(arr, dtype=np.float64))

        _write_alignment_families(gds, dataset)

        metadata = gds.create_group("metadata")
        _write_json_dataset(metadata, "json", _metadata_payload(dataset))

        provenance = gds.create_group("provenance")
        _write_json_dataset(provenance, "json", _provenance_payload(dataset))
    return out


def read_hdf5(path: str | Path, *, allow_opaque_alignments: bool = False) -> GroupDataset:
    """Read a scoped ``/gds`` HDF5 file."""
    h5py = _import_h5py()
    with h5py.File(path, "r") as h5:
        if "gds" not in h5:
            raise GroupSchemaError("HDF5 file does not contain /gds")
        gds = h5["gds"]
        version = _read_schema_version(gds)
        alignment_families, opaque_alignment_families = _read_alignment_families(
            gds,
            allow_opaque_alignments=allow_opaque_alignments,
        )

        subjects = _read_strings(gds["axes"], "subjects")
        contrasts = _read_strings(gds["axes"], "contrasts")
        axis_data = gds["axis_data"] if "axis_data" in gds else None
        col_data = None if axis_data is None else _read_axis_frame(axis_data, "subjects")
        row_data = None if axis_data is None else _read_axis_frame(axis_data, "samples")
        contrast_data = None if axis_data is None else _read_axis_frame(axis_data, "contrasts")
        space = _read_space(gds["space"])
        assays = {
            name: np.asarray(gds["assays"][name][()], dtype=np.float64)
            for name in gds["assays"]
        }
        metadata: dict[str, Any] = {}
        if "metadata" in gds and "json" in gds["metadata"]:
            metadata = json.loads(_read_text_dataset(gds["metadata"], "json"))
        metadata["schema_version"] = version
        if "provenance" in gds and "json" in gds["provenance"]:
            metadata["hdf5_provenance"] = json.loads(
                _read_text_dataset(gds["provenance"], "json")
            )
        if alignment_families is not None:
            metadata[ALIGNMENT_FAMILIES_METADATA_KEY] = alignment_families
        if opaque_alignment_families is not None:
            metadata[OPAQUE_ALIGNMENT_FAMILIES_METADATA_KEY] = opaque_alignment_families

    return GroupDataset(
        assays=assays,
        space=space,
        subjects=subjects,
        contrasts=contrasts,
        col_data=col_data,
        row_data=row_data,
        contrast_data=contrast_data,
        metadata=metadata,
    )
