"""BIDS-HDF5 study dataset reader."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fmrimod.sampling import SamplingFrame

from .backends.bids_h5_backend import (
    BidsH5ScanBackend,
    CompressionMode,
    SharedH5Connection,
    _decode_h5_value,
    bids_h5_scan_backend,
    h5_shared_connection,
)
from .errors import BackendIOError, FmriDatasetError
from .fmri_dataset import FmriDataset


def _as_list(value: str | Iterable[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _read_dataset(handle: object, path: str) -> object:
    if path not in handle:
        raise BackendIOError(
            f"Required dataset '{path}' not found",
            operation="read",
        )
    return _decode_h5_value(handle[path][()])


def _read_optional_scalar(handle: object, path: str) -> object | None:
    if path not in handle:
        return None
    return _decode_h5_value(handle[path][()])


def _read_scan_index(handle: object) -> pd.DataFrame:
    if "scan_index" not in handle:
        raise BackendIOError(
            "BIDS-HDF5 archive is missing /scan_index/",
            operation="read",
        )

    group = handle["scan_index"]
    fields = set(group.keys())

    def read_field(name: str) -> object | None:
        if name not in fields:
            return None
        return _decode_h5_value(group[name][()])

    scan_name = read_field("scan_name")
    if scan_name is None:
        raise BackendIOError(
            "BIDS-HDF5 archive is missing /scan_index/scan_name",
            operation="read",
        )
    scan_names = [str(v) for v in np.atleast_1d(scan_name)]
    n_scans = len(scan_names)

    def strings(name: str, default: str = "") -> list[str]:
        value = read_field(name)
        if value is None:
            return [default] * n_scans
        return [str(v) for v in np.atleast_1d(value)]

    def ints(name: str, default: int = 0) -> list[int]:
        value = read_field(name)
        if value is None:
            return [default] * n_scans
        return [int(v) for v in np.atleast_1d(value)]

    def bools(name: str, default: bool = False) -> list[bool]:
        value = read_field(name)
        if value is None:
            return [default] * n_scans
        return [bool(v) for v in np.atleast_1d(value)]

    n_time = ints("n_time")
    time_offset = read_field("time_offset")
    offsets = (
        [int(v) for v in np.atleast_1d(time_offset)]
        if time_offset is not None
        else [0] + list(np.cumsum(n_time)[:-1])
    )

    manifest = pd.DataFrame(
        {
            "scan_name": scan_names,
            "subject": strings("subject"),
            "task": strings("task"),
            "session": strings("session"),
            "run": strings("run"),
            "n_time": n_time,
            "time_offset": offsets,
            "has_events": bools("has_events"),
            "has_confounds": bools("has_confounds"),
        }
    )
    if len(manifest) == 0:
        raise BackendIOError(
            "BIDS-HDF5 archive contains no scans in /scan_index/",
            operation="read",
        )
    return manifest


def _read_frame(group: object, subgroup_name: str) -> pd.DataFrame | None:
    if subgroup_name not in group:
        return None
    subgroup = group[subgroup_name]
    if len(subgroup.keys()) == 0:
        return None
    columns: dict[str, object] = {}
    for key in subgroup.keys():
        columns[str(key)] = _decode_h5_value(subgroup[key][()])
    return pd.DataFrame(columns)


def _read_confounds(scan_group: object) -> pd.DataFrame | None:
    if "confounds" not in scan_group or "data" not in scan_group["confounds"]:
        return None
    dataset = scan_group["confounds/data"]
    data = np.asarray(dataset, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    names = dataset.attrs.get("names")
    if names is None:
        col_names = [f"confound_{i + 1}" for i in range(data.shape[1])]
    else:
        decoded = _decode_h5_value(np.asarray(names))
        col_names = [str(v) for v in np.atleast_1d(decoded)]
    return pd.DataFrame(data, columns=col_names)


def _read_censor(scan_group: object, n_time: int) -> NDArray[np.bool_]:
    if "censor" not in scan_group:
        return np.zeros(n_time, dtype=np.bool_)
    return np.asarray(scan_group["censor"], dtype=np.bool_)


def _bind_frames(frames: list[pd.DataFrame | None]) -> pd.DataFrame:
    non_empty = [f for f in frames if f is not None and len(f) > 0]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True, sort=False)


class _BidsSubjectAdapter:
    """Run-wise adapter over one subject's BIDS-HDF5 scan backends."""

    def __init__(
        self,
        backends: Sequence[BidsH5ScanBackend],
        sampling_frame: SamplingFrame,
    ) -> None:
        if not backends:
            raise FmriDatasetError("BIDS-HDF5 subject requires at least one scan")
        self._backends = list(backends)
        self._sampling_frame = sampling_frame
        for backend in self._backends:
            backend.open()

    def get_data(self, run: int) -> NDArray[np.float64]:
        if run < 0 or run >= len(self._backends):
            raise IndexError(f"Run {run} out of range [0, {len(self._backends)})")
        return np.asarray(self._backends[run].get_data(), dtype=np.float64)

    def get_mask(self) -> NDArray[np.bool_]:
        return self._backends[0].get_mask()

    def get_sampling_frame(self) -> SamplingFrame:
        return self._sampling_frame

    @property
    def n_runs(self) -> int:
        return len(self._backends)

    @property
    def n_timepoints(self) -> list[int]:
        return [backend.n_time for backend in self._backends]

    @property
    def n_voxels(self) -> int:
        return int(self._backends[0].get_mask().sum())


def _make_scan_backends(
    manifest: pd.DataFrame,
    connection: SharedH5Connection,
    n_features: int,
    tr: float,
    compression_mode: CompressionMode,
) -> dict[str, BidsH5ScanBackend]:
    out: dict[str, BidsH5ScanBackend] = {}
    for row in manifest.to_dict("records"):
        metadata = {
            "subject": row["subject"],
            "task": row["task"],
            "session": row["session"] if row["session"] else None,
            "run": row["run"],
            "tr": tr,
        }
        name = str(row["scan_name"])
        out[name] = bids_h5_scan_backend(
            h5_connection=connection,
            scan_group_path=f"/scans/{name}",
            n_features=n_features,
            n_time=int(row["n_time"]),
            metadata=metadata,
            compression_mode=compression_mode,
        )
    return out


def _build_subject_dataset(
    scan_rows: pd.DataFrame,
    scan_backends: Mapping[str, BidsH5ScanBackend],
    handle: object,
    tr: float,
    subject_id: str,
) -> FmriDataset:
    events: list[pd.DataFrame | None] = []
    censors: list[NDArray[np.bool_]] = []
    run_lengths: list[int] = []

    for run_id, row in enumerate(scan_rows.to_dict("records"), start=1):
        scan_name = str(row["scan_name"])
        scan_group = handle[f"scans/{scan_name}"]
        n_time = int(row["n_time"])
        run_lengths.append(n_time)

        event_frame = _read_frame(scan_group, "events") if row["has_events"] else None
        if event_frame is not None and len(event_frame) > 0:
            event_frame = event_frame.copy()
            event_frame["run"] = row["run"]
            event_frame["run_id"] = run_id
            event_frame["subject_id"] = subject_id
            event_frame["task"] = row["task"]
            if row["session"]:
                event_frame["session"] = row["session"]
        events.append(event_frame)
        censors.append(_read_censor(scan_group, n_time))

    backends = [scan_backends[str(name)] for name in scan_rows["scan_name"]]
    frame = SamplingFrame(blocklens=run_lengths, tr=tr)
    adapter = _BidsSubjectAdapter(backends, frame)
    return FmriDataset(
        adapter,
        event_table=_bind_frames(events),
        censor=censors if censors else None,
    )


def _compose_bids_h5_study_dataset(
    manifest: pd.DataFrame,
    scan_backends: dict[str, BidsH5ScanBackend],
    handle: object,
    connection: SharedH5Connection,
    tr: float,
    bids_metadata: dict[str, object],
    compression_mode: CompressionMode,
) -> "BidsH5StudyDataset":
    subject_ids = list(dict.fromkeys(str(v) for v in manifest["subject"]))
    datasets = [
        _build_subject_dataset(
            manifest.loc[manifest["subject"] == sid].reset_index(drop=True),
            scan_backends,
            handle,
            tr,
            sid,
        )
        for sid in subject_ids
    ]
    subjects = pd.DataFrame({"subject_id": subject_ids, "dataset": datasets})
    return BidsH5StudyDataset(
        subjects=subjects,
        scan_manifest=manifest.reset_index(drop=True),
        h5_connection=connection,
        compression_mode=compression_mode,
        bids_metadata=bids_metadata,
        scan_backends=scan_backends,
        tr=tr,
    )


class BidsH5StudyDataset:
    """Study dataset backed by a compressed BIDS-HDF5 archive."""

    def __init__(
        self,
        subjects: pd.DataFrame,
        scan_manifest: pd.DataFrame,
        h5_connection: SharedH5Connection,
        compression_mode: CompressionMode,
        bids_metadata: dict[str, object],
        scan_backends: dict[str, BidsH5ScanBackend],
        tr: float,
    ) -> None:
        self.subjects = subjects.reset_index(drop=True)
        self._scan_manifest = scan_manifest
        self.h5_connection = h5_connection
        self.compression_mode = compression_mode
        self.bids_metadata = MappingProxyType(dict(bids_metadata))
        self.scan_backends = scan_backends
        self._tr = float(tr)

    @property
    def subject_ids(self) -> list[str]:
        return [str(v) for v in self.subjects["subject_id"]]

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)

    @property
    def datasets(self) -> tuple[FmriDataset, ...]:
        return tuple(cast(FmriDataset, value) for value in self.subjects["dataset"])

    @property
    def event_table(self) -> pd.DataFrame:
        return _bind_frames([dataset.event_table for dataset in self.datasets])

    @property
    def TR(self) -> float:
        return self._tr

    @property
    def scan_manifest(self) -> pd.DataFrame:
        """Return one row per scan in the archive."""
        return self._scan_manifest.copy()

    def get_data_matrix(
        self,
        *,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        data = np.vstack([dataset.get_data_matrix() for dataset in self.datasets])
        if rows is not None:
            data = data[np.asarray(rows, dtype=np.intp), :]
        if cols is not None:
            data = data[:, np.asarray(cols, dtype=np.intp)]
        return data

    def get_data(
        self,
        rows: NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
    ) -> NDArray[np.float64]:
        return self.get_data_matrix(rows=rows, cols=cols)

    def participants(self) -> list[str]:
        return list(dict.fromkeys(str(v) for v in self._scan_manifest["subject"]))

    def tasks(self) -> list[str]:
        return list(dict.fromkeys(str(v) for v in self._scan_manifest["task"]))

    def sessions(self) -> list[str] | None:
        values = [
            str(v)
            for v in self._scan_manifest["session"]
            if pd.notna(v) and str(v) != ""
        ]
        unique = list(dict.fromkeys(values))
        return unique or None

    def subset(
        self,
        task: str | Iterable[str] | None = None,
        subject: str | Iterable[str] | None = None,
        session: str | Iterable[str] | None = None,
        run: str | Iterable[str] | None = None,
    ) -> "BidsH5StudyDataset":
        return subset_bids_h5(
            self,
            task=task,
            subject=subject,
            session=session,
            run=run,
        )

    def parcellation_info(self) -> dict[str, object] | None:
        if self.compression_mode == "latent":
            return None
        handle = self.h5_connection.handle
        if "parcellation/cluster_ids" not in handle:
            raise BackendIOError(
                "BIDS-HDF5 archive is missing /parcellation/cluster_ids",
                file=str(self.h5_connection.file),
                operation="read",
            )
        cluster_ids = np.asarray(handle["parcellation/cluster_ids"])
        cluster_map = (
            np.asarray(handle["parcellation/cluster_map"])
            if "parcellation/cluster_map" in handle
            else None
        )
        labels = (
            _decode_h5_value(handle["parcellation/cluster_meta/labels"][()])
            if "parcellation/cluster_meta/labels" in handle
            else None
        )
        return {
            "cluster_ids": cluster_ids,
            "cluster_map": cluster_map,
            "labels": labels,
            "n_parcels": int(cluster_ids.size),
        }

    def get_loadings(
        self,
        scan_name: str | None = None,
    ) -> NDArray[np.float64] | dict[str, NDArray[np.float64]]:
        self._require_latent("get_loadings")
        all_scans = [str(v) for v in self._scan_manifest["scan_name"]]

        def one(name: str) -> NDArray[np.float64]:
            if name not in all_scans:
                raise ValueError(f"scan_name '{name}' not found in this dataset")
            path = f"scans/{name}/data/loadings"
            if path in self.h5_connection.handle:
                return np.asarray(self.h5_connection.handle[path], dtype=np.float64)
            template = "latent_meta/template/loadings"
            if self._has_shared_template() and template in self.h5_connection.handle:
                return np.asarray(self.h5_connection.handle[template], dtype=np.float64)
            raise BackendIOError(
                f"No loadings found for scan '{name}' and no shared template",
                file=str(self.h5_connection.file),
                operation="read",
            )

        if scan_name is not None:
            return one(scan_name)
        return {name: one(name) for name in all_scans}

    def reconstruct_voxels(
        self,
        scan_name: str,
        rows: NDArray[np.intp] | list[int] | None = None,
        voxels: NDArray[np.intp] | list[int] | None = None,
    ) -> NDArray[np.float64]:
        self._require_latent("reconstruct_voxels")
        backend = self.scan_backends.get(scan_name)
        if backend is None:
            raise ValueError(f"scan_name '{scan_name}' not found in this dataset")
        basis = backend.get_data()
        loadings = self.get_loadings(scan_name)
        assert isinstance(loadings, np.ndarray)
        data = basis @ loadings.T
        offset_path = f"scans/{scan_name}/data/offset"
        if offset_path in self.h5_connection.handle:
            offset = np.asarray(
                self.h5_connection.handle[offset_path],
                dtype=np.float64,
            )
            data = data + offset[np.newaxis, :]
        if rows is not None:
            data = data[np.asarray(rows, dtype=np.intp), :]
        if voxels is not None:
            data = data[:, np.asarray(voxels, dtype=np.intp)]
        return data

    def encoding_info(self) -> dict[str, object] | None:
        if self.compression_mode != "latent":
            return None
        handle = self.h5_connection.handle
        params = _read_optional_scalar(handle, "latent_meta/encoding_params")
        decoded_params = None
        if isinstance(params, str):
            try:
                decoded_params = json.loads(params)
            except json.JSONDecodeError:
                decoded_params = None
        template_meta = None
        if self._has_shared_template() and "latent_meta/template/meta" in handle:
            raw_meta = _decode_h5_value(handle["latent_meta/template/meta"][()])
            if isinstance(raw_meta, str):
                try:
                    template_meta = json.loads(raw_meta)
                except json.JSONDecodeError:
                    template_meta = {}

        return {
            "encoding_family": _read_optional_scalar(
                handle,
                "latent_meta/encoding_family",
            ),
            "encoding_params": decoded_params,
            "n_components": int(_read_dataset(handle, "latent_meta/n_components")),
            "has_shared_template": self._has_shared_template(),
            "template_meta": template_meta,
        }

    def get_confounds(
        self,
        scan_name: str | Iterable[str] | None = None,
        subject: str | Iterable[str] | None = None,
        task: str | Iterable[str] | None = None,
    ) -> pd.DataFrame | dict[str, pd.DataFrame] | None:
        manifest = self._scan_manifest.copy()
        keep = manifest["has_confounds"].astype(bool)
        scan_filter = _as_list(scan_name)
        subject_filter = _as_list(subject)
        task_filter = _as_list(task)
        if scan_filter is not None:
            keep = keep & manifest["scan_name"].isin(scan_filter)
        if subject_filter is not None:
            keep = keep & manifest["subject"].isin(subject_filter)
        if task_filter is not None:
            keep = keep & manifest["task"].isin(task_filter)
        matching = manifest.loc[keep]
        if len(matching) == 0:
            return None

        result: dict[str, pd.DataFrame] = {}
        for name in matching["scan_name"]:
            frame = _read_confounds(self.h5_connection.handle[f"scans/{name}"])
            if frame is not None:
                result[str(name)] = frame
        if not result:
            return None
        if len(result) == 1:
            return next(iter(result.values()))
        return result

    def to_group(self) -> "BidsH5StudyDataset":
        return self

    def _require_latent(self, operation: str) -> None:
        if self.compression_mode != "latent":
            raise ValueError(f"{operation} is only available for latent-mode archives")

    def _has_shared_template(self) -> bool:
        value = _read_optional_scalar(
            self.h5_connection.handle,
            "latent_meta/has_shared_template",
        )
        return bool(value)


def bids_h5_dataset(file: str | Path, preload: bool = False) -> BidsH5StudyDataset:
    """Open a compressed BIDS-HDF5 study archive."""
    del preload
    connection = h5_shared_connection(file)
    handle = connection.handle

    fmt = _decode_h5_value(handle.attrs.get("format"))
    if fmt != "bids_h5_study":
        raise BackendIOError(
            f"Unsupported archive format '{fmt}' (expected 'bids_h5_study')",
            file=str(connection.file),
            operation="validate",
        )
    version = _decode_h5_value(handle.attrs.get("version", "1.0"))
    if not str(version).startswith("1."):
        raise BackendIOError(
            f"Unsupported BIDS-HDF5 schema version '{version}'",
            file=str(connection.file),
            operation="validate",
        )
    mode_value = _decode_h5_value(handle.attrs.get("compression_mode", "parcellated"))
    if mode_value not in ("parcellated", "latent"):
        raise BackendIOError(
            f"Unknown compression_mode '{mode_value}'",
            file=str(connection.file),
            operation="validate",
        )
    compression_mode: CompressionMode = mode_value

    manifest = _read_scan_index(handle)
    if compression_mode == "parcellated":
        cluster_ids = _read_dataset(handle, "parcellation/cluster_ids")
        n_features = int(np.asarray(cluster_ids).size)
    else:
        n_features = int(_read_dataset(handle, "latent_meta/n_components"))

    first_scan = str(manifest["scan_name"].iloc[0])
    tr = float(_read_dataset(handle, f"scans/{first_scan}/metadata/tr"))
    bids_metadata = {
        key: _read_optional_scalar(handle, f"bids/{key}")
        for key in ("space", "pipeline", "name")
    }
    scan_backends = _make_scan_backends(
        manifest=manifest,
        connection=connection,
        n_features=n_features,
        tr=tr,
        compression_mode=compression_mode,
    )
    return _compose_bids_h5_study_dataset(
        manifest=manifest,
        scan_backends=scan_backends,
        handle=handle,
        connection=connection,
        tr=tr,
        bids_metadata=bids_metadata,
        compression_mode=compression_mode,
    )


def subset_bids_h5(
    dataset: BidsH5StudyDataset,
    task: str | Iterable[str] | None = None,
    subject: str | Iterable[str] | None = None,
    session: str | Iterable[str] | None = None,
    run: str | Iterable[str] | None = None,
) -> BidsH5StudyDataset:
    """Return a filtered BIDS-HDF5 study dataset sharing the same file handle."""
    if not isinstance(dataset, BidsH5StudyDataset):
        raise TypeError("dataset must be a BidsH5StudyDataset")

    manifest = dataset.scan_manifest
    keep = pd.Series(True, index=manifest.index)
    filters = {
        "task": _as_list(task),
        "subject": _as_list(subject),
        "session": _as_list(session),
        "run": _as_list(run),
    }
    for column, values in filters.items():
        if values is not None:
            keep = keep & manifest[column].isin(values)
    sub_manifest = manifest.loc[keep].reset_index(drop=True)
    if len(sub_manifest) == 0:
        raise ValueError("subset_bids_h5: no scans match the provided filters")

    first_backend = next(iter(dataset.scan_backends.values()))
    scan_backends = _make_scan_backends(
        manifest=sub_manifest,
        connection=dataset.h5_connection,
        n_features=first_backend.n_features,
        tr=dataset.TR,
        compression_mode=dataset.compression_mode,
    )
    return _compose_bids_h5_study_dataset(
        manifest=sub_manifest,
        scan_backends=scan_backends,
        handle=dataset.h5_connection.handle,
        connection=dataset.h5_connection,
        tr=dataset.TR,
        bids_metadata=dict(dataset.bids_metadata),
        compression_mode=dataset.compression_mode,
    )


def participants(dataset: BidsH5StudyDataset) -> list[str]:
    return dataset.participants()


def tasks(dataset: BidsH5StudyDataset) -> list[str]:
    return dataset.tasks()


def sessions(dataset: BidsH5StudyDataset) -> list[str] | None:
    return dataset.sessions()


def scan_manifest(dataset: BidsH5StudyDataset) -> pd.DataFrame:
    return dataset.scan_manifest


def parcellation_info(dataset: BidsH5StudyDataset) -> dict[str, object] | None:
    return dataset.parcellation_info()


def get_confounds(
    dataset: BidsH5StudyDataset,
    scan_name: str | Iterable[str] | None = None,
    subject: str | Iterable[str] | None = None,
    task: str | Iterable[str] | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame] | None:
    return dataset.get_confounds(scan_name=scan_name, subject=subject, task=task)


def get_loadings(
    dataset: BidsH5StudyDataset,
    scan_name: str | None = None,
) -> NDArray[np.float64] | dict[str, NDArray[np.float64]]:
    return dataset.get_loadings(scan_name=scan_name)


def reconstruct_voxels(
    dataset: BidsH5StudyDataset,
    scan_name: str,
    rows: NDArray[np.intp] | list[int] | None = None,
    voxels: NDArray[np.intp] | list[int] | None = None,
) -> NDArray[np.float64]:
    return dataset.reconstruct_voxels(scan_name=scan_name, rows=rows, voxels=voxels)


def encoding_info(dataset: BidsH5StudyDataset) -> dict[str, object] | None:
    return dataset.encoding_info()


__all__ = [
    "BidsH5StudyDataset",
    "bids_h5_dataset",
    "encoding_info",
    "get_confounds",
    "get_loadings",
    "parcellation_info",
    "participants",
    "reconstruct_voxels",
    "scan_manifest",
    "sessions",
    "subset_bids_h5",
    "tasks",
]
