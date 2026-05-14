"""Write datasets to the BIDS-HDF5 study schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .bids_h5 import BidsH5StudyDataset, bids_h5_dataset
from .constructors import matrix_dataset
from .errors import ConfigError
from .fmri_dataset import FmriDataset
from .study import StudyDataset


def _as_string_array(values: Sequence[Any]) -> NDArray[np.object_]:
    return np.asarray(["" if v is None else str(v) for v in values], dtype=object)


def _write_string_array(group: Any, name: str, values: Sequence[Any]) -> None:
    import h5py

    dtype = h5py.string_dtype("utf-8")
    group.create_dataset(name, data=_as_string_array(values), dtype=dtype)


def _write_string_scalar(group: Any, name: str, value: Any) -> None:
    import h5py

    dtype = h5py.string_dtype("utf-8")
    group.create_dataset(name, data="" if value is None else str(value), dtype=dtype)


def _compression_kwargs(compression: int) -> dict[str, Any]:
    if compression == 0:
        return {}
    return {"compression": "gzip", "compression_opts": compression}


def _write_events(group: Any, events: pd.DataFrame, compression: int) -> None:
    if len(events) == 0:
        return
    event_group = group.create_group("events")
    for name in events.columns:
        values = events[name].to_numpy()
        if values.dtype.kind in {"O", "U", "S"}:
            _write_string_array(event_group, str(name), values.tolist())
        else:
            event_group.create_dataset(
                str(name),
                data=values,
                **_compression_kwargs(compression),
            )


def _write_confounds(
    group: Any,
    confounds: pd.DataFrame | None,
    compression: int,
) -> bool:
    if confounds is None:
        return False
    conf_group = group.create_group("confounds")
    dataset = conf_group.create_dataset(
        "data",
        data=confounds.to_numpy(dtype=float),
        **_compression_kwargs(compression),
    )
    dataset.attrs["names"] = _as_string_array(list(confounds.columns))
    return True


def _normalise_filter(
    values: Sequence[str] | str | None,
    *,
    strip_prefix: str = "",
) -> list[str] | str | None:
    if values is None:
        return None
    if isinstance(values, str):
        return values.removeprefix(strip_prefix)
    return [str(value).removeprefix(strip_prefix) for value in values]


def _subject_matches(subject_id: str, filters: set[str]) -> bool:
    if not filters:
        return True
    bare = subject_id.removeprefix("sub-")
    prefixed = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    return subject_id in filters or bare in filters or prefixed in filters


def _normalise_input(
    x: StudyDataset | Mapping[str, FmriDataset] | Sequence[FmriDataset] | str | Path,
    tasks: Sequence[str] | str | None = None,
    subjects: Sequence[str] | str | None = None,
    sessions: Sequence[str] | str | None = None,
    clusters: Any | None = None,
    mask: Any | None = None,
    summary_fun: Any | None = None,
) -> list[tuple[str, FmriDataset]]:
    if isinstance(x, (str, Path)) and Path(x).is_dir():
        return _normalise_bids_dir(
            Path(x),
            tasks=tasks,
            subjects=subjects,
            sessions=sessions,
            clusters=clusters,
            mask=mask,
            summary_fun=summary_fun,
        )
    if isinstance(x, (str, Path)):
        raise ValueError(f"BIDS directory does not exist: {x}")
    if isinstance(x, StudyDataset):
        return [
            (str(subject_id), dataset)
            for subject_id, dataset in zip(x.subject_ids, x.datasets)
        ]
    if isinstance(x, Mapping):
        return [(str(subject_id), dataset) for subject_id, dataset in x.items()]
    return [(str(index), dataset) for index, dataset in enumerate(x, start=1)]


def _normalise_bids_dir(
    root: Path,
    tasks: Sequence[str] | str | None = None,
    subjects: Sequence[str] | str | None = None,
    sessions: Sequence[str] | str | None = None,
    clusters: Any | None = None,
    mask: Any | None = None,
    summary_fun: Any | None = None,
) -> list[tuple[str, FmriDataset]]:
    try:
        from bids import BIDSLayout  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ConfigError(
            "pybids is required when compress_bids_study() is given a BIDS "
            "directory. Install with: pip install fmrimod[bids]",
            parameter="bids",
        ) from exc

    try:
        import nibabel as nib
    except ImportError as exc:
        raise ConfigError(
            "nibabel is required to load BIDS NIfTI files. "
            "Install with: pip install fmrimod[nifti]",
            parameter="nibabel",
        ) from exc

    layout = BIDSLayout(str(root))
    query: dict[str, Any] = {
        "datatype": "func",
        "suffix": "bold",
        "extension": [".nii.gz", ".nii"],
        "return_type": "filename",
    }
    task_filter = _normalise_filter(tasks)
    subject_filter = _normalise_filter(subjects, strip_prefix="sub-")
    session_filter = _normalise_filter(sessions, strip_prefix="ses-")
    if task_filter is not None:
        query["task"] = task_filter
    if subject_filter is not None:
        query["subject"] = subject_filter
    if session_filter is not None:
        query["session"] = session_filter

    bold_files = [Path(path) for path in layout.get(**query)]
    if not bold_files:
        raise ValueError("no BOLD files found in BIDS directory")

    mask_vec = _load_optional_spatial_vector(mask, name="mask", nib=nib)
    cluster_vec = _load_optional_spatial_vector(clusters, name="clusters", nib=nib)

    out: list[tuple[str, FmriDataset]] = []
    for bold_path in bold_files:
        entities = layout.parse_file_entities(str(bold_path))
        subject_label = str(entities.get("subject", "unknown"))
        subject_id = f"sub-{subject_label}"
        task = str(entities.get("task", "task"))
        session = entities.get("session")
        run = entities.get("run")

        metadata = layout.get_metadata(str(bold_path))
        tr = metadata.get("RepetitionTime")
        if tr is None:
            raise ValueError(f"BIDS file '{bold_path}' has no RepetitionTime metadata")

        img: Any = nib.load(str(bold_path))
        data = np.asarray(img.dataobj, dtype=np.float64)
        if data.ndim == 3:
            data = data[..., np.newaxis]
        if data.ndim != 4:
            raise ValueError(f"BIDS file '{bold_path}' is not 3D or 4D")
        n_time = data.shape[3]
        matrix = data.reshape(-1, n_time).T
        matrix, cluster_ids = _summarise_bids_matrix(
            matrix,
            mask=mask_vec,
            clusters=cluster_vec,
            summary_fun=summary_fun,
            source=bold_path,
        )

        events = _read_bids_events(layout, bold_path)
        if len(events) > 0:
            events["run_id"] = 1

        dataset = matrix_dataset(
            matrix,
            tr=float(tr),
            run_length=n_time,
            event_table=events,
        )
        dataset._bids_task = task
        dataset._bids_session = "" if session is None else str(session)
        dataset._bids_run = "1" if run is None else str(run)
        if cluster_ids is not None:
            dataset._bids_cluster_ids = cluster_ids
        out.append((subject_id, dataset))
    return out


def _load_optional_spatial_vector(
    value: Any | None,
    *,
    name: str,
    nib: Any,
) -> NDArray[Any] | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        img: Any = nib.load(str(value))
        arr = np.asarray(img.dataobj)
    else:
        arr = np.asarray(value)
    if arr.ndim > 3:
        raise ValueError(f"{name} must be a 1D or 3D spatial array")
    return arr.ravel()


def _summarise_bids_matrix(
    matrix: NDArray[np.float64],
    *,
    mask: NDArray[Any] | None,
    clusters: NDArray[Any] | None,
    summary_fun: Any | None,
    source: Path,
) -> tuple[NDArray[np.float64], NDArray[Any] | None]:
    n_voxels = matrix.shape[1]
    mask_bool: NDArray[np.bool_] | None = None
    if mask is not None:
        if mask.size != n_voxels:
            raise ValueError(f"mask length does not match voxel count for '{source}'")
        mask_bool = np.asarray(mask, dtype=np.bool_)
        matrix = matrix[:, mask_bool]

    if clusters is None:
        return matrix, None

    if clusters.size != n_voxels:
        raise ValueError(f"clusters length does not match voxel count for '{source}'")
    cluster_labels = clusters[mask_bool] if mask_bool is not None else clusters
    keep = pd.notna(cluster_labels) & (np.asarray(cluster_labels) != 0)
    cluster_labels = cluster_labels[keep]
    parcel_matrix = matrix[:, keep]
    cluster_ids = np.asarray(sorted(pd.unique(cluster_labels)))

    if len(cluster_ids) == 0:
        raise ValueError(f"clusters select no non-zero parcels for '{source}'")

    reducer = np.mean if summary_fun is None else summary_fun
    summaries = []
    for cluster_id in cluster_ids:
        values = parcel_matrix[:, cluster_labels == cluster_id]
        summary = reducer(values, axis=1)
        summaries.append(np.asarray(summary, dtype=np.float64))
    return np.column_stack(summaries), cluster_ids


def _read_bids_events(layout: Any, bold_path: Path) -> pd.DataFrame:
    events_getter = getattr(layout, "get_events", None)
    if events_getter is not None:
        try:
            events_obj = events_getter(str(bold_path))
            if events_obj is not None:
                return pd.DataFrame(events_obj)
        except Exception:
            pass

    nearest = None
    for strict in (True, False):
        try:
            nearest = layout.get_nearest(
                str(bold_path),
                suffix="events",
                extension=".tsv",
                return_type="filename",
                strict=strict,
                all_=False,
            )
        except Exception:
            nearest = None
        if nearest is not None:
            break

    if nearest is None:
        return pd.DataFrame()
    if isinstance(nearest, (list, tuple)):
        if not nearest:
            return pd.DataFrame()
        nearest = nearest[0]
    return pd.read_csv(str(nearest), sep="\t")


def _event_rows_for_run(dataset: FmriDataset, run_id: int) -> pd.DataFrame:
    events = dataset.event_table
    if events is None or len(events) == 0:
        return pd.DataFrame()
    if "run_id" in events.columns:
        return events.loc[events["run_id"] == run_id].reset_index(drop=True)
    if "run" in events.columns:
        return events.loc[events["run"] == run_id].reset_index(drop=True)
    if dataset.n_runs == 1:
        return events.reset_index(drop=True)
    return pd.DataFrame()


def _pca_encode(
    data: NDArray[np.floating[Any]],
    n_components: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    matrix = np.asarray(data, dtype=np.float64)
    offset = np.mean(matrix, axis=0)
    centered = matrix - offset[np.newaxis, :]
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    max_components = min(u.shape[1], vt.shape[0])
    k = max_components if n_components is None else min(int(n_components), max_components)
    basis = u[:, :k] * singular_values[:k]
    loadings = vt[:k, :].T
    return basis, loadings, offset


def _scan_censor(dataset: FmriDataset, run_index: int, n_time: int) -> NDArray[np.intp]:
    censor = dataset.get_censor(run_index)
    if censor is None:
        return np.zeros(n_time, dtype=np.intp)
    return np.asarray(censor, dtype=np.intp)


def compress_bids_study(
    x: StudyDataset | Mapping[str, FmriDataset] | Sequence[FmriDataset] | str | Path,
    file: str | Path,
    mode: Literal["parcellated", "latent"] = "parcellated",
    clusters: Sequence[Any] | None = None,
    summary_fun: Any | None = None,
    encoding: Any | None = None,
    n_components: int | None = None,
    template: Any | None = None,
    mask: Any | None = None,
    space: str = "MNI152NLin2009cAsym",
    tasks: Sequence[str] | str | None = None,
    subjects: Sequence[str] | str | None = None,
    sessions: Sequence[str] | str | None = None,
    confounds: Mapping[str, pd.DataFrame] | None = None,
    compression: int = 4,
    verbose: bool = False,
) -> BidsH5StudyDataset:
    """Write datasets to a BIDS-HDF5 archive and return its reader.

    Already materialized :class:`FmriDataset` objects are written in the same
    v1.0 archive schema consumed by :func:`bids_h5_dataset`. BIDS directory
    ingestion stays at the optional IO boundary and requires PyBIDS plus
    nibabel; the in-memory path only requires ``h5py``.
    """
    del encoding, template, verbose

    if mode not in {"parcellated", "latent"}:
        raise ValueError("mode must be 'parcellated' or 'latent'")
    if compression < 0 or compression > 9:
        raise ValueError("compression must be between 0 and 9")

    subject_filter = {subjects} if isinstance(subjects, str) else set(subjects or [])
    task_value = (
        tasks[0]
        if isinstance(tasks, Sequence) and not isinstance(tasks, str) and tasks
        else tasks
    )
    session_value = (
        sessions[0]
        if isinstance(sessions, Sequence) and not isinstance(sessions, str) and sessions
        else sessions
    )
    task = str(task_value or "task")
    session = "" if session_value is None else str(session_value)

    entries = _normalise_input(
        x,
        tasks=tasks,
        subjects=subjects,
        sessions=sessions,
        clusters=clusters,
        mask=mask,
        summary_fun=summary_fun,
    )
    if subject_filter:
        entries = [
            (sid, dataset)
            for sid, dataset in entries
            if _subject_matches(sid, subject_filter)
        ]
    if not entries:
        raise ValueError("no datasets to write")

    try:
        import h5py
    except ImportError as exc:
        raise ConfigError(
            "h5py is required for BIDS-HDF5 writing. "
            "Install with: pip install fmrimod[hdf5]",
            parameter="h5py",
        ) from exc

    path = Path(file)
    path.parent.mkdir(parents=True, exist_ok=True)

    scan_rows: list[dict[str, Any]] = []
    time_offset = 0
    first_n_features: int | None = None
    feature_ids: NDArray[Any] | None = None

    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "bids_h5_study"
        handle.attrs["version"] = "1.0"
        handle.attrs["compression_mode"] = mode
        handle.attrs["writer_version"] = "fmrimod"

        bids_group = handle.create_group("bids")
        _write_string_scalar(bids_group, "space", space)
        _write_string_scalar(bids_group, "pipeline", "fmrimod")
        _write_string_scalar(bids_group, "name", path.stem)

        scans_group = handle.create_group("scans")

        for subject_id, dataset in entries:
            for run_index in range(dataset.n_runs):
                run_id = run_index + 1
                task_for_scan = str(getattr(dataset, "_bids_task", task))
                session_for_scan = str(getattr(dataset, "_bids_session", session))
                run_label = str(getattr(dataset, "_bids_run", run_id))
                data = np.asarray(dataset.get_run_data(run_index), dtype=np.float64)
                n_time, n_features = data.shape

                dataset_cluster_ids = getattr(dataset, "_bids_cluster_ids", None)
                if dataset_cluster_ids is not None:
                    dataset_feature_ids = np.asarray(dataset_cluster_ids)
                    if dataset_feature_ids.size != n_features:
                        raise ValueError("cluster id count must match feature count")
                    if feature_ids is None:
                        feature_ids = dataset_feature_ids
                    elif not np.array_equal(feature_ids, dataset_feature_ids):
                        raise ValueError("all BIDS scans must use the same cluster ids")
                if first_n_features is None:
                    first_n_features = n_features
                elif mode == "parcellated" and n_features != first_n_features:
                    raise ValueError(
                        "all parcellated scans must have the same feature count"
                    )

                scan_name = (
                    f"{subject_id}_ses-{session_for_scan}_task-{task_for_scan}_"
                    f"run-{run_label}"
                    if session_for_scan
                    else f"{subject_id}_task-{task_for_scan}_run-{run_label}"
                )
                scan_group = scans_group.create_group(scan_name)
                data_group = scan_group.create_group("data")
                if mode == "latent":
                    basis, loadings, offset = _pca_encode(data, n_components)
                    data_group.create_dataset(
                        "basis",
                        data=basis,
                        **_compression_kwargs(compression),
                    )
                    data_group.create_dataset(
                        "loadings",
                        data=loadings,
                        **_compression_kwargs(compression),
                    )
                    data_group.create_dataset(
                        "offset",
                        data=offset,
                        **_compression_kwargs(compression),
                    )
                    n_features_out = basis.shape[1]
                else:
                    data_group.create_dataset(
                        "summary_data",
                        data=data,
                        **_compression_kwargs(compression),
                    )
                    n_features_out = n_features

                metadata = scan_group.create_group("metadata")
                metadata.create_dataset("tr", data=dataset.sampling_frame.TR)
                scan_group.create_dataset(
                    "censor",
                    data=_scan_censor(dataset, run_index, n_time),
                )

                events = _event_rows_for_run(dataset, run_id)
                _write_events(scan_group, events, compression=compression)
                confound_frame = confounds.get(scan_name) if confounds is not None else None
                has_confounds = _write_confounds(
                    scan_group,
                    confound_frame,
                    compression=compression,
                )

                scan_rows.append(
                    {
                        "scan_name": scan_name,
                        "subject": subject_id,
                        "task": task_for_scan,
                        "session": session_for_scan,
                        "run": run_label,
                        "n_time": n_time,
                        "time_offset": time_offset,
                        "has_events": len(events) > 0,
                        "has_confounds": has_confounds,
                        "n_features": n_features_out,
                    }
                )
                time_offset += n_time

        if first_n_features is None:
            raise ValueError("no scans were written")

        if mode == "parcellated":
            parcel_group = handle.create_group("parcellation")
            cluster_ids = feature_ids
            if cluster_ids is None:
                cluster_ids = (
                    np.asarray(clusters)
                    if clusters is not None
                    else np.arange(1, first_n_features + 1, dtype=np.intp)
                )
            if cluster_ids.size != first_n_features:
                raise ValueError("clusters must match the feature count")
            parcel_group.create_dataset("cluster_ids", data=cluster_ids)
        else:
            latent_group = handle.create_group("latent_meta")
            n_written_components = int(scan_rows[0]["n_features"])
            latent_group.create_dataset("n_components", data=n_written_components)
            _write_string_scalar(latent_group, "encoding_family", "pca")
            _write_string_scalar(
                latent_group,
                "encoding_params",
                f'{{"n_components": {n_written_components}}}',
            )
            latent_group.create_dataset("has_shared_template", data=False)

        index = handle.create_group("scan_index")
        _write_string_array(index, "scan_name", [row["scan_name"] for row in scan_rows])
        _write_string_array(index, "subject", [row["subject"] for row in scan_rows])
        _write_string_array(index, "task", [row["task"] for row in scan_rows])
        _write_string_array(index, "session", [row["session"] for row in scan_rows])
        _write_string_array(index, "run", [row["run"] for row in scan_rows])
        index.create_dataset(
            "n_time",
            data=np.asarray([row["n_time"] for row in scan_rows], dtype=np.intp),
        )
        index.create_dataset(
            "time_offset",
            data=np.asarray([row["time_offset"] for row in scan_rows], dtype=np.intp),
        )
        index.create_dataset(
            "has_events",
            data=np.asarray([row["has_events"] for row in scan_rows], dtype=bool),
        )
        index.create_dataset(
            "has_confounds",
            data=np.asarray([row["has_confounds"] for row in scan_rows], dtype=bool),
        )

    return bids_h5_dataset(path)


__all__ = ["compress_bids_study"]
