"""Read-side contracts for BIDS-HDF5 study archives."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")

from fmrimod.dataset import (  # noqa: E402
    BackendIOError,
    BidsH5ScanBackend,
    BidsH5StudyDataset,
    bids_h5_dataset,
    encoding_info,
    get_confounds,
    get_loadings,
    parcellation_info,
    participants,
    reconstruct_voxels,
    scan_manifest,
    sessions,
    study_to_group,
    subset_bids_h5,
    tasks,
    validate_backend,
)


def _write_str_array(group, name: str, values: list[str]) -> None:
    dtype = h5py.string_dtype("utf-8")
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype)


def _write_str_scalar(group, name: str, value: str) -> None:
    dtype = h5py.string_dtype("utf-8")
    group.create_dataset(name, data=value, dtype=dtype)


@pytest.fixture()
def parcellated_bids_h5(tmp_path):
    path = tmp_path / "study_parcellated.h5"
    scans = [
        ("sub-01_task-nback_run-1", "sub-01", "nback", "", "1", 3),
        ("sub-01_task-nback_run-2", "sub-01", "nback", "", "2", 2),
        ("sub-02_task-nback_run-1", "sub-02", "nback", "", "1", 4),
    ]
    matrices = {
        name: (np.arange(n_time * 3, dtype=float).reshape(n_time, 3) + i * 100)
        for i, (name, _sub, _task, _ses, _run, n_time) in enumerate(scans)
    }

    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "bids_h5_study"
        handle.attrs["version"] = "1.0"
        handle.attrs["compression_mode"] = "parcellated"

        index = handle.create_group("scan_index")
        _write_str_array(index, "scan_name", [s[0] for s in scans])
        _write_str_array(index, "subject", [s[1] for s in scans])
        _write_str_array(index, "task", [s[2] for s in scans])
        _write_str_array(index, "session", [s[3] for s in scans])
        _write_str_array(index, "run", [s[4] for s in scans])
        index.create_dataset("n_time", data=np.asarray([s[5] for s in scans]))
        index.create_dataset("time_offset", data=np.asarray([0, 3, 5]))
        index.create_dataset("has_events", data=np.asarray([True, False, False]))
        index.create_dataset("has_confounds", data=np.asarray([False, False, True]))

        bids = handle.create_group("bids")
        _write_str_scalar(bids, "space", "MNI152")
        _write_str_scalar(bids, "pipeline", "test-pipeline")
        _write_str_scalar(bids, "name", "synthetic")

        parc = handle.create_group("parcellation")
        parc.create_dataset("cluster_ids", data=np.asarray([11, 12, 13]))

        for name, _sub, _task, _ses, _run, n_time in scans:
            scan = handle.create_group(f"scans/{name}")
            data_group = scan.create_group("data")
            data_group.create_dataset("summary_data", data=matrices[name])
            metadata = scan.create_group("metadata")
            metadata.create_dataset("tr", data=2.0)
            scan.create_dataset("censor", data=np.zeros(n_time, dtype=np.intp))

        event_group = handle["scans/sub-01_task-nback_run-1"].create_group("events")
        event_group.create_dataset("onset", data=np.asarray([2.0, 8.0]))
        event_group.create_dataset("duration", data=np.asarray([1.5, 1.5]))
        _write_str_array(event_group, "trial_type", ["target", "control"])

        conf_group = handle["scans/sub-02_task-nback_run-1"].create_group("confounds")
        confounds = np.arange(8, dtype=float).reshape(4, 2)
        conf_dataset = conf_group.create_dataset("data", data=confounds)
        conf_dataset.attrs["names"] = np.asarray(
            ["motion_x", "motion_y"],
            dtype=h5py.string_dtype("utf-8"),
        )

    return path, matrices


@pytest.fixture()
def latent_bids_h5(tmp_path):
    path = tmp_path / "study_latent.h5"
    scan_name = "sub-01_task-rest_run-1"
    basis = np.asarray([[1.0, 0.5], [0.0, 1.0], [2.0, -1.0]])
    loadings = np.asarray([[1.0, 0.0], [0.5, 2.0], [-1.0, 1.0], [2.0, 0.5]])
    offset = np.asarray([0.1, 0.2, 0.3, 0.4])

    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "bids_h5_study"
        handle.attrs["version"] = "1.0"
        handle.attrs["compression_mode"] = "latent"

        index = handle.create_group("scan_index")
        _write_str_array(index, "scan_name", [scan_name])
        _write_str_array(index, "subject", ["sub-01"])
        _write_str_array(index, "task", ["rest"])
        _write_str_array(index, "session", ["ses-1"])
        _write_str_array(index, "run", ["1"])
        index.create_dataset("n_time", data=np.asarray([basis.shape[0]]))
        index.create_dataset("has_events", data=np.asarray([False]))
        index.create_dataset("has_confounds", data=np.asarray([False]))

        meta = handle.create_group("latent_meta")
        meta.create_dataset("n_components", data=2)
        _write_str_scalar(meta, "encoding_family", "pca")
        _write_str_scalar(meta, "encoding_params", json.dumps({"center": True}))
        meta.create_dataset("has_shared_template", data=False)

        scan = handle.create_group(f"scans/{scan_name}")
        data_group = scan.create_group("data")
        data_group.create_dataset("basis", data=basis)
        data_group.create_dataset("loadings", data=loadings)
        data_group.create_dataset("offset", data=offset)
        metadata = scan.create_group("metadata")
        metadata.create_dataset("tr", data=1.5)

    return path, scan_name, basis, loadings, offset


def test_bids_h5_parcellated_study_accessors(parcellated_bids_h5) -> None:
    path, matrices = parcellated_bids_h5
    dataset = bids_h5_dataset(path)

    assert isinstance(dataset, BidsH5StudyDataset)
    assert dataset.compression_mode == "parcellated"
    assert participants(dataset) == ["sub-01", "sub-02"]
    assert tasks(dataset) == ["nback"]
    assert sessions(dataset) is None
    assert scan_manifest(dataset).shape[0] == 3

    info = parcellation_info(dataset)
    assert info is not None
    np.testing.assert_array_equal(info["cluster_ids"], np.asarray([11, 12, 13]))

    expected = np.vstack(
        [matrices[name] for name in scan_manifest(dataset)["scan_name"]]
    )
    np.testing.assert_array_equal(dataset.get_data_matrix(), expected)


def test_bids_h5_parcellated_subset_and_metadata(parcellated_bids_h5) -> None:
    path, matrices = parcellated_bids_h5
    dataset = bids_h5_dataset(path)
    sub01 = subset_bids_h5(dataset, subject="sub-01")

    assert sub01.subject_ids == ["sub-01"]
    assert scan_manifest(sub01)["scan_name"].tolist() == [
        "sub-01_task-nback_run-1",
        "sub-01_task-nback_run-2",
    ]
    expected = np.vstack(
        [matrices["sub-01_task-nback_run-1"], matrices["sub-01_task-nback_run-2"]]
    )
    np.testing.assert_array_equal(sub01.get_data_matrix(), expected)

    assert {"run_id", "subject_id", "task"}.issubset(set(dataset.event_table.columns))
    assert dataset.event_table["subject_id"].unique().tolist() == ["sub-01"]


def test_bids_h5_parcellated_errors_and_absent_optional_data(
    tmp_path,
    parcellated_bids_h5,
) -> None:
    path, _matrices = parcellated_bids_h5
    dataset = bids_h5_dataset(path)

    with pytest.raises(BackendIOError, match="not found"):
        bids_h5_dataset(tmp_path / "missing.h5")

    wrong_format = tmp_path / "wrong_format.h5"
    with h5py.File(wrong_format, "w") as handle:
        handle.attrs["format"] = "not_bids_h5"
    with pytest.raises(BackendIOError, match="Unsupported archive format"):
        bids_h5_dataset(wrong_format)

    bad_mode = tmp_path / "bad_mode.h5"
    with h5py.File(bad_mode, "w") as handle:
        handle.attrs["format"] = "bids_h5_study"
        handle.attrs["compression_mode"] = "unknown"
    with pytest.raises(BackendIOError, match="Unknown compression_mode"):
        bids_h5_dataset(bad_mode)

    assert get_confounds(dataset, scan_name="sub-01_task-nback_run-1") is None
    with pytest.raises(ValueError, match="no scans match"):
        subset_bids_h5(dataset, task="rest")

    first_backend = next(iter(dataset.scan_backends.values()))
    assert validate_backend(first_backend)


def test_bids_h5_confounds_and_group_conversion(parcellated_bids_h5) -> None:
    path, _matrices = parcellated_bids_h5
    dataset = bids_h5_dataset(path)

    confounds = get_confounds(dataset, subject="sub-02")
    assert isinstance(confounds, pd.DataFrame)
    assert confounds.columns.tolist() == ["motion_x", "motion_y"]
    assert confounds.shape == (4, 2)

    group = study_to_group(dataset)
    assert group.n_subjects == 2
    assert group.subjects["subject_id"].tolist() == ["sub-01", "sub-02"]


def test_bids_h5_latent_accessors(latent_bids_h5) -> None:
    path, scan_name, basis, loadings, offset = latent_bids_h5
    dataset = bids_h5_dataset(path)

    assert dataset.compression_mode == "latent"
    assert dataset.subject_ids == ["sub-01"]
    assert sessions(dataset) == ["ses-1"]
    np.testing.assert_array_equal(dataset.get_data_matrix(), basis)

    loaded = get_loadings(dataset, scan_name=scan_name)
    np.testing.assert_array_equal(loaded, loadings)

    reconstructed = reconstruct_voxels(dataset, scan_name=scan_name)
    np.testing.assert_array_almost_equal(reconstructed, basis @ loadings.T + offset)

    subset = reconstruct_voxels(
        dataset,
        scan_name=scan_name,
        rows=[0, 2],
        voxels=[1, 3],
    )
    np.testing.assert_array_almost_equal(
        subset,
        (basis @ loadings.T + offset)[[0, 2]][:, [1, 3]],
    )

    info = encoding_info(dataset)
    assert info is not None
    assert info["encoding_family"] == "pca"
    assert info["encoding_params"] == {"center": True}
    assert info["n_components"] == 2


def test_bids_h5_latent_and_parcellated_mode_errors(
    parcellated_bids_h5,
    latent_bids_h5,
) -> None:
    parcellated_path, _matrices = parcellated_bids_h5
    latent_path, scan_name, _basis, _loadings, _offset = latent_bids_h5
    parcellated = bids_h5_dataset(parcellated_path)
    latent = bids_h5_dataset(latent_path)

    assert parcellation_info(latent) is None
    with pytest.raises(ValueError, match="latent-mode"):
        get_loadings(parcellated)
    with pytest.raises(ValueError, match="latent-mode"):
        reconstruct_voxels(parcellated, scan_name="sub-01_task-nback_run-1")
    with pytest.raises(ValueError, match="not found"):
        get_loadings(latent, scan_name="missing")
    with pytest.raises(ValueError, match="not found"):
        reconstruct_voxels(latent, scan_name="missing")

    subset = subset_bids_h5(latent, task="rest")
    assert subset.compression_mode == "latent"
    assert scan_manifest(subset)["scan_name"].tolist() == [scan_name]


def test_bids_h5_latent_shared_template_fallback(tmp_path) -> None:
    path = tmp_path / "shared_template_latent.h5"
    scan_name = "sub-01_task-rest_run-1"
    basis = np.asarray([[1.0, 0.5], [0.0, 1.0], [2.0, -1.0]])
    loadings = np.asarray([[1.0, 0.0], [0.5, 2.0], [-1.0, 1.0]])
    offset = np.asarray([0.1, 0.2, 0.3])

    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "bids_h5_study"
        handle.attrs["version"] = "1.0"
        handle.attrs["compression_mode"] = "latent"

        index = handle.create_group("scan_index")
        _write_str_array(index, "scan_name", [scan_name])
        _write_str_array(index, "subject", ["sub-01"])
        _write_str_array(index, "task", ["rest"])
        _write_str_array(index, "session", [""])
        _write_str_array(index, "run", ["1"])
        index.create_dataset("n_time", data=np.asarray([basis.shape[0]]))
        index.create_dataset("has_events", data=np.asarray([False]))
        index.create_dataset("has_confounds", data=np.asarray([False]))

        meta = handle.create_group("latent_meta")
        meta.create_dataset("n_components", data=2)
        _write_str_scalar(meta, "encoding_family", "pca")
        _write_str_scalar(meta, "encoding_params", json.dumps({"center": False}))
        meta.create_dataset("has_shared_template", data=True)
        template = meta.create_group("template")
        template.create_dataset("loadings", data=loadings)
        _write_str_scalar(template, "meta", json.dumps({"source": "template_project"}))

        scan = handle.create_group(f"scans/{scan_name}")
        data_group = scan.create_group("data")
        data_group.create_dataset("basis", data=basis)
        data_group.create_dataset("offset", data=offset)
        metadata = scan.create_group("metadata")
        metadata.create_dataset("tr", data=2.0)

    dataset = bids_h5_dataset(path)

    np.testing.assert_array_equal(get_loadings(dataset, scan_name=scan_name), loadings)
    np.testing.assert_array_almost_equal(
        reconstruct_voxels(dataset, scan_name=scan_name),
        basis @ loadings.T + offset,
    )
    info = encoding_info(dataset)
    assert info is not None
    assert info["has_shared_template"] is True
    assert info["template_meta"] == {"source": "template_project"}

    subset = subset_bids_h5(dataset, subject="sub-01")
    np.testing.assert_array_equal(get_loadings(subset, scan_name=scan_name), loadings)


def test_bids_h5_scan_backend_registry(parcellated_bids_h5) -> None:
    path, _matrices = parcellated_bids_h5
    dataset = bids_h5_dataset(path)
    first_backend = next(iter(dataset.scan_backends.values()))
    assert isinstance(first_backend, BidsH5ScanBackend)
    assert first_backend.get_dims().spatial == (3, 1, 1)
