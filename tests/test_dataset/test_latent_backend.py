"""Contracts for canonical latent storage backends."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.dataset import backend_get_loadings, latent_dataset
from fmrimod.dataset.backends.latent_backend import (
    InMemoryLatentBackend,
    LatentBackend,
)


def test_in_memory_latent_backend_reconstructs_voxels_and_metadata() -> None:
    scores = np.array(
        [
            [1.0, 0.5],
            [2.0, -1.0],
            [-0.5, 1.5],
            [0.25, 2.0],
        ]
    )
    loadings = np.array(
        [
            [1.0, 0.0],
            [0.5, 2.0],
            [-1.0, 0.25],
        ]
    )
    offset = np.array([0.1, -0.2, 0.3])

    backend = InMemoryLatentBackend(
        scores,
        loadings=loadings,
        offset=offset,
        run_lengths=[2, 2],
    )
    backend.open()

    assert backend.run_lengths == [2, 2]
    assert backend.get_dims().spatial == (3, 1, 1)
    assert backend.get_dims().time == 4
    np.testing.assert_allclose(
        backend.get_data(rows=np.array([1, 3]), cols=np.array([0])),
        scores[[1, 3]][:, [0]],
    )
    np.testing.assert_allclose(backend.get_loadings(1), loadings[:, [1]])
    np.testing.assert_allclose(backend_get_loadings(backend, 1), loadings[:, [1]])

    rows = np.array([0, 2])
    voxels = np.array([1, 2])
    expected = scores[rows] @ loadings[voxels].T + offset[voxels][np.newaxis, :]
    np.testing.assert_allclose(
        backend.reconstruct_voxels(rows=rows, voxels=voxels),
        expected,
    )

    metadata = backend.get_metadata()
    assert metadata["storage_format"] == "latent"
    assert metadata["n_components"] == 2
    assert metadata["n_voxels"] == 3
    assert metadata["n_runs"] == 2
    assert metadata["has_offset"] is True


def test_hdf5_latent_backend_reads_parts_and_reconstructs(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    basis_a = np.array([[1.0, 0.0], [0.5, 2.0]])
    basis_b = np.array([[2.0, -1.0], [0.0, 1.5], [1.25, 0.25]])
    basis = np.vstack([basis_a, basis_b])
    loadings = np.array(
        [
            [1.0, 0.0],
            [0.5, 2.0],
            [-1.0, 0.25],
        ]
    )
    offset = np.array([0.1, -0.2, 0.3])

    path_a = tmp_path / "run-a.h5"
    path_b = tmp_path / "run-b.h5"
    with h5py.File(path_a, "w") as handle:
        handle.create_dataset("basis", data=basis_a)
        handle.create_dataset("loadings", data=loadings)
        handle.create_dataset("offset", data=offset)
    with h5py.File(path_b, "w") as handle:
        handle.create_dataset("basis", data=basis_b)

    backend = LatentBackend([path_a, path_b])
    backend.open()

    assert backend.run_lengths == [2, 3]
    assert backend.get_dims().spatial == (3, 1, 1)
    assert backend.get_dims().time == 5
    np.testing.assert_allclose(backend.get_data(), basis)
    np.testing.assert_allclose(
        backend.reconstruct_voxels(),
        basis @ loadings.T + offset[np.newaxis, :],
    )
    np.testing.assert_allclose(backend.get_loadings([0, 1]), loadings)
    assert backend.get_metadata()["format"] == "latent_h5"


def test_latent_dataset_source_uses_storage_backed_backend(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    basis = np.array([[1.0, 0.0], [0.5, 2.0], [2.0, -1.0]])
    loadings = np.array(
        [
            [1.0, 0.0],
            [0.5, 2.0],
            [-1.0, 0.25],
        ]
    )
    path = tmp_path / "latent.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("basis", data=basis)
        handle.create_dataset("loadings", data=loadings)

    ds = latent_dataset(source=path, tr=1.25)

    assert ds.n_runs == 1
    assert ds.n_voxels == 3
    assert ds.sampling_frame.TR == 1.25
    np.testing.assert_allclose(ds.get_latent_scores(), basis)
    np.testing.assert_allclose(ds.get_data(0), basis @ loadings.T)
    assert ds.get_component_info()["storage_format"] == "latent"
