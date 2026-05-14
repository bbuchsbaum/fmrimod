"""Canonical latent dataset contracts."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.dataset import FmriDataset, LatentDataset, latent_dataset
from fmrimod.dataset.backends.latent_backend import (
    InMemoryLatentBackend,
    LatentBackend,
)


def _events(n_scans: int, tr: float) -> pd.DataFrame:
    onsets = np.arange(0.0, n_scans * tr, 12.0)
    labels = np.array(["A", "B"] * ((len(onsets) + 1) // 2), dtype=object)[
        : len(onsets)
    ]
    return pd.DataFrame(
        {
            "onset": onsets,
            "duration": np.full(len(onsets), 6.0),
            "trial_type": labels,
            "run": np.ones(len(onsets), dtype=int),
        }
    )


def test_array_latent_dataset_is_canonical_fmri_dataset() -> None:
    rng = np.random.default_rng(20260514)
    scores = rng.normal(size=(60, 3))
    loadings = rng.normal(size=(3, 5))
    ds = latent_dataset(
        scores,
        loadings=loadings,
        tr=2.0,
        event_table=_events(scores.shape[0], 2.0),
    )

    assert isinstance(ds, LatentDataset)
    assert isinstance(ds, FmriDataset)
    assert isinstance(ds.storage_backend, InMemoryLatentBackend)
    assert ds.n_timepoints == 60
    assert ds.run_lengths == [60]
    assert ds.n_voxels == 5

    np.testing.assert_allclose(ds.get_latent_scores(), scores)
    np.testing.assert_allclose(ds.get_scores(0), scores)
    np.testing.assert_allclose(ds.get_spatial_loadings(), loadings.T)
    np.testing.assert_allclose(ds.get_data(0), scores @ loadings)
    np.testing.assert_allclose(
        ds.get_data(rows=np.array([0, 2]), cols=np.array([1])),
        (scores @ loadings)[[0, 2]][:, [1]],
    )


def test_storage_backed_latent_dataset_reads_hdf5_and_reconstructs(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    basis = np.array([[1.0, 0.0], [0.5, 2.0], [2.0, -1.0], [0.0, 1.5]])
    loadings = np.array([[1.0, 0.0], [0.5, 2.0], [-1.0, 0.25]])
    offset = np.array([0.1, -0.2, 0.3])
    path = tmp_path / "latent.lv.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("basis", data=basis)
        handle.create_dataset("loadings", data=loadings)
        handle.create_dataset("offset", data=offset)

    ds = latent_dataset(source=path, TR=1.5, run_length=0)

    assert isinstance(ds.storage_backend, LatentBackend)
    assert ds.n_runs == 1
    assert ds.n_timepoints == 4
    assert ds.TR == 1.5
    np.testing.assert_allclose(ds.get_latent_scores(cols=1), basis[:, [1]])
    np.testing.assert_allclose(ds.get_spatial_loadings(components=0), loadings[:, [0]])
    np.testing.assert_allclose(
        ds.reconstruct_voxels(rows=np.array([1, 3]), voxels=np.array([0, 2])),
        (basis @ loadings.T + offset[np.newaxis, :])[[1, 3]][:, [0, 2]],
    )
    assert ds.get_component_info()["format"] == "latent_h5"


def test_latent_dataset_can_feed_canonical_fmri_lm() -> None:
    rng = np.random.default_rng(31415)
    n_scans = 72
    tr = 2.0
    scores = rng.normal(size=(n_scans, 4))
    loadings = rng.normal(size=(4, 6))
    ds = fm.latent_dataset(
        scores,
        loadings=loadings,
        tr=tr,
        event_table=_events(n_scans, tr),
    )

    fit = fm.fmri_lm("hrf(trial_type)", ds)

    assert fit.n_voxels == loadings.shape[1]
    assert fit.betas.shape[1] == loadings.shape[1]


def test_fmridataset_latent_facades_are_identity_aliases() -> None:
    fmridataset = pytest.importorskip("fmridataset")
    facade_latent = importlib.import_module("fmridataset.latent_dataset")
    facade_backend = importlib.import_module("fmridataset.backends.latent_backend")

    assert fmridataset.LatentDataset is LatentDataset
    assert fmridataset.latent_dataset is latent_dataset
    assert facade_latent.LatentDataset is LatentDataset
    assert facade_latent.latent_dataset is latent_dataset
    assert facade_backend.LatentBackend is LatentBackend
    assert facade_backend.InMemoryLatentBackend is InMemoryLatentBackend
