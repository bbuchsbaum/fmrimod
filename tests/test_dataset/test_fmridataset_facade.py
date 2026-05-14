"""Compatibility import tests for the fmridataset facade package."""

from __future__ import annotations

import importlib

import fmridataset
import fmridataset.backend_constructors as facade_backend_constructors
import fmridataset.backend_protocol as facade_backend_protocol
import fmridataset.backend_registry as facade_backend_registry
import fmridataset.backends.latent_backend as facade_latent_backend
import fmridataset.backends.matrix_backend as facade_matrix_backend
import fmridataset.data_access as facade_data_access
import fmridataset.dataset as facade_dataset
import fmridataset.dataset_constructors as facade_constructors
import fmridataset.dataset_methods as facade_dataset_methods
import fmridataset.errors as facade_errors
import fmridataset.latent_dataset as facade_latent_dataset
import fmridataset.mask_utils as facade_mask_utils
import fmridataset.sampling_frame as facade_sampling
import fmridataset.selectors as facade_selectors
import fmrimod
import fmrimod.dataset as dataset
from fmrimod.sampling import SamplingFrame


def test_fmridataset_root_reexports_canonical_objects() -> None:
    facade = importlib.reload(fmridataset)

    assert facade.__version__ == fmrimod.__version__
    assert facade.FmriDataset is dataset.FmriDataset
    assert facade.MatrixBackend is dataset.MatrixBackend
    assert facade.LatentBackend is dataset.LatentBackend
    assert facade.LatentDataset is dataset.LatentDataset
    assert facade.matrix_dataset is dataset.matrix_dataset
    assert facade.fmri_dataset is dataset.fmri_dataset
    assert facade.matrix_backend is dataset.matrix_backend
    assert facade.latent_backend is dataset.latent_backend
    assert facade.latent_dataset is dataset.latent_dataset
    assert facade.get_data is dataset.get_data
    assert facade.DataChunk is dataset.DataChunk
    assert facade.data_chunks is dataset.data_chunks
    assert facade.SeriesSelector is dataset.SeriesSelector
    assert facade.FmriSeries is dataset.FmriSeries
    assert facade.fmri_series is dataset.fmri_series
    assert facade.get_TR is dataset.get_TR
    assert facade.mask_to_logical is dataset.mask_to_logical
    assert facade.SamplingFrame is SamplingFrame


def test_fmridataset_submodules_are_reexport_facades() -> None:
    facade_data_chunks = importlib.import_module("fmridataset.data_chunks")
    facade_fmri_series = importlib.import_module("fmridataset.fmri_series")

    assert facade_dataset.FmriDataset is dataset.FmriDataset
    assert facade_constructors.matrix_dataset is dataset.matrix_dataset
    assert facade_constructors.fmri_dataset is dataset.fmri_dataset
    assert facade_backend_constructors.matrix_backend is dataset.matrix_backend
    assert facade_backend_protocol.StorageBackend is dataset.StorageBackend
    assert facade_backend_protocol.BackendDims is dataset.BackendDims
    assert facade_backend_registry.BackendRegistry is dataset.BackendRegistry
    assert facade_backend_registry.create_backend is dataset.create_backend
    assert facade_matrix_backend.MatrixBackend is dataset.MatrixBackend
    assert facade_latent_backend.LatentBackend is dataset.LatentBackend
    assert (
        facade_latent_backend.InMemoryLatentBackend
        is dataset.InMemoryLatentBackend
    )
    assert facade_errors.FmriDatasetError is dataset.FmriDatasetError
    assert facade_errors.BackendIOError is dataset.BackendIOError
    assert facade_errors.ConfigError is dataset.ConfigError
    assert facade_sampling.SamplingFrame is SamplingFrame
    assert facade_data_access.get_data is dataset.get_data
    assert facade_data_access.get_data_matrix is dataset.get_data_matrix
    assert facade_data_access.get_mask is dataset.get_mask
    assert facade_data_chunks.DataChunk is dataset.DataChunk
    assert facade_data_chunks.data_chunks is dataset.data_chunks
    assert facade_data_chunks.voxel_index_chunks is dataset.voxel_index_chunks
    assert facade_dataset_methods.get_TR is dataset.get_TR
    assert facade_dataset_methods.blockids is dataset.blockids
    assert facade_dataset_methods.samples is dataset.samples
    assert facade_selectors.IndexSelector is dataset.IndexSelector
    assert facade_selectors.resolve_indices is dataset.resolve_indices
    assert facade_fmri_series.FmriSeries is dataset.FmriSeries
    assert facade_fmri_series.fmri_series is dataset.fmri_series
    assert facade_mask_utils.mask_to_logical is dataset.mask_to_logical
    assert facade_mask_utils.mask_to_volume is dataset.mask_to_volume
    assert facade_latent_dataset.LatentDataset is dataset.LatentDataset
    assert facade_latent_dataset.latent_dataset is dataset.latent_dataset
