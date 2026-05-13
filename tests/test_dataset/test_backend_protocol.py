"""Tests for canonical storage backend protocol contracts."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.dataset import BackendDims, ConfigError, MatrixBackend, StorageBackend


def test_backend_dims_validates_spatial_rank_and_time() -> None:
    dims = BackendDims(spatial=(3, 4, 5), time=10)
    assert dims.n_spatial == 60

    with pytest.raises(ConfigError, match="exactly 3"):
        BackendDims(spatial=(3, 4), time=10)
    with pytest.raises(ConfigError, match="time dimension"):
        BackendDims(spatial=(3, 4, 5), time=0)


def test_matrix_backend_is_storage_backend_and_validates() -> None:
    backend = MatrixBackend(np.ones((5, 3)))
    assert isinstance(backend, StorageBackend)
    assert backend.validate() is True
