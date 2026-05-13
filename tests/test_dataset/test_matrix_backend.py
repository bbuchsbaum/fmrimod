"""Tests for canonical in-memory matrix backend."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.dataset import BackendDims, ConfigError, MatrixBackend, matrix_backend


def test_matrix_backend_reports_dims_mask_and_metadata() -> None:
    mat = np.arange(12, dtype=float).reshape(4, 3)
    backend = MatrixBackend(mat, metadata={"source": "unit"})

    assert backend.get_dims() == BackendDims(spatial=(3, 1, 1), time=4)
    np.testing.assert_array_equal(backend.get_mask(), np.array([True, True, True]))
    assert backend.get_metadata() == {"format": "matrix", "source": "unit"}


def test_matrix_backend_constructor_is_canonical() -> None:
    mat = np.arange(12, dtype=float).reshape(4, 3)
    backend = matrix_backend(mat, metadata={"source": "constructor"})

    assert isinstance(backend, MatrixBackend)
    assert backend.get_dims() == BackendDims(spatial=(3, 1, 1), time=4)
    assert backend.get_metadata()["source"] == "constructor"


def test_matrix_backend_applies_mask_before_column_selection() -> None:
    mat = np.arange(20, dtype=float).reshape(5, 4)
    backend = MatrixBackend(mat, mask=np.array([True, False, True, True]))

    np.testing.assert_array_equal(backend.get_data(), mat[:, [0, 2, 3]])
    np.testing.assert_array_equal(
        backend.get_data(rows=np.array([1, 3]), cols=np.array([1])),
        mat[[1, 3]][:, [2]],
    )


def test_matrix_backend_validates_inputs() -> None:
    with pytest.raises(ConfigError, match="2-D"):
        MatrixBackend(np.zeros((2, 2, 2)))
    with pytest.raises(ConfigError, match="mask length"):
        MatrixBackend(np.zeros((2, 3)), mask=np.ones(2, dtype=bool))
    with pytest.raises(ConfigError, match="prod"):
        MatrixBackend(np.zeros((2, 3)), spatial_dims=(2, 2, 1))
