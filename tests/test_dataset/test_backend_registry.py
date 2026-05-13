"""Tests for canonical dataset backend registry."""

from __future__ import annotations

import numpy as np
import pytest

from fmrimod.dataset import (
    BackendRegistry,
    ConfigError,
    MatrixBackend,
    create_backend,
    get_backend_registry,
    is_backend_registered,
    list_backend_names,
    register_backend,
    unregister_backend,
)


def test_builtin_matrix_backend_is_registered() -> None:
    assert is_backend_registered("matrix")
    assert "matrix" in list_backend_names()

    backend = create_backend("matrix", data_matrix=np.ones((4, 2)))
    assert isinstance(backend, MatrixBackend)


def test_registry_metadata_lookup() -> None:
    info = get_backend_registry("matrix")
    assert info["name"] == "matrix"
    assert "In-memory" in info["description"]


def test_register_and_unregister_custom_backend() -> None:
    registry = get_backend_registry()
    assert isinstance(registry, BackendRegistry)

    def factory() -> MatrixBackend:
        return MatrixBackend(np.ones((2, 1)))

    register_backend("tmp_matrix_test", factory, overwrite=True)
    try:
        assert is_backend_registered("tmp_matrix_test")
        assert isinstance(create_backend("tmp_matrix_test"), MatrixBackend)
        with pytest.raises(ConfigError, match="already registered"):
            register_backend("tmp_matrix_test", factory)
    finally:
        assert unregister_backend("tmp_matrix_test") is True
