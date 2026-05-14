"""Optional dataset dependency failures should stay inside dataset errors."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from fmrimod.dataset import ConfigError, fmri_dataset
from fmrimod.dataset.adapters.neuroim_adapter import NeuroVecAdapter
from fmrimod.dataset.adapters.nibabel_adapter import NibabelAdapter


def _block_import(monkeypatch: pytest.MonkeyPatch, module: str) -> None:
    original_import: Callable[..., object] = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == module or name.startswith(f"{module}."):
            raise ImportError(f"blocked optional dependency: {module}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)


def test_nibabel_adapter_missing_dependency_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_import(monkeypatch, "nibabel")

    with pytest.raises(ConfigError, match=r"fmrimod\[nibabel\]") as excinfo:
        NibabelAdapter([object()], tr=2.0)

    assert excinfo.value.parameter == "nibabel"


def test_neurovec_adapter_missing_dependency_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_import(monkeypatch, "neuroim")

    with pytest.raises(ConfigError, match="pip install neuroim") as excinfo:
        NeuroVecAdapter(object(), tr=2.0)

    assert excinfo.value.parameter == "neuroim"


def test_neurovec_path_constructor_missing_dependency_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _block_import(monkeypatch, "neuroim")
    path = tmp_path / "sub-01_bold.nii.gz"
    path.write_bytes(b"")

    with pytest.raises(ConfigError, match="load NIfTI paths") as excinfo:
        NeuroVecAdapter.from_paths(path, tr=2.0)

    assert excinfo.value.parameter == "neuroim"


def test_neurovec_array_constructor_missing_dependency_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_import(monkeypatch, "neuroim")

    with pytest.raises(ConfigError, match="4-D ndarray input") as excinfo:
        NeuroVecAdapter.from_array(np.zeros((2, 2, 2, 3)), tr=2.0)

    assert excinfo.value.parameter == "neuroim"


def test_fmri_dataset_path_constructor_missing_dependency_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _block_import(monkeypatch, "neuroim")
    path = tmp_path / "sub-01_bold.nii.gz"
    path.write_bytes(b"")

    with pytest.raises(ConfigError, match="load NIfTI paths"):
        fmri_dataset(path, tr=2.0)
