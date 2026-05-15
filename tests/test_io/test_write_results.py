"""Tests for high-level write_results orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass

import nibabel as nib
import numpy as np
import pytest

import fmrimod
from fmrimod.io.results import ResultsManifest, write_results


@dataclass(frozen=True)
class _Column:
    name: str
    index: int
    role: str
    term: str | None = None
    condition: str | None = None
    basis_ix: int | None = None
    basis_name: str | None = None
    model_source: str | None = None


class _DesignColumns:
    def __init__(self, columns):
        self.columns = tuple(columns)

    def __iter__(self):
        return iter(self.columns)

    @property
    def names(self):
        return tuple(column.name for column in self.columns)


class _DummySource:
    def get_affine(self):
        return np.eye(4, dtype=np.float64)


class _DummyDataset:
    def __init__(self):
        self._source = _DummySource()

    def get_mask(self):
        return np.ones((2, 2, 2), dtype=bool)


class _DummyModel:
    def __init__(self, columns):
        self.dataset = _DummyDataset()
        self._columns = _DesignColumns(columns)

    def design_columns(self):
        return self._columns


class _DummyContrast:
    def __init__(self):
        self.estimate = np.arange(8, dtype=np.float64)
        self.stat = np.arange(8, dtype=np.float64) + 10
        self.se = np.ones(8, dtype=np.float64)
        self.p_value = np.full(8, 0.05, dtype=np.float64)
        self.stat_type = "t"
        self.df = 6


class _DummyResult:
    def __init__(self, columns=None):
        if columns is None:
            columns = [
                _Column("face", 0, "task", term="stim", condition="face"),
                _Column("house", 1, "task", term="stim", condition="house"),
                _Column("motion_x", 2, "nuisance", term="motion"),
                _Column("drift_1", 3, "drift", term="drift"),
            ]
        self.betas = np.vstack(
            [
                np.arange(8, dtype=np.float64),
                np.arange(8, dtype=np.float64) + 100,
                np.arange(8, dtype=np.float64) + 200,
                np.arange(8, dtype=np.float64) + 300,
            ]
        )
        self.contrasts = {"faces-vs-houses": _DummyContrast()}
        self.model = _DummyModel(columns)

    def design_columns(self):
        return self.model.design_columns()


def test_write_results_writes_manifest_and_task_beta_bundle(tmp_path):
    out = tmp_path / "out"
    manifest = write_results(
        _DummyResult(),
        out,
        subject="01",
        task="nback",
        contrasts=False,
    )

    assert isinstance(manifest, ResultsManifest)
    assert manifest.manifest_path.exists()
    assert len(manifest.files) == 1
    beta_file = manifest.files[0]
    assert beta_file.kind == "beta_bundle"
    assert beta_file.layout == "4d"
    assert beta_file.beta_group == "task"
    assert [volume.label for volume in beta_file.volumes] == ["face", "house"]

    img = nib.load(str(beta_file.path))
    data = np.asanyarray(img.dataobj)
    assert data.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(data[..., 0].reshape(-1), np.arange(8))
    np.testing.assert_allclose(data[..., 1].reshape(-1), np.arange(8) + 100)

    payload = json.loads(manifest.manifest_path.read_text())
    assert payload["schema_version"] == "fmrimod.results_manifest.v1"
    assert payload["entities"]["subject"] == "01"
    assert payload["files"][0]["path"] == beta_file.path.name
    assert payload["files"][0]["volumes"][0]["condition"] == "face"


def test_write_results_can_bundle_nuisance_betas_on_request(tmp_path):
    manifest = write_results(
        _DummyResult(),
        tmp_path / "out",
        subject="01",
        task="nback",
        betas="all",
        beta_groups=("task", "nuisance", "drift"),
        contrasts=False,
    )

    by_group = {file.beta_group: file for file in manifest.files}
    assert set(by_group) == {"task", "nuisance", "drift"}
    assert [volume.label for volume in by_group["nuisance"].volumes] == ["motion_x"]
    assert [volume.label for volume in by_group["drift"].volumes] == ["drift_1"]


def test_write_results_writes_contrast_statmaps_with_manifest(tmp_path):
    pytest.importorskip("neuroim")
    manifest = write_results(
        _DummyResult(),
        tmp_path / "out",
        subject="01",
        task="nback",
        betas=False,
        stats=("effect", "stat"),
    )

    assert [file.stat for file in manifest.files] == ["effect", "stat"]
    assert all(file.kind == "contrast_statmap" for file in manifest.files)
    assert all(file.path.exists() for file in manifest.files)
    assert "contrast-facesvshouses_stat-effect" in manifest.files[0].path.name
    assert "contrast-facesvshouses_stat-stat" in manifest.files[1].path.name


def test_write_results_requires_mask_when_not_inferable(tmp_path):
    class _NoMaskResult:
        def __init__(self):
            self.betas = np.ones((1, 4), dtype=np.float64)
            self.contrasts = {}
            self.model = object()

    with pytest.raises(ValueError, match="Could not infer mask"):
        write_results(
            _NoMaskResult(),
            tmp_path / "out",
            subject="01",
            task="nback",
            betas=False,
            contrasts=False,
        )


def test_write_results_allows_explicit_mask_and_affine(tmp_path):
    class _BareResult:
        def __init__(self):
            self.betas = np.ones((1, 8), dtype=np.float64)
            self.contrasts = {}

    mask = np.ones((2, 2, 2), dtype=bool)
    affine = np.eye(4, dtype=np.float64)
    manifest = write_results(
        _BareResult(),
        tmp_path / "out",
        subject="01",
        task="nback",
        mask=mask,
        affine=affine,
        column_names=["task_a"],
        contrasts=False,
    )

    assert len(manifest.files) == 1
    assert nib.load(str(manifest.files[0].path)).shape == (2, 2, 2, 1)


def test_write_results_overwrite_guard(tmp_path):
    out_dir = tmp_path / "out"
    write_results(
        _DummyResult(),
        out_dir,
        subject="01",
        task="nback",
        contrasts=False,
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_results(
            _DummyResult(),
            out_dir,
            subject="01",
            task="nback",
            contrasts=False,
            overwrite=False,
        )


def test_write_results_cleans_temporary_directory_on_failure(monkeypatch, tmp_path):
    def _boom(*args, **kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr("fmrimod.io.results._write_4d_nifti", _boom)

    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="write failed"):
        write_results(
            _DummyResult(),
            out_dir,
            subject="01",
            task="nback",
            contrasts=False,
        )

    assert out_dir.exists()
    assert not list(out_dir.glob(".write_results-*"))


def test_write_results_detects_sanitized_label_collisions(tmp_path):
    result = _DummyResult(
        columns=[
            _Column("a-b", 0, "task"),
            _Column("ab", 1, "task"),
            _Column("motion_x", 2, "nuisance"),
            _Column("drift_1", 3, "drift"),
        ]
    )

    with pytest.raises(ValueError, match="Sanitized label collision"):
        write_results(
            result,
            tmp_path / "out",
            subject="01",
            task="nback",
            contrasts=False,
        )


def test_top_level_write_results_wrapper(tmp_path):
    out = fmrimod.write_results(
        _DummyResult(),
        tmp_path / "out",
        subject="01",
        task="nback",
        contrasts=False,
    )
    assert isinstance(out, ResultsManifest)
    assert out[0].exists()
