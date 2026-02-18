"""Tests for high-level write_results orchestration."""

from pathlib import Path

import numpy as np
import pytest

import fmrimod
from fmrimod.io.results import write_results


class _DummySource:
    def get_affine(self):
        return np.eye(4, dtype=np.float64)


class _DummyDataset:
    def __init__(self):
        self._source = _DummySource()

    def get_mask(self):
        return np.ones((2, 2, 2), dtype=bool)


class _DummyModel:
    def __init__(self):
        self.dataset = _DummyDataset()


class _DummyResult:
    def __init__(self):
        self.betas = np.ones((2, 8), dtype=np.float64)
        self.contrasts = {"c1": object()}
        self.model = _DummyModel()


def test_write_results_delegates_and_moves_files(monkeypatch, tmp_path):
    calls = {"betas": 0, "contrasts": 0}

    def _stub_write_betas(*, output_dir, entities, **kwargs):
        calls["betas"] += 1
        p1 = Path(output_dir) / f"sub-{entities.subject}_betas.nii.gz"
        p2 = Path(output_dir) / f"sub-{entities.subject}_betas.json"
        p1.touch()
        p2.touch()
        return [p1, p2]

    def _stub_write_contrasts(*, output_dir, entities, **kwargs):
        calls["contrasts"] += 1
        p1 = Path(output_dir) / f"sub-{entities.subject}_contrast-c1_tstat.nii.gz"
        p2 = Path(output_dir) / f"sub-{entities.subject}_contrast-c1.json"
        p1.touch()
        p2.touch()
        return [p1, p2]

    monkeypatch.setattr("fmrimod.io.results.write_betas", _stub_write_betas)
    monkeypatch.setattr("fmrimod.io.results.write_contrasts", _stub_write_contrasts)

    out = tmp_path / "out"
    paths = write_results(
        _DummyResult(),
        out,
        subject="01",
        task="nback",
        save_betas=True,
        save_contrasts=True,
    )
    assert calls["betas"] == 1
    assert calls["contrasts"] == 1
    assert len(paths) == 4
    for p in paths:
        assert p.exists()
        assert p.parent == out


def test_write_results_requires_mask_when_not_inferable(monkeypatch, tmp_path):
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
            save_betas=False,
            save_contrasts=False,
        )


def test_write_results_allows_explicit_mask_and_affine(monkeypatch, tmp_path):
    captured = {}

    def _stub_write_betas(*, mask, affine, output_dir, entities, **kwargs):
        captured["mask"] = mask
        captured["affine"] = affine
        p = Path(output_dir) / "beta.nii.gz"
        p.touch()
        return [p]

    monkeypatch.setattr("fmrimod.io.results.write_betas", _stub_write_betas)
    monkeypatch.setattr("fmrimod.io.results.write_contrasts", lambda **kwargs: [])

    class _BareResult:
        def __init__(self):
            self.betas = np.ones((1, 8), dtype=np.float64)
            self.contrasts = {}

    mask = np.ones((2, 2, 2), dtype=bool)
    affine = np.eye(4, dtype=np.float64)
    out = write_results(
        _BareResult(),
        tmp_path / "out",
        subject="01",
        task="nback",
        mask=mask,
        affine=affine,
        save_contrasts=False,
    )
    assert len(out) == 1
    np.testing.assert_array_equal(captured["mask"], mask)
    np.testing.assert_allclose(captured["affine"], affine, atol=1e-12)


def test_write_results_overwrite_guard(monkeypatch, tmp_path):
    def _stub_write_betas(*, output_dir, **kwargs):
        p = Path(output_dir) / "collision.nii.gz"
        p.touch()
        return [p]

    monkeypatch.setattr("fmrimod.io.results.write_betas", _stub_write_betas)
    monkeypatch.setattr("fmrimod.io.results.write_contrasts", lambda **kwargs: [])

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "collision.nii.gz").touch()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_results(
            _DummyResult(),
            out_dir,
            subject="01",
            task="nback",
            save_contrasts=False,
            overwrite=False,
        )


def test_top_level_write_results_wrapper(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "fmrimod.io.results.write_betas",
        lambda **kwargs: [Path(kwargs["output_dir"]) / "x.nii.gz"],
    )
    monkeypatch.setattr("fmrimod.io.results.write_contrasts", lambda **kwargs: [])

    # Ensure stubbed path actually exists before move.
    def _stub_betas(**kwargs):
        p = Path(kwargs["output_dir"]) / "x.nii.gz"
        p.touch()
        return [p]

    monkeypatch.setattr("fmrimod.io.results.write_betas", _stub_betas)

    out = fmrimod.write_results(
        _DummyResult(),
        tmp_path / "out",
        subject="01",
        task="nback",
        save_contrasts=False,
    )
    assert len(out) == 1
    assert out[0].exists()

