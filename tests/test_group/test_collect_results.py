"""collect_results pairs BIDS maps by subject."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fmrimod.group import collect_results, meta_fe, reduce

nibabel = pytest.importorskip("nibabel")


def _write_map(path: Path, value: float, affine: np.ndarray) -> None:
    data = np.full((2, 2, 2), value, dtype=np.float64)
    img = nibabel.Nifti1Image(data, affine)
    path.parent.mkdir(parents=True, exist_ok=True)
    nibabel.save(img, str(path))


def test_collect_results_pairs_beta_and_se(tmp_path: Path) -> None:
    affine = np.eye(4)
    root = tmp_path / "study"
    for i, (beta, se) in enumerate(((0.2, 0.1), (0.4, 0.2), (0.3, 0.15)), start=1):
        _write_map(
            root / f"sub-{i:02d}_space-MNI_desc-beta_bold.nii.gz",
            beta,
            affine,
        )
        _write_map(
            root / f"sub-{i:02d}_space-MNI_desc-se_bold.nii.gz",
            se,
            affine,
        )
    ds = collect_results(root, space="MNI")
    assert ds.subjects == ("sub-01", "sub-02", "sub-03")
    assert "var" in ds.assays
    np.testing.assert_allclose(ds.assay("beta")[0, :, 0], [0.2, 0.4, 0.3])
    np.testing.assert_allclose(ds.assay("se")[0, :, 0], [0.1, 0.2, 0.15])
    reduced = reduce(ds, method="meta:fe")
    assert reduced.assay("beta_g").shape[1] == 1
    # meta:fe must run — this is not a synthetic unit-var placeholder.
    assert np.isfinite(reduced.assay("beta_g")[0, 0, 0])
    _ = meta_fe


def test_collect_results_beta_only_omits_var(tmp_path: Path) -> None:
    affine = np.eye(4)
    root = tmp_path / "study"
    _write_map(root / "sub-01_desc-beta_bold.nii.gz", 1.0, affine)
    _write_map(root / "sub-02_desc-beta_bold.nii.gz", 1.5, affine)
    ds = collect_results(root)
    assert "beta" in ds.assays
    assert "var" not in ds.assays
