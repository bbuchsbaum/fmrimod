"""collect_results pairs BIDS maps by subject."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fmrimod.group import collect_results, meta_fe, reduce
from fmrimod.group.errors import AdapterContractError

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


def _write_array(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nibabel.save(nibabel.Nifti1Image(data.astype(np.float64), affine), str(path))


def _study(root: Path, se_geometry: dict[int, tuple[tuple[int, ...], np.ndarray]]) -> None:
    """Two subjects on a (2, 3, 4) grid; ``se_geometry`` overrides SE shape/affine."""
    shape = (2, 3, 4)
    rng = np.random.default_rng(34)
    for i in (1, 2):
        beta = rng.normal(size=shape)
        _write_array(root / f"sub-{i:02d}_desc-beta_bold.nii.gz", beta, np.eye(4))
        se_shape, se_affine = se_geometry.get(i, (shape, np.eye(4)))
        se = np.arange(np.prod(se_shape), dtype=np.float64).reshape(se_shape) + i
        _write_array(root / f"sub-{i:02d}_desc-se_bold.nii.gz", se, se_affine)


def test_collect_results_matched_se_keeps_voxel_order(tmp_path: Path) -> None:
    _study(tmp_path, {})
    ds = collect_results(tmp_path)
    for j, i in enumerate((1, 2)):
        expected = np.arange(24, dtype=np.float64).reshape(2, 3, 4) + i
        np.testing.assert_array_equal(ds.assay("se")[:, j, 0], expected.reshape(-1))
    np.testing.assert_array_equal(ds.assay("var"), ds.assay("se") ** 2)


@pytest.mark.parametrize("subject", [1, 2])
def test_collect_results_rejects_se_with_same_size_other_shape(tmp_path: Path, subject: int) -> None:
    _study(tmp_path, {subject: ((4, 3, 2), np.eye(4))})
    with pytest.raises(AdapterContractError, match=rf"se map.*sub-{subject:02d}"):
        collect_results(tmp_path)


@pytest.mark.parametrize("subject", [1, 2])
@pytest.mark.parametrize(
    "affine",
    [
        np.array([[1.0, 0, 0, 5.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),  # translated
        np.diag([-1.0, 1.0, 1.0, 1.0]),  # left-right flip
    ],
    ids=["translated", "flipped"],
)
def test_collect_results_rejects_se_with_other_affine(
    tmp_path: Path, subject: int, affine: np.ndarray
) -> None:
    _study(tmp_path, {subject: ((2, 3, 4), affine)})
    with pytest.raises(AdapterContractError, match=rf"se map.*sub-{subject:02d}"):
        collect_results(tmp_path)
