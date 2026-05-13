"""Tests for BIDS NIfTI export helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from fmrimod.bids.export import BidsEntities, write_betas, write_contrasts


def _read_volume(path) -> NDArray[np.float64]:
    from neuroim import read_image

    return np.asarray(read_image(path, type="vol").as_array(), dtype=np.float64)


def test_write_betas_uses_neuroim_nifti_writer(tmp_path) -> None:
    pytest.importorskip("neuroim")
    mask = np.array([[[True]], [[False]], [[True]]])
    betas = np.array([[1.5, -2.0]], dtype=np.float64)
    written = write_betas(
        betas,
        mask,
        np.eye(4),
        tmp_path,
        BidsEntities(subject="01", task="demo"),
        column_names=["cond_a"],
    )

    nii_paths = [path for path in written if path.suffix == ".gz"]
    assert len(nii_paths) == 1
    vol = _read_volume(nii_paths[0])
    assert vol.shape == mask.shape
    np.testing.assert_allclose(vol[mask], betas[0])
    np.testing.assert_allclose(vol[~mask], 0.0)


@dataclass
class _ContrastResult:
    estimate: NDArray[np.float64]
    stat: NDArray[np.float64]
    p_value: NDArray[np.float64]
    se: NDArray[np.float64]
    stat_type: str = "t"
    df: float = 10.0


def test_write_contrasts_uses_neuroim_nifti_writer(tmp_path) -> None:
    pytest.importorskip("neuroim")
    mask = np.array([[[True]], [[True]]])
    contrast = _ContrastResult(
        estimate=np.array([0.5, 0.25]),
        stat=np.array([2.0, 1.0]),
        p_value=np.array([0.01, 0.25]),
        se=np.array([0.25, 0.25]),
    )

    written = write_contrasts(
        {"a_gt_b": contrast},
        mask,
        np.eye(4),
        tmp_path,
        BidsEntities(subject="01", task="demo"),
        stats=("beta", "tstat"),
    )

    nii_paths = sorted(path for path in written if path.suffix == ".gz")
    assert len(nii_paths) == 2
    vols = {path.name: _read_volume(path) for path in nii_paths}
    beta_name = next(name for name in vols if "stat-beta" in name)
    t_name = next(name for name in vols if "stat-tstat" in name)
    np.testing.assert_allclose(vols[beta_name][mask], contrast.estimate)
    np.testing.assert_allclose(vols[t_name][mask], contrast.stat)
