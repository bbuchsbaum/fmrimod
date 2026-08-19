"""Wave 2: create_lwu_grid composes with build_sbhm_library.

Guards fmrilss ``f851fb0``. Cheap pass disqualified: returning a
parameter table that ``build_sbhm_library`` cannot consume.
"""

from __future__ import annotations

from fmrimod.sampling import SamplingFrame
from fmrimod.single.sbhm import LwuGrid, build_sbhm_library, create_lwu_grid


def test_create_lwu_grid_composes_with_sbhm_library() -> None:
    sframe = SamplingFrame(blocklens=60, tr=1.0)
    grid = create_lwu_grid(n_tau=2, n_sigma=2, n_rho=2)
    assert isinstance(grid, LwuGrid)
    assert len(grid.parameters) == 8
    assert set(grid.parameters.columns) >= {"tau", "sigma", "rho"}

    library_H = grid.to_library_H(sframe.samples)
    assert library_H.shape == (len(sframe.samples), 8)
    assert library_H.shape[0] == 60

    built = build_sbhm_library(library_H, r=2)
    assert built.B.shape == (60, 2)
    assert built.A.shape == (8, 2)
    assert built.library_H.shape[1] == len(grid.parameters)


def test_create_lwu_grid_hrfs_are_height_normalized() -> None:
    sframe = SamplingFrame(blocklens=40, tr=1.0)
    grid = create_lwu_grid(n_tau=2, n_sigma=1, n_rho=1, tau_range=(5.0, 7.0))
    H = grid.to_library_H(sframe.samples)
    peaks = H.max(axis=0)
    assert peaks.min() > 0.9
    assert peaks.max() <= 1.0 + 1e-8
