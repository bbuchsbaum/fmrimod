"""Regression tests for cancellation-safe RSS recovery in the raw solver."""

from __future__ import annotations

import numpy as np

from fmrimod.glm.solver import fast_lm_matrix, fast_preproject
from fmrimod.glm.strategies import _fit_chunked_lm


def _boundary_case() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260513)
    n_timepoints = 80
    n_regressors = 4
    n_voxels = 6

    X = np.column_stack(
        [
            np.ones(n_timepoints),
            rng.standard_normal((n_timepoints, n_regressors - 1)),
        ]
    )
    betas = rng.standard_normal((n_regressors, n_voxels)) * 1e6
    Y = X @ betas + rng.standard_normal((n_timepoints, n_voxels)) * 1e-2
    return X, Y


def test_solver_fast_rss_recomputes_boundary_voxels_from_residuals() -> None:
    X, Y = _boundary_case()

    proj = fast_preproject(X)
    result = fast_lm_matrix(X, Y, proj, return_fitted=False)
    reference = fast_lm_matrix(X, Y, proj, return_fitted=True)

    np.testing.assert_allclose(result.betas, reference.betas, rtol=0, atol=0)
    np.testing.assert_allclose(result.rss, reference.rss, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result.sigma2,
        reference.sigma2,
        rtol=1e-12,
        atol=1e-12,
    )


def test_chunked_fast_rss_recomputes_boundary_voxels_from_residuals() -> None:
    X, Y = _boundary_case()

    proj = fast_preproject(X)
    result = _fit_chunked_lm(
        X,
        Y,
        proj,
        chunk_size=2,
        n_jobs=1,
        blas_threads=None,
        compute_dtype=np.float64,
    )
    reference = fast_lm_matrix(X, Y, proj, return_fitted=True)

    np.testing.assert_allclose(result.betas, reference.betas, rtol=0, atol=0)
    np.testing.assert_allclose(result.rss, reference.rss, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result.sigma2,
        reference.sigma2,
        rtol=1e-12,
        atol=1e-12,
    )
