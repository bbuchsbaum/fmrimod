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

    # Chunked GEMM (Pinv @ Y[:, slice]) and single-shot GEMM (Pinv @ Y)
    # disagree at BLAS round-off, so betas differ by ~1e-15 relative, and
    # the cancellation-safe direct-residual recomputation propagates that
    # to RSS at the ~|beta|^2 * eps level — far above 1e-12 with the
    # 1e6-scale betas this fixture uses. The point of the regression is
    # that the chunked path still routes boundary voxels through the
    # residual recomputation; tolerances reflect realised float64 noise.
    np.testing.assert_allclose(result.betas, reference.betas, rtol=1e-12, atol=0)
    np.testing.assert_allclose(result.rss, reference.rss, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        result.sigma2,
        reference.sigma2,
        rtol=1e-7,
        atol=1e-9,
    )


def _near_collinear_case() -> tuple[np.ndarray, np.ndarray]:
    """fmrireg ``.rd_near_collinear``: cond(X) ~ 2e7 on a large offset.

    The condition number sits inside the window where the Cholesky of
    ``X'X`` still "succeeds", so normal-equation betas lose most of their
    digits and ``yTy - b'X'y`` amplifies the error into the RSS.
    """
    rng = np.random.default_rng(11)
    n = 200
    x1 = rng.standard_normal(n)
    X = np.column_stack(
        [np.ones(n), x1, x1 + 1e-7 * rng.standard_normal(n), rng.standard_normal(n)]
    )
    B = np.column_stack([[0.0, 1.0, 2.0, -1.0], [0.0, -1.0, 0.5, 3.0]])
    Y = 1000.0 + X @ B + rng.standard_normal((n, 2))
    return X, Y


def _qr_reference(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Q, R = np.linalg.qr(X)
    betas = np.linalg.solve(R, Q.T @ Y)
    rss = np.sum((Y - X @ betas) ** 2, axis=0)
    return betas, rss


def test_near_collinear_design_matches_qr_solve_for_betas_and_rss() -> None:
    # fmrireg a369e95 (P-D4): betas and the exact RSS from one QR pass.
    X, Y = _near_collinear_case()
    assert 1e7 < np.linalg.cond(X) < 6.7e7
    ref_b, ref_rss = _qr_reference(X, Y)

    proj = fast_preproject(X)
    for return_fitted in (False, True):
        res = fast_lm_matrix(X, Y, proj, return_fitted=return_fitted)
        np.testing.assert_allclose(res.betas, ref_b, rtol=1e-6, atol=0)
        np.testing.assert_allclose(res.rss, ref_rss, rtol=1e-8, atol=0)
        np.testing.assert_allclose(res.sigma2, ref_rss / (X.shape[0] - 4), rtol=1e-8, atol=0)

    chunked = _fit_chunked_lm(
        X,
        Y,
        proj,
        chunk_size=1,
        n_jobs=1,
        blas_threads=None,
        compute_dtype=np.float64,
    )
    np.testing.assert_allclose(chunked.betas, ref_b, rtol=1e-6, atol=0)
    np.testing.assert_allclose(chunked.rss, ref_rss, rtol=1e-8, atol=0)

    # Inference on the well-determined column must not be corrupted.
    R_inv = np.linalg.inv(np.linalg.qr(X)[1])
    XtXinv_ref = R_inv @ R_inv.T
    np.testing.assert_allclose(proj.XtXinv[3, 3], XtXinv_ref[3, 3], rtol=1e-8)
