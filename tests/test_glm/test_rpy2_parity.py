"""Optional cross-language parity tests for GLM soft-subspace paths.

These tests compare Python implementations against real R `fmrireg`
functions via rpy2. They are skipped automatically when rpy2 or fmrireg
is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from fmrimod.glm.preprocess import soft_subspace_projection
from fmrimod.glm.strategies import fit_run_ols
from fmrimod.model.config import FmriLmConfig, SoftSubspaceOptions, VolumeWeightOptions


@dataclass
class RContext:
    fmrireg: object
    FloatVector: object
    r: object


@pytest.fixture(scope="module")
def rctx() -> RContext:
    pytest.importorskip("rpy2")

    from rpy2.robjects.packages import PackageNotInstalledError, importr
    from rpy2.robjects.vectors import FloatVector
    import rpy2.robjects as ro

    try:
        fmrireg = importr("fmrireg")
    except PackageNotInstalledError as exc:
        pytest.skip(f"fmrireg R package not installed: {exc}")

    return RContext(fmrireg=fmrireg, FloatVector=FloatVector, r=ro.r)


def _to_r_matrix(arr: np.ndarray, rctx: RContext):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    n, p = arr.shape
    return rctx.r.matrix(rctx.FloatVector(arr.ravel(order="F")), nrow=n, ncol=p)


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


@pytest.mark.rpy2
def test_soft_subspace_projection_auto_matches_fmrireg(rctx):
    rng = np.random.default_rng(123)
    n, p, v, q = 60, 3, 8, 4
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    Y = rng.standard_normal((n, v))
    N = rng.standard_normal((n, q))

    py_Xc, py_Yc = soft_subspace_projection(X, Y, N, lam="auto")

    r_proj = rctx.fmrireg.soft_projection(
        _to_r_matrix(N, rctx),
        **{"lambda": "auto"},
        Y=_to_r_matrix(Y, rctx),
    )
    r_clean = rctx.fmrireg.apply_soft_projection(
        r_proj,
        _to_r_matrix(Y, rctx),
        _to_r_matrix(X, rctx),
    )
    r_Yc = _as_2d(np.array(r_clean.rx2("Y")))
    r_Xc = _as_2d(np.array(r_clean.rx2("X")))

    np.testing.assert_allclose(py_Xc, r_Xc, atol=1e-8)
    np.testing.assert_allclose(py_Yc, r_Yc, atol=1e-8)


@pytest.mark.rpy2
def test_soft_subspace_projection_gcv_matches_fmrireg(rctx):
    rng = np.random.default_rng(321)
    n, p, v, q = 70, 4, 6, 3
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    Y = rng.standard_normal((n, v))
    N = rng.standard_normal((n, q))

    py_Xc, py_Yc = soft_subspace_projection(X, Y, N, lam="gcv")

    r_proj = rctx.fmrireg.soft_projection(
        _to_r_matrix(N, rctx),
        **{"lambda": "gcv"},
        Y=_to_r_matrix(Y, rctx),
    )
    r_clean = rctx.fmrireg.apply_soft_projection(
        r_proj,
        _to_r_matrix(Y, rctx),
        _to_r_matrix(X, rctx),
    )
    r_Yc = _as_2d(np.array(r_clean.rx2("Y")))
    r_Xc = _as_2d(np.array(r_clean.rx2("X")))

    np.testing.assert_allclose(py_Xc, r_Xc, atol=3e-6)
    np.testing.assert_allclose(py_Yc, r_Yc, atol=3e-6)


@pytest.mark.rpy2
def test_fit_run_ols_soft_subspace_end_to_end_matches_fmrireg_pipeline(rctx):
    rng = np.random.default_rng(77)
    n, p, v, q = 64, 3, 5, 3
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    Y = rng.standard_normal((n, v))
    N = rng.standard_normal((n, q))

    cfg = FmriLmConfig(
        soft_subspace=SoftSubspaceOptions(enabled=True, nuisance_matrix=N, lam="auto")
    )
    py_result, _, py_X_used, py_Y_used = fit_run_ols(X, Y, cfg)

    r_proj = rctx.fmrireg.soft_projection(
        _to_r_matrix(N, rctx),
        **{"lambda": "auto"},
        Y=_to_r_matrix(Y, rctx),
    )
    r_clean = rctx.fmrireg.apply_soft_projection(
        r_proj,
        _to_r_matrix(Y, rctx),
        _to_r_matrix(X, rctx),
    )
    r_Yc = _as_2d(np.array(r_clean.rx2("Y")))
    r_Xc = _as_2d(np.array(r_clean.rx2("X")))
    r_fit = rctx.r["lm.fit"](_to_r_matrix(r_Xc, rctx), _to_r_matrix(r_Yc, rctx))
    r_betas = _as_2d(np.array(r_fit.rx2("coefficients")))
    r_resid = _as_2d(np.array(r_fit.rx2("residuals")))
    r_dfres = float(np.array(r_fit.rx2("df.residual")).ravel()[0])
    r_sigma2 = np.sum(r_resid ** 2, axis=0) / r_dfres

    np.testing.assert_allclose(py_X_used, r_Xc, atol=1e-8)
    np.testing.assert_allclose(py_Y_used, r_Yc, atol=1e-8)
    np.testing.assert_allclose(py_result.betas, r_betas, atol=1e-7)
    np.testing.assert_allclose(py_result.sigma2, r_sigma2, atol=1e-7)
    assert abs(py_result.dfres - r_dfres) < 1e-8


@pytest.mark.rpy2
def test_fit_run_ols_weighted_soft_subspace_end_to_end_matches_fmrireg_pipeline(rctx):
    rng = np.random.default_rng(91)
    n, p, v, q = 72, 2, 4, 2
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    Y = rng.standard_normal((n, v))
    N = rng.standard_normal((n, q))
    weights = np.linspace(0.2, 1.0, n)

    cfg = FmriLmConfig(
        volume_weights=VolumeWeightOptions(enabled=True, weights=weights),
        soft_subspace=SoftSubspaceOptions(enabled=True, nuisance_matrix=N, lam="gcv"),
    )
    py_result, _, py_X_used, py_Y_used = fit_run_ols(X, Y, cfg)

    sw = np.sqrt(weights)[:, np.newaxis]
    r_Xw = X * sw
    r_Yw = Y * sw
    r_proj = rctx.fmrireg.soft_projection(
        _to_r_matrix(N, rctx),
        **{"lambda": "gcv"},
        Y=_to_r_matrix(r_Yw, rctx),
    )
    r_clean = rctx.fmrireg.apply_soft_projection(
        r_proj,
        _to_r_matrix(r_Yw, rctx),
        _to_r_matrix(r_Xw, rctx),
    )
    r_Yc = _as_2d(np.array(r_clean.rx2("Y")))
    r_Xc = _as_2d(np.array(r_clean.rx2("X")))
    r_fit = rctx.r["lm.fit"](_to_r_matrix(r_Xc, rctx), _to_r_matrix(r_Yc, rctx))
    r_betas = _as_2d(np.array(r_fit.rx2("coefficients")))
    r_resid = _as_2d(np.array(r_fit.rx2("residuals")))
    r_dfres = float(np.array(r_fit.rx2("df.residual")).ravel()[0])
    r_sigma2 = np.sum(r_resid ** 2, axis=0) / r_dfres

    np.testing.assert_allclose(py_X_used, r_Xc, atol=5e-7)
    np.testing.assert_allclose(py_Y_used, r_Yc, atol=5e-7)
    np.testing.assert_allclose(py_result.betas, r_betas, atol=1e-6)
    np.testing.assert_allclose(py_result.sigma2, r_sigma2, atol=1e-6)
    assert abs(py_result.dfres - r_dfres) < 1e-8
