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
from fmrimod.glm.strategies import fit_run_ols, _pool_run_results
from fmrimod.model.config import FmriLmConfig, SoftSubspaceOptions, VolumeWeightOptions


@dataclass
class RContext:
    fmrireg: object
    FloatVector: object
    IntVector: object
    StrVector: object
    r: object


@pytest.fixture(scope="module")
def rctx() -> RContext:
    pytest.importorskip("rpy2")

    from rpy2.robjects.packages import PackageNotInstalledError, importr
    from rpy2.robjects.vectors import FloatVector, IntVector, StrVector
    import rpy2.robjects as ro

    try:
        fmrireg = importr("fmrireg")
    except PackageNotInstalledError as exc:
        pytest.skip(f"fmrireg R package not installed: {exc}")

    return RContext(
        fmrireg=fmrireg,
        FloatVector=FloatVector,
        IntVector=IntVector,
        StrVector=StrVector,
        r=ro.r,
    )


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


@pytest.mark.rpy2
def test_pool_run_results_matches_fmrireg_meta_betas(rctx):
    rng = np.random.default_rng(2026)
    n, p, v = 80, 3, 6

    X1 = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    X2 = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    true_betas = np.array([[0.1], [1.2], [-0.7]], dtype=np.float64)
    B = np.repeat(true_betas, v, axis=1)
    Y1 = X1 @ B + 0.2 * rng.standard_normal((n, v))
    Y2 = X2 @ B + 0.35 * rng.standard_normal((n, v))

    cfg = FmriLmConfig()
    py_res1, py_proj1, _, _ = fit_run_ols(X1, Y1, cfg)
    py_res2, py_proj2, _, _ = fit_run_ols(X2, Y2, cfg)
    pooled_py = _pool_run_results([py_res1, py_res2], [py_proj1, py_proj2])

    r_lm1 = rctx.r["lm.fit"](_to_r_matrix(X1, rctx), _to_r_matrix(Y1, rctx))
    r_lm2 = rctx.r["lm.fit"](_to_r_matrix(X2, rctx), _to_r_matrix(Y2, rctx))

    r_coef1 = _as_2d(np.array(r_lm1.rx2("coefficients")))
    r_coef2 = _as_2d(np.array(r_lm2.rx2("coefficients")))
    r_resid1 = _as_2d(np.array(r_lm1.rx2("residuals")))
    r_resid2 = _as_2d(np.array(r_lm2.rx2("residuals")))
    r_df1 = float(np.array(r_lm1.rx2("df.residual")).ravel()[0])
    r_df2 = float(np.array(r_lm2.rx2("df.residual")).ravel()[0])
    r_sigma1 = np.sqrt(np.sum(r_resid1**2, axis=0) / r_df1)
    r_sigma2 = np.sqrt(np.sum(r_resid2**2, axis=0) / r_df2)

    r_xtxinv1 = np.linalg.inv(X1.T @ X1)
    r_xtxinv2 = np.linalg.inv(X2.T @ X2)
    var_names = [f"b{i + 1}" for i in range(p)]

    r_beta_stats = rctx.r("fmrireg:::beta_stats_matrix")
    r_bstats1 = r_beta_stats(
        _to_r_matrix(r_coef1, rctx),
        _to_r_matrix(r_xtxinv1, rctx),
        rctx.FloatVector(r_sigma1),
        r_df1,
        rctx.StrVector(var_names),
    )
    r_bstats2 = r_beta_stats(
        _to_r_matrix(r_coef2, rctx),
        _to_r_matrix(r_xtxinv2, rctx),
        rctx.FloatVector(r_sigma2),
        r_df2,
        rctx.StrVector(var_names),
    )

    r_meta_betas = rctx.r("fmrireg:::meta_betas")
    r_bstats_list = rctx.r["list"](run1=r_bstats1, run2=r_bstats2)
    r_meta = r_meta_betas(
        r_bstats_list,
        rctx.IntVector(list(range(1, p + 1))),
        weighting="inv_var",
    )
    r_pooled = _as_2d(np.array(r_meta.rx2("data").rx2(1).rx2("estimate").rx2(1)))

    # fmrireg meta_betas stores estimate as V x p, while Python uses p x V.
    np.testing.assert_allclose(pooled_py["betas"], r_pooled.T, atol=1e-7)

    r_rss_total = np.sum(r_resid1**2, axis=0) + np.sum(r_resid2**2, axis=0)
    r_df_total = r_df1 + r_df2
    r_sigma_pool = np.sqrt(r_rss_total / r_df_total)
    np.testing.assert_allclose(pooled_py["sigma"], r_sigma_pool, atol=1e-8)
    assert abs(float(pooled_py["dfres"]) - r_df_total) < 1e-8
