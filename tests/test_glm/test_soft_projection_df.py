"""Residual df after soft subspace projection.

Port of fmrireg ``test-review-E-soft.R`` (E7, 5edfd16; wired into the
runwise fitter in c24c720). After ``Y <- P Y``, ``X <- P X`` the OLS residual
is ``(I - H) P eps``, so the unbiased residual df is ``tr((I - H) P^2)``,
not ``n - rank(PX)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.design import RealizedDesign
from fmrimod.glm.preprocess import soft_subspace_projection
from fmrimod.glm.strategies import fit_run_ols
from fmrimod.model.config import FmriLmConfig, SoftSubspaceOptions


def _dense_df(X: np.ndarray, N: np.ndarray, lam: float) -> float:
    n = X.shape[0]
    Xp, P = soft_subspace_projection(X, np.eye(n), N, lam)
    H = Xp @ np.linalg.solve(Xp.T @ Xp, Xp.T)
    return float(np.trace((np.eye(n) - H) @ P @ P))


def _soft_config(N: np.ndarray, lam: float | str) -> FmriLmConfig:
    return FmriLmConfig(
        soft_subspace=SoftSubspaceOptions(enabled=True, nuisance_matrix=N, lam=lam)
    )


@pytest.mark.parametrize("lam", [0.5, 50.0, 1e3])
def test_fit_run_ols_residual_df_is_trace_of_residual_projection(lam):
    rng = np.random.default_rng(1)
    n, k = 80, 12
    N = rng.standard_normal((n, k))
    X = np.column_stack([np.ones(n), rng.standard_normal(n), np.sin(np.arange(n) / 5.0)])
    Y = rng.standard_normal((n, 4))

    result, proj, _, _ = fit_run_ols(X, Y, _soft_config(N, lam))

    dense = _dense_df(X, N, lam)
    assert dense < n - X.shape[1]
    np.testing.assert_allclose(result.dfres, dense, rtol=1e-8)
    np.testing.assert_allclose(proj.dfres, dense, rtol=1e-8)
    np.testing.assert_allclose(result.sigma2, result.rss / dense, rtol=1e-12)


def test_soft_projection_residual_df_limits():
    rng = np.random.default_rng(3)
    n, k = 50, 5
    N = rng.standard_normal((n, k))
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    Y = rng.standard_normal((n, 2))

    soft, _, _, _ = fit_run_ols(X, Y, _soft_config(N, 1e12))
    np.testing.assert_allclose(soft.dfres, n - 2, rtol=1e-6)
    hard, _, _, _ = fit_run_ols(X, Y, _soft_config(N, 1e-12))
    np.testing.assert_allclose(hard.dfres, n - k - 2, rtol=1e-6)


def test_fmri_lm_soft_projection_sigma2_is_unbiased_under_the_null():
    rng = np.random.default_rng(21)
    n, V, k, lam = 200, 3000, 20, 50.0
    N = rng.standard_normal((n, k))
    X = np.column_stack(
        [np.sin(np.arange(n) / 6.0), np.cos(np.arange(n) / 9.0), np.ones(n)]
    )
    Y = rng.standard_normal((n, V))
    events = pd.DataFrame({"onset": [0.0], "condition": ["A"], "duration": [0.0], "run": [1]})
    ds = fm.fmri_dataset(Y, tr=2.0, events=events, run_length=[n])

    fit = fm.fmri_lm(
        RealizedDesign.from_array(X, ["a", "b", "int"]), ds, config=_soft_config(N, lam)
    )

    dense = _dense_df(X, N, lam)
    np.testing.assert_allclose(fit.residual_df, dense, rtol=1e-8)
    # n - rank would be 197 against ~178: sigma^2 biased low by ~10%.
    assert abs(float(np.mean(fit.sigma ** 2)) - 1.0) < 0.02
