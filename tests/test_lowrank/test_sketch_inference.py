"""Sketch-and-solve fits report inference for the sketched estimator.

Port of fmrireg ``test-sketch-inference.R`` / ``test-sketch-rank-safety.R``
(cdb5751, 923b662; R/sketch_time.R:169-247). For ``b_s = G^-1 Xs' S z`` with
``Xs = S X``, ``G = Xs'Xs`` and ``K = S S'``: ``Var(b_s | S) = sigma^2
G^-1 Xs' K Xs G^-1``, ``sigma^2 = ||r_s||^2 / tr(PK)`` and the Satterthwaite
df ``tr(PK)^2 / tr(PKPK)`` (about ``m - p``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.design import RealizedDesign
from fmrimod.glm import SketchEngineOptions
from fmrimod.lowrank.engine import LowRankConfig, fit_sketched

_NAMES = ["a", "b", "int", "lin"]


def _null_problem(T: int = 200, V: int = 4000, seed: int = 1):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    X = np.column_stack(
        [np.sin(t / 4.0), np.cos(t / 11.0), np.ones(T), np.linspace(-1.0, 1.0, T)]
    )
    Y = rng.standard_normal((T, V))
    events = pd.DataFrame({"onset": [0.0], "condition": ["A"], "duration": [0.0], "run": [1]})
    return X, Y, fm.fmri_dataset(Y, tr=2.0, events=events, run_length=[T])


@pytest.fixture(scope="module")
def null_problem():
    return _null_problem()


@pytest.mark.parametrize("kind", ["gaussian", "srht", "countsketch"])
@pytest.mark.parametrize("m", [40, 100])
def test_sketched_fit_is_calibrated_under_the_null(null_problem, kind, m):
    X, _, ds = null_problem
    fit = fm.fmri_lm(
        RealizedDesign.from_array(X, _NAMES),
        ds,
        engine=SketchEngineOptions(sketch_kind=kind, sketch_size=m, seed=3),
    )
    p_value = fit.contrast(np.array([1.0, 0.0, 0.0, 0.0]), name="a").p_value

    assert 0.035 <= float(np.mean(p_value < 0.05)) <= 0.065
    assert abs(float(np.mean(fit.sigma ** 2)) - 1.0) < 0.03
    assert fit.residual_df < m


def test_fit_sketched_matches_the_conditional_covariance_formula():
    X, Y, _ = _null_problem(V=50)
    cfg = LowRankConfig(sketch_kind="gaussian", sketch_size=60, seed=5)
    fit = fit_sketched(X, Y, cfg)

    from fmrimod.lowrank.sketch import make_sketch

    S = make_sketch(X.shape[0], 60, kind="gaussian", rng=np.random.default_rng(5))
    Xs, Ys = S @ X, S @ Y
    Q, R = np.linalg.qr(Xs)
    R_inv = np.linalg.inv(R)
    G_inv = R_inv @ R_inv.T
    K = S @ S.T
    cov = G_inv @ (Xs.T @ K @ Xs) @ G_inv
    KQ = K @ Q
    QKQ = Q.T @ KQ
    kappa = np.trace(K) - np.trace(QKQ)
    df = kappa ** 2 / (np.sum(K * K) - 2 * np.sum(KQ * KQ) + np.sum(QKQ * QKQ))
    rss = np.sum((Ys - Q @ (Q.T @ Ys)) ** 2, axis=0)

    np.testing.assert_allclose(fit.betas, R_inv @ (Q.T @ Ys), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fit.cov_unscaled, cov, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(fit.sigma2, rss / kappa, rtol=1e-10)
    np.testing.assert_allclose(fit.dfres, df, rtol=1e-10)


def test_unsketched_fit_keeps_exact_ols_inference():
    X, Y, _ = _null_problem(V=20)
    fit = fit_sketched(X, Y, LowRankConfig(sketch_ratio=1.0, seed=1))
    np.testing.assert_allclose(fit.cov_unscaled, np.linalg.inv(X.T @ X), rtol=1e-10)
    assert fit.dfres == X.shape[0] - X.shape[1]


def test_rank_deficient_sketch_raises_instead_of_pinv():
    # CountSketch maps each scan to one row: spike regressors collide.
    rng = np.random.default_rng(0)
    T = 120
    spikes = np.zeros((T, 6))
    spikes[rng.choice(T, size=6, replace=False), np.arange(6)] = 1.0
    X = np.column_stack([np.ones(T), spikes])
    Y = rng.standard_normal((T, 3))
    cfg = LowRankConfig(sketch_kind="countsketch", sketch_size=8, seed=2)
    with pytest.raises(ValueError, match="too small for this design"):
        fit_sketched(X, Y, cfg)
