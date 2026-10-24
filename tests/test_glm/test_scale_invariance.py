"""t, F and run pooling do not depend on the scale of Y.

Port of fmrireg ``test-review-C-con-stats.R`` C7 (0f9138b) and the B10 /
P2 scale-invariance checks (004b0f3, 3ed5e8d): there is no absolute
tolerance on a standard error or residual variance; only an exactly zero
standard error is special (t = 0, p = 1, R ``.safe_tstat``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fmrimod as fm
from fmrimod.glm.combine import combine_contrasts
from fmrimod.glm.compat import compute_lm_contrasts_from_suffstats, fmri_ols_fit
from fmrimod.glm.contrasts import contrast_f, contrast_t
from fmrimod.glm.fmri_lm import FmriLm
from fmrimod.glm.matrix import fit_glm_from_suffstats
from fmrimod.glm.solver import LmResult, Projection
from fmrimod.glm.strategies import _pool_run_results
from fmrimod.model.config import FmriLmConfig

_T = np.array([0.0, 1.0, -1.0, 0.0])
_F = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])


def _cs_design(n: int = 60, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    Y = (X @ np.array([0.5, 1.0, -0.5, 0.25]))[:, None] + rng.standard_normal((n, 5))
    return X, Y


def _fast(X: np.ndarray, Y: np.ndarray):
    Q, R = np.linalg.qr(X)
    B = np.linalg.solve(R, Q.T @ Y)
    df = X.shape[0] - X.shape[1]
    sigma = np.sqrt(np.sum((Y - X @ B) ** 2, axis=0) / df)
    R_inv = np.linalg.inv(R)
    return B, sigma, R_inv @ R_inv.T, df


def _two_run_dataset(scale: float, seed: int = 3, n: int = 60, V: int = 4):
    rng = np.random.default_rng(seed)
    onsets = np.arange(4.0, 100.0, 12.0)
    cond = np.array(["A", "B"] * 10)[: len(onsets)]
    events = pd.DataFrame(
        {
            "onset": np.concatenate([onsets, onsets]),
            "condition": np.concatenate([cond, cond]),
            "duration": 0.0,
            "run": np.repeat([1, 2], len(onsets)),
        }
    )
    Y = (rng.standard_normal((2 * n, V)) + 100.0) * scale
    return fm.fmri_dataset(Y, tr=2.0, events=events, run_length=[n, n])


@pytest.mark.parametrize("scale", [1e-8, 1e-4, 1e4])
def test_contrast_t_and_f_are_scale_invariant(scale):
    X, Y = _cs_design()
    B0, s0, V0, df = _fast(X, Y)
    t0 = contrast_t(_T, B0, V0, s0, df)
    F0 = contrast_f(_F, B0, V0, s0, df)

    B, s, V, _ = _fast(X, Y * scale)
    t = contrast_t(_T, B, V, s, df)
    F = contrast_f(_F, B, V, s, df)

    np.testing.assert_allclose(t.stat, t0.stat, rtol=1e-8)
    np.testing.assert_allclose(t.p_value, t0.p_value, rtol=1e-8)
    np.testing.assert_allclose(F.stat, F0.stat, rtol=1e-8)
    np.testing.assert_allclose(F.p_value, F0.p_value, rtol=1e-8)


@pytest.mark.parametrize("scale", [1e-10, 1e-8, 1e-4, 1e4])
def test_runwise_pooled_betas_t_and_f_are_scale_invariant(scale):
    ref = fm.fmri_lm("hrf(condition)", _two_run_dataset(1.0))
    fit = fm.fmri_lm("hrf(condition)", _two_run_dataset(scale))

    np.testing.assert_allclose(fit.betas / scale, ref.betas, rtol=1e-8)
    np.testing.assert_allclose(fit.tstat(), ref.tstat(), rtol=1e-8)
    F_ref = ref.contrast(np.eye(4)[:2], name="F")
    F_fit = fit.contrast(np.eye(4)[:2], name="F")
    np.testing.assert_allclose(F_fit.stat, F_ref.stat, rtol=1e-8)
    np.testing.assert_allclose(F_fit.p_value, F_ref.p_value, rtol=1e-8)


@pytest.mark.parametrize("scale", [1e-8, 1e-12])
def test_suffstats_paths_are_scale_invariant(scale):
    X, Y = _cs_design()
    df = X.shape[0] - X.shape[1]

    def stats(Ys):
        fit = fit_glm_from_suffstats(None, X.T @ X, X.T @ Ys, np.sum(Ys * Ys, axis=0), df)
        con = compute_lm_contrasts_from_suffstats(
            X.T @ X, X.T @ Ys, np.sum(Ys * Ys, axis=0), df,
            contrasts={"t": _T}, output="list",
        )
        return fit.tstat(), con["t"].stat

    t_fit0, t_con0 = stats(Y)
    t_fit, t_con = stats(Y * scale)
    np.testing.assert_allclose(t_fit, t_fit0, rtol=1e-8)
    np.testing.assert_allclose(t_con, t_con0, rtol=1e-8)


@pytest.mark.parametrize("scale", [1.0, 1e-18, 1e-150])
def test_fmri_ols_fit_t_has_no_absolute_se_threshold(scale):
    # REG-B-07: R group OLS (fmrigds ols:voxelwise) has no SE threshold.
    X, Y = _cs_design()
    Ys = Y * scale
    B, sigma, XtXinv, _ = _fast(X, Ys)
    t_ref = B / (sigma[None, :] * np.sqrt(np.diag(XtXinv))[:, None])

    out = fmri_ols_fit(Ys, X)
    np.testing.assert_allclose(out["t"], t_ref, rtol=1e-8)

    vox = fmri_ols_fit(Ys, X, voxelwise=np.random.default_rng(1).standard_normal(Ys.shape))
    t_vox = np.asarray(vox["t"], dtype=float)
    assert np.all(np.isfinite(t_vox)) and np.all(t_vox != 0.0)


@pytest.mark.parametrize("method", ["ivw", "fixed"])
def test_combine_contrasts_t_is_scale_invariant(method):
    X, Y = _cs_design()

    def pooled(scale):
        parts = []
        for half in (slice(0, 30), slice(30, 60)):
            B, s, V, df = _fast(X[half], Y[half] * scale)
            parts.append(contrast_t(_T, B, V, s, df))
        return combine_contrasts(parts, method=method)

    for scale in (1e-10, 1e-20):
        np.testing.assert_allclose(pooled(scale).stat, pooled(1.0).stat, rtol=1e-8)


def test_zero_standard_error_gives_t_zero_p_one_on_every_path():
    # R .safe_tstat: an exactly zero SE (perfect fit) gives t = 0, p = 1,
    # including for a non-zero estimate.
    betas = np.array([[2.0, 0.0], [1.0, 3.0]])
    XtXinv = np.eye(2)
    sigma = np.array([0.0, 0.0])

    t = contrast_t(np.array([1.0, 0.0]), betas, XtXinv, sigma, 10.0)
    np.testing.assert_array_equal(t.stat, [0.0, 0.0])
    np.testing.assert_array_equal(t.p_value, [1.0, 1.0])

    F = contrast_f(np.eye(2), betas, XtXinv, sigma, 10.0)
    np.testing.assert_array_equal(F.stat, [0.0, 0.0])
    np.testing.assert_array_equal(F.p_value, [1.0, 1.0])

    fit = FmriLm(
        betas=betas, sigma=sigma, residual_df=10.0, XtXinv=XtXinv,
        model=None, config=FmriLmConfig(),  # type: ignore[arg-type]
    )
    np.testing.assert_array_equal(fit.tstat(), np.zeros((2, 2)))


def test_runwise_pooling_takes_the_zero_variance_limit():
    # R do_fixef (1733791): a run with se == 0 has infinite precision, so
    # it takes all the weight (equal weight among zero-variance runs).
    XtXinv = np.diag([0.5, 0.25])
    proj = Projection(
        Pinv=np.zeros((2, 4)), XtXinv=XtXinv, dfres=2.0, rank=2,
        is_full_rank=True, ill_conditioned=False,
    )

    def run(betas, sigma2):
        betas = np.asarray(betas, dtype=float)
        sigma2 = np.asarray(sigma2, dtype=float)
        return LmResult(
            betas=betas, rss=2.0 * sigma2, sigma2=sigma2, dfres=2.0, rank=2,
        )

    runs = [
        run([[1.0, 1.0], [2.0, 2.0]], [0.0, 1.0]),
        run([[3.0, 3.0], [4.0, 4.0]], [1.0, 1.0]),
        run([[7.0, 7.0], [8.0, 8.0]], [0.0, 4.0]),
    ]
    pooled = _pool_run_results(runs, [proj, proj, proj])
    # Voxel 0: runs 1 and 3 are exact fits, run 2 gets no weight.
    np.testing.assert_allclose(pooled["betas"][:, 0], [4.0, 5.0])
    # Voxel 1: ordinary inverse-variance weights 1 : 1 : 1/4.
    w = np.array([1.0, 1.0, 0.25])
    np.testing.assert_allclose(
        pooled["betas"][:, 1],
        [w @ [1.0, 3.0, 7.0] / w.sum(), w @ [2.0, 4.0, 8.0] / w.sum()],
    )
