"""Robust IRLS is a WLS fit with the weights it returns.

Ports the contract of fmrireg ``test-review-D-robust.R`` (D1-D3, fb00ab1):
each IRLS step weights the *base* run matrices (whitened when AR ran) by one
row-weight vector; the returned weights are the ones used for the final
betas; residual df = sum(w > 0) - rank (``stats::lm.wfit``) and the
inference df = sum(w) - rank (``.robust_effective_df``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

import fmrimod as fm
from fmrimod.accessors import p_values
from fmrimod.ar.estimation import fit_noise
from fmrimod.ar.whitening import ar_whiten_matrix
from fmrimod.dataset.data_access import get_run_data
from fmrimod.design import RealizedDesign
from fmrimod.glm.errors import FmriCapabilityError
from fmrimod.model.config import AROptions, FmriLmConfig, RobustOptions


def _spike_fixture():
    rng = np.random.default_rng(4)
    n, V = 120, 5
    t = np.arange(n)
    X = np.column_stack([np.sin(t / 5.0), np.cos(t / 7.0), np.ones(n)])
    Y = X @ np.array([[3.0] * V, [-2.0] * V, [50.0] * V]) + rng.standard_normal((n, V))
    Y[[6, 39, 40, 94]] += 25.0
    events = pd.DataFrame({"onset": [0.0], "condition": ["A"], "duration": [0.0], "run": [1]})
    ds = fm.fmri_dataset(Y, tr=2.0, events=events, run_length=[n])
    return X, Y, ds


def _wls(X, Y, w):
    sw = np.sqrt(w)[:, None]
    return np.linalg.lstsq(X * sw, Y * sw, rcond=None)[0]


def _r_irls(X, Y, c=4.685, max_iter=2, tol=1e-4):
    """Transcription of R robust_iterative_fitter (bisquare, scale_scope="run")."""
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    for _ in range(max_iter):
        B_prev = B
        res = np.abs(Y - X @ B)
        sigma = 1.4826 * np.median(np.median(res, axis=1))
        u = np.median(res / sigma, axis=1)
        w = np.where(u <= c, (1.0 - (u / c) ** 2) ** 2, 0.0)
        B = _wls(X, Y, w)
        if np.max(np.abs(B - B_prev)) / max(1.0, np.max(np.abs(B_prev))) <= tol:
            break
    return B, w


def _fit(ds, X, **robust):
    cfg = FmriLmConfig(robust=RobustOptions(**robust))
    return fm.fmri_lm(RealizedDesign.from_array(X, ["a", "b", "int"]), ds, config=cfg)


def test_bisquare_irls_is_wls_of_original_data_with_returned_weights():
    X, Y, ds = _spike_fixture()
    fit = _fit(ds, X, type="bisquare", max_iter=2)
    w = fit.robust_weights

    assert w.shape == (X.shape[0],)
    assert np.sum(w == 0.0) == 4  # the spikes, so iteration 2 must bite
    np.testing.assert_allclose(fit.betas, _wls(X, Y, w), rtol=0, atol=1e-8)
    assert fit.residual_df == int(np.sum(w > 0)) - 3

    B_r, w_r = _r_irls(X, Y)
    np.testing.assert_allclose(w, w_r, rtol=0, atol=1e-10)
    np.testing.assert_allclose(fit.betas, B_r, rtol=0, atol=1e-8)

    sw = np.sqrt(w)[:, None]
    rss = np.sum(((Y - X @ fit.betas) * sw) ** 2, axis=0)
    np.testing.assert_allclose(fit.sigma ** 2, rss / fit.residual_df, rtol=1e-10)


def test_robust_contrast_df_is_sum_of_weights_minus_rank():
    X, Y, ds = _spike_fixture()
    fit = _fit(ds, X, type="huber", max_iter=3)
    w = fit.robust_weights

    con = fit.contrast(np.array([1.0, 0.0, 0.0]), name="a")
    np.testing.assert_allclose(con.df, float(np.sum(w)) - 3, rtol=1e-12)
    assert fit.residual_df == int(np.sum(w > 0)) - 3


def test_estimate_p_values_use_the_robust_inference_df():
    X, Y, ds = _spike_fixture()
    fit = _fit(ds, X, type="huber", max_iter=3)
    df = float(np.sum(fit.robust_weights)) - 3
    expected = 2.0 * sp_stats.t.sf(np.abs(fit.tstat()), df=df)
    np.testing.assert_allclose(p_values(fit), expected, rtol=1e-12)
    con = fit.contrast(np.array([1.0, 0.0, 0.0]), name="a")
    np.testing.assert_allclose(p_values(fit)[0], con.p_value, rtol=1e-10)


@pytest.mark.parametrize("scope", ["run", "voxel", "global"])
def test_every_scale_scope_returns_row_weights_used_for_the_fit(scope):
    X, Y, ds = _spike_fixture()
    fit = _fit(ds, X, type="huber", max_iter=2, scale_scope=scope)
    w = fit.robust_weights
    assert w.shape == (X.shape[0],)
    np.testing.assert_allclose(fit.betas, _wls(X, Y, w), rtol=0, atol=1e-8)


def test_robust_composes_on_volume_weighted_rows():
    # The IRLS weights multiply the volume weights already applied to the
    # run; a zero-volume-weight row is not an observation (fmrireg
    # obs_weights in robust_iterative_fitter).
    from fmrimod.model.config import VolumeWeightOptions

    X, Y, ds = _spike_fixture()
    vw = np.random.default_rng(9).uniform(0.4, 1.0, X.shape[0])
    vw[[10, 11, 70]] = 0.0
    cfg = FmriLmConfig(
        robust=RobustOptions(type="huber", max_iter=3),
        volume_weights=VolumeWeightOptions(weights=vw),
    )
    fit = fm.fmri_lm(RealizedDesign.from_array(X, ["a", "b", "int"]), ds, config=cfg)
    w = fit.robust_weights

    assert np.all(w[[10, 11, 70]] == 0.0)
    np.testing.assert_allclose(fit.betas, _wls(X, Y, w * vw), rtol=0, atol=1e-8)
    assert fit.residual_df == int(np.sum(w > 0)) - 3


def _ar1_two_run(phi=0.5, n=150, V=20, seed=21):
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
    Y = np.empty((2 * n, V))
    for col in range(V):
        for run in range(2):
            e = rng.standard_normal(n)
            x = np.zeros(n)
            for i in range(n):
                x[i] = e[i] + (phi * x[i - 1] if i else 0.0)
            Y[run * n:(run + 1) * n, col] = x
    return fm.fmri_dataset(Y + 100.0, tr=2.0, events=events, run_length=[n, n])


def test_ar_plus_robust_weights_the_whitened_data():
    ds = _ar1_two_run()
    ar = fm.fmri_lm("hrf(condition)", ds, config=FmriLmConfig(ar=AROptions(struct="ar1")))

    # With every weight 1 the robust refit must reproduce the AR fit exactly:
    # whitened X paired with whitened Y.
    flat = fm.fmri_lm(
        "hrf(condition)",
        ds,
        config=FmriLmConfig(
            ar=AROptions(struct="ar1"), robust=RobustOptions(type="huber", k_huber=1e6)
        ),
    )
    np.testing.assert_array_equal(flat.robust_weights, np.ones(300))
    np.testing.assert_allclose(flat.betas, ar.betas, rtol=1e-10, atol=1e-10)

    huber = fm.fmri_lm(
        "hrf(condition)",
        ds,
        config=FmriLmConfig(ar=AROptions(struct="ar1"), robust=RobustOptions(type="huber")),
    )
    se = ar.se()[:2]
    assert np.max(np.abs(huber.betas[:2] - ar.betas[:2]) / se) < 0.05
    ranks = sum(p.rank for p in huber.projections)
    assert huber.residual_df == int(np.sum(huber.robust_weights > 0)) - ranks


def test_reestimate_phi_refits_with_phi_from_unwhitened_robust_residuals():
    # fmrireg process_run_ar_robust (reestimate_phi = TRUE): phi is
    # re-estimated from Y - X b_robust on the unwhitened run, without a design
    # correction; the run is re-whitened and the IRLS redone on it.
    n = 150
    ds = _ar1_two_run(n=n)
    once = fm.fmri_lm(
        "hrf(condition)",
        ds,
        config=FmriLmConfig(ar=AROptions(struct="ar1"), robust=RobustOptions(type="huber")),
    )
    fit = fm.fmri_lm(
        "hrf(condition)",
        ds,
        config=FmriLmConfig(
            ar=AROptions(struct="ar1"),
            robust=RobustOptions(type="huber", reestimate_phi=True),
        ),
    )
    assert not np.allclose(fit.ar_params, once.ar_params)
    for r in range(2):
        X = fit.model.design_matrix_array(run=r)
        Y = get_run_data(ds, r)
        resid = Y - X @ once.run_results[r].betas
        phi = fit_noise(resid, method="ar", p=1, p_max=1).phi[0]
        np.testing.assert_allclose(fit.ar_params[r], phi, rtol=0, atol=1e-12)
        Xw, Yw = ar_whiten_matrix(X, Y, phi)
        w = fit.robust_weights[r * n : (r + 1) * n]
        np.testing.assert_allclose(
            fit.run_results[r].betas, _wls(Xw, Yw, w), rtol=0, atol=1e-8
        )


def test_reestimate_phi_with_global_ar_is_refused():
    # Global pooling holds one phi fixed across runs; fmrireg skips the
    # re-estimation there, which would silently ignore the option.
    ds = _ar1_two_run(n=60, V=4)
    cfg = FmriLmConfig(
        ar=AROptions(struct="ar1", pooling="global"),
        robust=RobustOptions(type="huber", reestimate_phi=True),
    )
    with pytest.raises(FmriCapabilityError, match="reestimate_phi"):
        fm.fmri_lm("hrf(condition)", ds, config=cfg)
