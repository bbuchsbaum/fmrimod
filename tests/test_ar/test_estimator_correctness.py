"""Transcribed from fmriAR ``tests/testthat/test-estimator-correctness.R``.

Cheap-pass disqualifier: storing ``plan.censor`` while still fragment-centering
the ACVF cannot recover φ under 20–40% scrubbing, honour parcel censoring, or
keep lag-1 of two-frame fragments positive.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fmrimod.ar.acvf import (
    acvf_from_pooled,
    pooled_acvf_segments,
    sigma2_from_gamma_phi,
    valid_segments,
)
from fmrimod.ar.estimation import fit_noise
from fmrimod.ar.numhelpers import ar_to_pacf, enforce_stationary_ar, pacf_to_ar
from fmrimod.ar.plan import plan_from_phi
from fmrimod.ar.whitening import whiten_apply


def _ar_sim(n: int, phi, nvox: int = 1, sd: float = 1.0, seed: int | None = None):
    rng = np.random.RandomState(seed)
    phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    p = int(phi.size)
    burn = 200
    out = np.zeros((n, nvox))
    for j in range(nvox):
        y = rng.randn(n + burn) * sd
        for t in range(1, n + burn):
            lagsum = 0.0
            for k in range(min(p, t)):
                lagsum += float(phi[k]) * y[t - k - 1]
            y[t] += lagsum
        out[:, j] = y[burn:]
    return out


def _plan_phi(plan, parcel: str | None = None):
    if plan.phi_by_parcel is not None:
        keys = list(plan.phi_by_parcel.keys())
        key = parcel if parcel is not None else keys[0]
        return np.asarray(plan.phi_by_parcel[key], dtype=np.float64)
    assert plan.phi is not None
    return np.asarray(plan.phi[0], dtype=np.float64)


def _min_root(phi) -> float:
    phi = np.asarray(phi, dtype=np.float64).ravel()
    nz = np.where(phi != 0)[0]
    if nz.size:
        phi = phi[: int(nz[-1]) + 1]
    if phi.size == 0:
        return float("inf")
    coeffs = np.concatenate([[1.0], -phi])
    roots = np.roots(coeffs[::-1])
    return float(np.min(np.abs(roots))) if roots.size else float("inf")


def _lag1(M: np.ndarray) -> float:
    acc = []
    for j in range(M.shape[1]):
        y = M[:, j] - M[:, j].mean()
        acc.append(float(np.dot(y[1:], y[:-1]) / np.dot(y, y)))
    return float(np.mean(acc))


def test_every_pooling_mode_recovers_known_ar1():
    resid = _ar_sim(400, 0.5, nvox=40, seed=101)
    parcels = np.tile(np.arange(1, 5), 10)
    for pspec in (1, "auto"):
        plans = {
            "global": fit_noise(resid, pooling="global", method="ar", p=pspec, p_max=4),
            "run": fit_noise(
                resid,
                runs=np.repeat([1, 2], 200),
                pooling="run",
                method="ar",
                p=pspec,
                p_max=4,
            ),
            "parcel": fit_noise(
                resid,
                parcels=parcels,
                pooling="parcel",
                method="ar",
                p=pspec,
                p_max=4,
            ),
        }
        for nm, plan in plans.items():
            phi = _plan_phi(plan)
            assert abs(phi[0] - 0.5) < 0.08, f"{nm} p={pspec}"


def test_censoring_does_not_attenuate_ar_estimate():
    truth = 0.6
    for frac in (0.0, 0.2, 0.3, 0.4):
        est = []
        for s in range(1, 6):
            resid = _ar_sim(250, truth, nvox=30, seed=800 + s)
            rng = np.random.RandomState(900 + s)
            cens = None
            if frac > 0:
                cens = np.sort(
                    rng.choice(250, size=int(round(frac * 250)), replace=False)
                )
            plan = fit_noise(resid, censor=cens, pooling="global", method="ar", p=1)
            est.append(_plan_phi(plan)[0])
        assert abs(float(np.mean(est)) - truth) < 0.12, f"censor fraction {frac}"


def test_short_fragments_do_not_force_correlation_of_minus_one():
    y = _ar_sim(400, 0.7, nvox=1, seed=2101).ravel()
    keep = np.where(np.resize([True, True, False], 400))[0]
    seg = np.cumsum(np.concatenate([[1], (np.diff(keep) > 1).astype(np.intp)]))
    pooled = pooled_acvf_segments(
        y[keep][:, None],
        seg,
        1,
        center_id=np.ones(keep.size, dtype=np.intp),
    )
    g, _ = acvf_from_pooled(pooled)
    assert g[1] / g[0] > 0.4


def test_parcel_pooling_honours_censor():
    resid = _ar_sim(300, 0.5, nvox=20, seed=1401)
    parcels = np.tile(np.arange(1, 5), 5)
    cens = np.arange(9, 290, 10)
    spoiled = resid.copy()
    spoiled[cens] += 500
    uncensored = fit_noise(spoiled, parcels=parcels, pooling="parcel", method="ar", p=1)
    censored = fit_noise(
        spoiled, parcels=parcels, pooling="parcel", method="ar", p=1, censor=cens
    )
    assert not np.allclose(_plan_phi(uncensored), _plan_phi(censored))
    assert abs(_plan_phi(censored)[0] - 0.5) < 0.12


def test_parcel_fixed_p_is_not_bic_shrunk():
    resid = _ar_sim(400, [0.6, 0.25], nvox=40, seed=201)
    parcels = np.tile(np.arange(1, 5), 10)
    plan = fit_noise(resid, parcels=parcels, pooling="parcel", method="ar", p=2)
    phi = _plan_phi(plan)
    assert phi.size == 2
    assert plan.order[0] == 2


def test_censored_frames_do_not_leak():
    resid = _ar_sim(300, 0.5, nvox=20, seed=1301)
    cens = np.arange(9, 290, 10)
    spoiled = resid.copy()
    spoiled[cens] += 500
    a = fit_noise(resid, censor=cens, pooling="global", method="ar", p=2)
    b = fit_noise(spoiled, censor=cens, pooling="global", method="ar", p=2)
    np.testing.assert_allclose(_plan_phi(a), _plan_phi(b), atol=1e-8)


def test_whitening_censored_fit_reduces_variance():
    resid = _ar_sim(200, 0.6, nvox=25, seed=1201)
    rng = np.random.RandomState(1202)
    cens = np.sort(rng.choice(200, size=60, replace=False))
    X = np.column_stack([np.ones(200), rng.randn(200)])
    plan = fit_noise(
        resid, censor=cens, pooling="global", method="ar", p="auto", p_max=6
    )
    out = whiten_apply(plan, X, resid)
    assert float(np.var(out.Y)) < float(np.var(resid))


def test_per_run_mean_offsets_do_not_contaminate():
    rng = np.random.RandomState(1501)
    n, nv = 400, 40
    runs = np.repeat(np.arange(1, 5), 100)
    white = rng.randn(n, nv)
    offsets = np.repeat(rng.randn(4) * 10, 100)[:, None]
    shifted = white + offsets
    for pooling in ("global", "run", "parcel"):
        kwargs = {
            "resid": shifted,
            "runs": runs,
            "pooling": pooling,
            "method": "ar",
            "p": 1,
        }
        if pooling == "parcel":
            kwargs["parcels"] = np.tile(np.arange(1, 5), 10)
        plan = fit_noise(**kwargs)
        assert abs(_plan_phi(plan)[0]) < 0.15, pooling


def test_order_selection_bounded_by_sample_size():
    rng = np.random.RandomState(2801)
    short = rng.randn(11, 5)
    auto = fit_noise(short, pooling="global", method="ar", p="auto", p_max=8)
    assert auto.order[0] <= 2
    X = np.column_stack([np.ones(11), rng.randn(11)])
    w = whiten_apply(auto, X, short)
    assert float(np.std(w.Y) / np.std(short)) < 1.05
    fixed = fit_noise(rng.randn(25, 5), pooling="global", method="ar", p=4)
    assert _plan_phi(fixed).size == 4


def test_plan_carries_noise_scale():
    a = _ar_sim(400, 0.5, nvox=20, seed=3301)
    b = a * 10
    pa = fit_noise(a, pooling="global", method="ar", p=1)
    pb = fit_noise(b, pooling="global", method="ar", p=1)
    np.testing.assert_allclose(_plan_phi(pa), _plan_phi(pb), atol=1e-8)
    assert abs(pb.sigma2[0] / pa.sigma2[0] - 100) < 1e-4
    assert abs(pb.gamma[0][0] / pa.gamma[0][0] - 100) < 1e-4


def test_psd_projection_repairs_non_psd_acvf():
    bad = [
        (np.array([1.0, 1.4]), np.array([10.0, 9.0])),
        (np.array([1.0, -0.99, 0.98, -0.97]), np.array([50.0, 40.0, 30.0, 5.0])),
        (np.array([1.0, 0.2, 0.1, 0.9]), np.array([100.0, 90.0, 80.0, 3.0])),
        (np.array([1.0, 0.1, 1.3, 0.1]), np.array([60.0, 50.0, 4.0, 30.0])),
    ]
    ran = 0
    for num, pairs in bad:
        g_unb = num / pairs
        toe = np.empty((g_unb.size, g_unb.size))
        for i in range(g_unb.size):
            for j in range(g_unb.size):
                toe[i, j] = g_unb[abs(i - j)]
        if np.min(np.linalg.eigvalsh(toe)) >= 0:
            continue
        ran += 1
        from fmrimod.ar.acvf import PooledAcvf

        g, _ = acvf_from_pooled(PooledAcvf(num=num, pairs=pairs))
        toe2 = np.empty((g.size, g.size))
        for i in range(g.size):
            for j in range(g.size):
                toe2[i, j] = g[abs(i - j)]
        assert np.min(np.linalg.eigvalsh(toe2)) >= -1e-8 * max(1.0, abs(g[0]))
        assert np.all(np.isfinite(g))
    assert ran > 0


def test_valid_segments_break_at_runs_and_censor():
    seg = valid_segments(
        10, runs=np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]), censor=np.array([2, 7])
    )
    np.testing.assert_array_equal(seg.idx, [0, 1, 3, 4, 5, 6, 8, 9])
    np.testing.assert_array_equal(seg.starts0, [0, 2, 4, 6])
    np.testing.assert_array_equal(seg.run_id, [1, 1, 1, 1, 2, 2, 2, 2])


def test_lag_products_confined_to_segment():
    m = np.concatenate([np.zeros(5), np.full(5, 10.0)])[:, None]
    seg = np.concatenate([np.ones(5, dtype=np.intp), np.full(5, 2, dtype=np.intp)])
    pooled = pooled_acvf_segments(m, seg, 2, center_id=seg)
    assert abs(pooled.num[1]) < 1e-12
    assert pooled.pairs[1] == 8


def test_enforce_stationary_ar_margin():
    margin = 1e-6
    cases = [
        np.full(10, 0.99),
        np.full(6, 0.99),
        np.full(3, 0.9),
        np.array([0.5, 0.3]),
        np.array([1.9, -0.99]),
        np.full(4, 0.999),
        np.full(8, 0.95),
    ]
    checked = 0
    for v in cases:
        out = enforce_stationary_ar(v, 0.99)
        if out.size == 0:
            continue
        checked += 1
        assert np.all(np.isfinite(out))
        assert _min_root(out) > 1.0 + margin / 2
    assert checked > 0
    clamp_only = pacf_to_ar(np.clip(ar_to_pacf(np.full(6, 0.99)), -0.99, 0.99))
    assert _min_root(clamp_only) < 1.0 + margin / 2


def test_arma_warns_across_censoring_gaps():
    rng = np.random.RandomState(1)
    resid = np.column_stack([rng.randn(300) for _ in range(10)])
    for t in range(1, 300):
        resid[t] += 0.5 * resid[t - 1]
    cens = np.arange(9, 290, 10)
    with pytest.warns(UserWarning, match="censoring gaps"):
        fit_noise(resid, method="arma", p=1, q=1, pooling="global", censor=cens)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit_noise(resid, method="arma", p=1, q=1, pooling="global")


def test_character_parcel_labels_are_refused():
    a = _ar_sim(200, 0.5, nvox=16, seed=8123)
    chr_lab = np.resize(["A", "B", "C", "D"], 16)
    with pytest.raises(ValueError, match="must be integer, numeric, or factor labels"):
        fit_noise(a, parcels=chr_lab, pooling="parcel", method="ar", p=2)
    with pytest.raises(ValueError, match="whole-number"):
        fit_noise(
            a,
            parcels=np.resize([1.0, 2.5], 16),
            pooling="parcel",
            method="ar",
            p=2,
        )


def test_sigma2_from_gamma_phi_refuses_short_gamma():
    g = np.array([2.0, 1.2, 0.7, 0.4])
    assert sigma2_from_gamma_phi(g, np.array([0.5, 0.2, 0.1])) == pytest.approx(
        2.0 - (0.5 * 1.2 + 0.2 * 0.7 + 0.1 * 0.4)
    )
    assert np.isnan(sigma2_from_gamma_phi(g[:2], np.array([0.5, 0.2, 0.1])))
    assert sigma2_from_gamma_phi(g, np.array([])) == pytest.approx(2.0)
    assert np.isnan(sigma2_from_gamma_phi(g, np.array([0.5, np.nan, 0.1])))
    assert sigma2_from_gamma_phi(
        np.array([2.0, 1.0, 0.5]), np.array([-0.9, -0.9])
    ) == pytest.approx(2.0)
    assert sigma2_from_gamma_phi(
        np.array([1.0, -0.9, -0.9]), np.array([-0.9, -0.9])
    ) == pytest.approx(1e-12)


def test_whiten_apply_does_not_mutate_inputs():
    resid = _ar_sim(200, 0.6, nvox=20, seed=2401)
    parcels = np.tile(np.arange(1, 5), 5)
    rng = np.random.RandomState(0)
    X = np.column_stack(
        [np.ones(200), (np.arange(200) > 100).astype(float), rng.randn(200)]
    )
    x0 = X.copy()
    y0 = resid.copy()
    for pooling in ("global", "parcel"):
        kwargs = {"resid": resid, "method": "ar", "p": 1, "pooling": pooling}
        if pooling == "parcel":
            kwargs["parcels"] = parcels
        plan = fit_noise(**kwargs)
        whiten_apply(plan, X, resid, parcels=parcels if pooling == "parcel" else None)
        np.testing.assert_array_equal(X, x0)
        np.testing.assert_array_equal(resid, y0)


def test_whiten_apply_rejects_malformed_runs():
    resid = _ar_sim(20, 0.5, nvox=3, seed=3201)
    rng = np.random.RandomState(1)
    X = np.column_stack([np.ones(20), rng.randn(20)])
    plan = fit_noise(resid, pooling="global", method="ar", p=1)
    bad_na = np.repeat([1, 2], 10).astype(float)
    bad_na[4] = np.nan
    with pytest.raises(ValueError, match="NA"):
        whiten_apply(plan, X, resid, runs=bad_na)
    with pytest.raises(ValueError, match="length"):
        whiten_apply(plan, X, resid, runs=np.ones(5, dtype=int))
    out = whiten_apply(plan, X, resid, runs=np.repeat(["a", "b"], 10))
    assert not np.any(np.isnan(out.Y))
    assert out.Y.shape == resid.shape


def test_compat_plan_from_phi_without_ma():
    rng = np.random.RandomState(3101)
    X = np.column_stack([np.ones(60), rng.randn(60)])
    Y = _ar_sim(60, 0.5, nvox=4, seed=3101)
    for pooling in ("global", "run"):
        runs = np.repeat([1, 2], 30) if pooling == "run" else None
        plan = plan_from_phi(np.array([0.4, 0.2]), pooling=pooling, runs=runs)
        assert plan.order == (2, 0)
        out = whiten_apply(plan, X, Y, runs=runs)
        assert out.Y.shape == Y.shape
        assert not np.allclose(out.Y, Y)


def test_whitening_reduces_autocorrelation():
    resid = _ar_sim(400, 0.6, nvox=30, seed=707)
    rng = np.random.RandomState(0)
    X = np.column_stack([np.ones(400), rng.randn(400)])
    plan = fit_noise(resid, pooling="global", method="ar", p=1)
    out = whiten_apply(plan, X, resid)
    assert _lag1(resid) > 0.4
    assert abs(_lag1(out.Y)) < 0.08
