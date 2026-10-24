"""Transcribed from fmriAR ``tests/testthat/test-estimator-correctness.R``.

Cheap-pass disqualifier: storing ``plan.censor`` while still fragment-centering
the ACVF cannot recover φ under 20–40% scrubbing, honour parcel censoring, or
keep lag-1 of two-frame fragments positive.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.signal import lfilter

from fmrimod.ar.acvf import (
    _bic_n,
    acvf_from_pooled,
    pooled_acvf_segments,
    shrink_to_pd,
    sigma2_from_gamma_phi,
    valid_segments,
)
from fmrimod.ar.estimation import fit_noise
from fmrimod.ar.numhelpers import (
    ar_to_pacf,
    enforce_stationary_ar,
    levinson_durbin,
    pacf_to_ar,
)
from fmrimod.ar.plan import plan_from_phi
from fmrimod.ar.whitening import whiten_apply


def _arma_sim(n: int, nvox: int, phi=(), theta=(), rng=None, burn: int = 200):
    """``arima.sim`` analogue: ARMA(phi, theta) columns, unit innovations."""
    rng = np.random.default_rng() if rng is None else rng
    e = rng.standard_normal((n + burn, nvox))
    b = np.concatenate([[1.0], np.asarray(theta, dtype=np.float64)])
    a = np.concatenate([[1.0], -np.asarray(phi, dtype=np.float64)])
    return lfilter(b, a, e, axis=0)[burn:]


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


def _arma11_rho1(phi: float, theta: float) -> float:
    return (1 + phi * theta) * (phi + theta) / (1 + 2 * phi * theta + theta**2)


def test_arma_is_estimated_from_voxels_not_their_mean():
    """fmriAR 617bd87 (test-audit-regressions.R:207-222), row AR-03.

    Near-white voxels plus a weak shared slow signal: the voxel-mean series
    is dominated by the shared component. Cheap pass disqualified: keeping
    the voxel-mean Hannan-Rissanen fit fails the lag-1 match (phi ~0.89).
    """
    rng = np.random.default_rng(9112)
    n, v = 400, 200
    shared = lfilter([1.0], [1.0, -0.9], rng.standard_normal(n))
    R = rng.standard_normal((n, v)) + 0.3 * shared[:, None]
    plan = fit_noise(R, method="arma", p=1, q=1)
    rho1 = _arma11_rho1(plan.phi[0][0], plan.theta[0][0])
    assert abs(rho1 - _lag1(R)) < 0.05
    W = whiten_apply(plan, np.ones((n, 1)), R).Y
    assert abs(_lag1(W)) < 0.03


def test_arma_respects_censoring_gaps_instead_of_splicing_them():
    """Lags stay inside contiguous segments (R estimator-correctness :678-697).

    This replaces the wave 1 warning that ``method="arma"`` is biased under
    censoring: the pooled estimator no longer is, so fmriAR removed it.
    """
    rng = np.random.default_rng(6861)
    est = []
    for _ in range(20):
        resid = _arma_sim(300, 20, [0.5], [0.4], rng=rng)
        cens = np.sort(rng.choice(300, 75, replace=False))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plan = fit_noise(resid, method="arma", p=1, q=1, censor=cens)
        est.append([plan.phi[0][0], plan.theta[0][0]])
    est = np.array(est)
    assert abs(est[:, 0].mean() - 0.5) < 0.05
    assert abs(est[:, 1].mean() - 0.4) < 0.05


def test_pooled_arma_recovers_parameters_and_voxel_scale_sigma2():
    """R test-audit-regressions.R:224-237, :818-825 and :808-816."""
    rng = np.random.default_rng(9113)
    R = _arma_sim(400, 30, [0.5], [0.4], rng=rng)
    runs = np.repeat([1, 2], 200)
    pg = fit_noise(R, runs=runs, method="arma", p=1, q=1)
    pr = fit_noise(R, runs=runs, method="arma", p=1, q=1, pooling="run")
    assert abs(pg.phi[0][0] - 0.5) < 0.1
    assert abs(pg.theta[0][0] - 0.4) < 0.1
    for i in range(2):
        assert abs(pr.phi[i][0] - pg.phi[0][0]) < 0.1
        assert abs(pr.theta[i][0] - pg.theta[0][0]) < 0.1
        assert abs(pr.sigma2[i] - 1) < 0.15
    # Innovation variance at voxel scale, not the voxel-mean's.
    assert abs(pg.sigma2[0] - 1) < 0.1
    # Refinement iterations change theta and stay accurate.
    p2 = fit_noise(R, method="arma", p=1, q=1, hr_iter=2)
    assert not np.allclose(p2.theta[0], fit_noise(R, method="arma", p=1, q=1).theta[0])
    assert abs(p2.phi[0][0] - 0.5) < 0.08
    assert abs(p2.theta[0][0] - 0.4) < 0.08
    # ARMA(0, 0) is a white plan carrying the voxel variance.
    W = 2.0 * rng.standard_normal((300, 10))
    p0 = fit_noise(W, method="arma", p=0, q=0)
    assert p0.order == (0, 0)
    assert abs(p0.sigma2[0] - 4) < 0.4
    np.testing.assert_array_equal(whiten_apply(p0, np.ones((300, 1)), W).Y, W)


def test_pooled_arma_falls_back_to_white_noise_with_a_warning():
    """R test-audit-regressions.R:742-747."""
    from fmrimod.ar.hr_arma import _hr_arma_pooled

    units = [(np.random.default_rng(1).standard_normal((3, 2)), np.array([0]))]
    with pytest.warns(UserWarning, match="white-noise"):
        fit = _hr_arma_pooled(units, 1, 1)
    assert not fit.ok
    np.testing.assert_array_equal(fit.phi, [0.0])
    np.testing.assert_array_equal(fit.theta, [0.0])


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


def test_explicit_p_above_p_max_is_honoured():
    """fmriAR 617bd87 (test-audit-regressions.R:21-30), row AR-07.

    Cheap pass disqualified: raising the default ``p_max`` would also give
    order 8, so the fit keeps the default ``p_max=6``.
    """
    resid = _ar_sim(400, [0.3, 0.1, 0, 0, 0, 0, 0, 0.1], nvox=10, seed=9101)
    plan = fit_noise(resid, p=8)
    assert plan.order == (8, 0)
    assert _plan_phi(plan).size == 8
    plan_par = fit_noise(resid, p=8, pooling="parcel", parcels=np.tile([1, 2], 5))
    assert plan_par.order[0] == 8
    for bad in (1.5, -1, True, "two"):
        with pytest.raises(ValueError, match="whole number"):
            fit_noise(resid, p=bad)
    assert fit_noise(resid, p=2.0).order == (2, 0)


def test_bic_uses_the_standard_penalty_on_frames_times_effective_voxels():
    """fmriAR 617bd87 + 61b89ff (test-audit-regressions.R:32-43, :827-845).

    BIC = N log sigma2 + k log N with N = frames x V_eff (Kish design
    effect). Row AR-01. Cheap pass disqualified: fixing only 2n -> n passes
    (a) but under-selects in (b), because frames-only N is the 0.4.1 bug;
    counting every voxel (V instead of V_eff) passes (b) but fails (c).
    """
    rng = np.random.default_rng(9102)
    # (a) one voxel, true AR(1): the halved penalty over-selected 18% of the time.
    sel1 = [len(fit_noise(_arma_sim(200, 1, [0.5], rng=rng), p="auto").phi[0])
            for _ in range(300)]
    assert np.mean(np.array(sel1) > 1) < 0.08
    # The parcel series estimator scores BIC the same way.
    sel1p = [
        fit_noise(_arma_sim(200, 1, [0.5], rng=rng), p="auto", pooling="parcel",
                  parcels=np.array([1])).order[0]
        for _ in range(100)
    ]
    assert np.mean(np.array(sel1p) > 1) < 0.08
    # (b) pooled voxels with a weak second lag: frames-only N under-selects.
    sel2 = [len(fit_noise(_arma_sim(300, 30, [0.5, -0.1], rng=rng), p="auto").phi[0])
            for _ in range(20)]
    assert np.mean(np.array(sel2) == 2) >= 0.9
    # (c) identical voxels carry one series' worth of evidence.
    same = np.tile(rng.standard_normal((300, 1)), (1, 40))
    assert _bic_n(same, 300) == pytest.approx(300.0)


def test_bic_n_matches_the_reference_design_effect():
    """``_bic_n`` equals frames x V_eff, with or without the ``vbar`` shortcut."""
    rng = np.random.default_rng(9312)
    M = rng.standard_normal((300, 40)) + 0.7 * rng.standard_normal((300, 1))
    g = np.repeat([1, 2], 150)
    vbar = vmean = 0.0
    for r in (1, 2):
        m = M[g == r] - M[g == r].mean(axis=0)
        vbar += float(np.sum(m * m)) / 40
        vmean += float(np.sum(m.mean(axis=1) ** 2))
    rbar = min(max((40 * vmean / vbar - 1) / 39, 0.0), 1.0)
    v_eff = max(1.0, min(40.0, 40 / (1 + 39 * rbar)))
    assert _bic_n(M, 1000, groups=g) == pytest.approx(1000 * v_eff, rel=1e-12)
    pooled = pooled_acvf_segments(M, g, 0, center_id=g)
    assert _bic_n(M, 1000, groups=g, vbar=pooled.num[0]) == pytest.approx(
        1000 * v_eff, rel=1e-12
    )
    assert _bic_n(M[:, :1], 1000) == 1000


def _r_global_ar(resid, runs, p, p_max=6, censor=()):
    """Transcription of fmriAR 0.4.2 ``.ar_from_pooled_runs`` (617bd87).

    Yule-Walker on the frame-weighted mean of the runs' unrepaired
    autocorrelations, PD-repaired at each candidate order; under
    ``p="auto"`` one BIC search on N = sum of per-run frames x V_eff.
    """
    censor = {int(c) for c in censor}
    graw, lens, n_bics = [], [], []
    for r in np.unique(runs):
        rows = np.where(runs == r)[0]
        keep = np.array([i for i in rows if i not in censor])
        seg = np.cumsum(np.concatenate([[1], (np.diff(keep) > 1).astype(int)]))
        n_eff = keep.size
        p_cap = min(max(p_max, 0 if p == "auto" else p), n_eff - 1)
        pooled = pooled_acvf_segments(resid[keep], seg, p_cap)
        g, _ = acvf_from_pooled(pooled, order=p_cap, tol=-np.inf)
        graw.append(g)
        lens.append(n_eff)
        n_bics.append(_bic_n(resid[keep], n_eff, vbar=pooled.num[0]))
    L = min(g.size for g in graw)
    w = np.asarray(lens, dtype=float) / sum(lens)
    g = float(w @ [g[0] for g in graw]) * (w @ np.vstack([g[:L] / g[0] for g in graw]))

    def fit(pp):
        phi, s2 = levinson_durbin(shrink_to_pd(g[: pp + 1]), pp)
        return enforce_stationary_ar(phi, 0.99), max(s2, 1e-12)

    if p != "auto":
        return fit(min(p, L - 1))[0]
    nb, best_phi = sum(n_bics), np.array([])
    best_bic = nb * np.log(g[0]) + np.log(nb)
    for pp in range(1, min(L - 1, sum(lens) // 5) + 1):
        phi, s2 = fit(pp)
        bic = nb * np.log(s2) + (pp + 1) * np.log(nb)
        if phi.size == pp and bic < best_bic:
            best_bic, best_phi = bic, phi
    return best_phi


def test_global_pooling_solves_yule_walker_on_pooled_autocorrelation():
    """fmriAR 617bd87 (.ar_from_pooled_runs; test-audit-regressions.R:269-293).

    Row AR-12. Cheap pass disqualified: an accuracy-only test passes both the
    old weighted coefficient average and the pooled solve (their fixed-p
    difference is <= 0.002), so this is a formula check.
    """
    rng = np.random.default_rng(9117)
    lens = [120, 200, 80]
    runs = np.repeat([1, 2, 3], lens)
    for i in range(6):
        Y = np.vstack([_arma_sim(L, 10, [0.5, 0.2], rng=rng) for L in lens])
        cens = np.sort(rng.choice(400, 25, replace=False)) if i % 2 else ()
        fitted = fit_noise(Y, runs=runs, p=2, censor=np.asarray(cens, dtype=int))
        np.testing.assert_allclose(
            fitted.phi[0], _r_global_ar(Y, runs, 2, censor=cens), rtol=0, atol=1e-10
        )
    lens = [100, 100, 100, 100]
    runs = np.repeat([1, 2, 3, 4], lens)
    orders = []
    for _ in range(4):
        Y = np.vstack([_arma_sim(L, 20, [0.4, -0.1], rng=rng) for L in lens])
        fitted = fit_noise(Y, runs=runs, p="auto")
        ref = _r_global_ar(Y, runs, "auto")
        assert fitted.order[0] == ref.size
        np.testing.assert_allclose(fitted.phi[0], ref, rtol=0, atol=1e-10)
        orders.append(fitted.order[0])
    assert orders == [2, 2, 2, 2]


def test_global_pooling_is_invariant_to_per_run_scale():
    """Autocorrelations, not autocovariances, are pooled (R :269-277)."""
    R = _arma_sim(300, 10, [0.5, 0.2], rng=np.random.default_rng(9116))
    runs = np.repeat([1, 2], 150)
    R2 = R.copy()
    R2[runs == 2] *= 10
    a = fit_noise(R, runs=runs, p=2).phi[0]
    b = fit_noise(R2, runs=runs, p=2).phi[0]
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)


def test_parcel_pooling_estimates_from_voxels_not_the_parcel_mean():
    """fmriAR 5f5fc32 (test-audit-regressions.R:493-508), row AR-04.

    A parcel-mean series is dominated by what its voxels share, so its phi
    tracked the shared component (0.82-0.91), not the voxel noise the plan
    is applied to.
    """
    rng = np.random.default_rng(9201)
    n = 300
    pf = np.repeat([1, 2, 3, 4], 20)
    R = _arma_sim(n, pf.size, [0.3], rng=rng)
    for k in range(1, 5):
        R[:, pf == k] += 0.5 * lfilter([1.0], [1.0, -0.9], rng.standard_normal(n))[:, None]
    plan = fit_noise(R, pooling="parcel", parcels=pf, p=1)
    est = np.array([plan.phi_by_parcel[str(k)][0] for k in range(1, 5)])
    vox = np.array([_lag1(R[:, pf == k]) for k in range(1, 5)])
    mean_series = np.array([_lag1(R[:, pf == k].mean(1, keepdims=True)) for k in range(1, 5)])
    assert np.max(np.abs(est - vox)) < 0.03
    assert np.all(mean_series - vox > 0.2)


def test_single_all_voxel_parcel_reproduces_global_pooling():
    """R test-audit-regressions.R:510-518. Cheap pass disqualified: any
    rescaling of the parcel-mean autocovariance breaks this identity."""
    R = _arma_sim(250, 12, [0.5, 0.2], rng=np.random.default_rng(9202))
    cens = np.array([29, 30, 139])
    for pp in (2, "auto"):
        g = fit_noise(R, p=pp, censor=cens)
        pc = fit_noise(R, p=pp, censor=cens, pooling="parcel", parcels=np.ones(12, int))
        np.testing.assert_allclose(pc.phi_by_parcel["1"], g.phi[0], rtol=0, atol=1e-10)


def test_segment_lag_sums_match_a_plain_reference():
    """R test-audit-regressions.R:636-655: segments of length 1 and lags
    longer than any segment; centring per ``center_id`` group."""
    rng = np.random.default_rng(9301)
    M = rng.standard_normal((40, 3))
    seg = np.repeat([1, 2, 3, 4], [15, 1, 20, 4])
    for center in (None, np.repeat([1, 2], 20)):
        if center is None:
            Mc = M - M.mean(axis=0)
        else:
            Mc = M.copy()
            for g in (1, 2):
                Mc[center == g] -= Mc[center == g].mean(axis=0)
        for L in (0, 3, 25):
            out = pooled_acvf_segments(M, seg, L, center_id=center)
            ref_num = np.zeros(L + 1)
            ref_pairs = np.zeros(L + 1)
            for h in range(L + 1):
                for t in range(40):
                    s = t - h
                    if s >= 0 and seg[s] == seg[t]:
                        ref_num[h] += Mc[t] @ Mc[s] / 3
                        ref_pairs[h] += 1
            np.testing.assert_allclose(out.num, ref_num, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(out.pairs, ref_pairs)
