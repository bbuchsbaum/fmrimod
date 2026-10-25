"""Smaller fmriAR 0.4.x divergences, each against the R 0.4.2 test expectation."""

import sys
import warnings

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fmriar_common import sim_arma  # noqa: E402

from fmrimod.ar import (  # noqa: E402
    acorr_diagnostics,
    afni_restricted_plan,
    enforce_invertible_ma,
    fit_noise,
    sandwich_from_whitened_resid,
    whiten,
    whiten_apply,
    whiten_with_phi,
)

warnings.simplefilter("ignore")

print("== AR-06 whiten_apply logical censor mask (R :45-61)")
rng = np.random.default_rng(9103)
R = sim_arma(200, 4, [0.5], rng=rng)
X = np.column_stack([np.ones(200), rng.standard_normal(200)])
mask = np.zeros(200, bool)
mask[[49, 119, 120]] = True
plan = fit_noise(R, p=1, censor=mask)
a = whiten_apply(plan, X, R, censor=mask).Y
b = whiten_apply(plan, X, R, censor=np.where(mask)[0]).Y
print("  mask vs indices max|diff| =", float(np.max(np.abs(a - b))),
      " rows differing:", np.where(np.any(np.abs(a - b) > 1e-12, axis=1))[0].tolist())
w1 = whiten(X, R, censor=mask, p=1).Y
w2 = whiten(X, R, censor=np.where(mask)[0], p=1).Y
print("  whiten(mask) vs whiten(indices) max|diff| =", float(np.max(np.abs(w1 - w2))))

print("== AR-07 explicit p > p_max (R :21-30)")
R8 = sim_arma(400, 10, [0.3, 0.1, 0, 0, 0, 0, 0, 0.1], rng=np.random.default_rng(9101))
print("  fit_noise(p=8).order =", fit_noise(R8, p=8).order, "(R 0.4.2: p = 8)")
print("  parcel fit_noise(p=8).order =",
      fit_noise(R8, p=8, pooling="parcel", parcels=np.tile([1, 2], 5)).order)
try:
    fit_noise(R8, p=1.5)
    print("  fit_noise(p=1.5) accepted silently (R: error 'whole number')")
except Exception as e:  # noqa: BLE001
    print("  fit_noise(p=1.5) ->", type(e).__name__, e)

print("== AR-09 acorr_diagnostics aggregate (R :337-355)")
rng = np.random.default_rng(9121)
n, v = 300, 50
g = lfilter([1.0], [1.0, -0.9], rng.standard_normal(n))
R9 = rng.standard_normal((n, v)) + 0.5 * g[:, None]
per_vox = acorr_diagnostics(R9, max_lag=3, aggregate="none")["acf"]
print("  mean per-voxel lag1 =", round(float(per_vox[0].mean()), 3),
      " fmrimod aggregate='mean' lag1 =", round(float(acorr_diagnostics(R9, max_lag=3)["acf"][0]), 3))

print("== AR-10 enforce_invertible_ma unit root (R :102-116)")
for th in ([-1.0], [1.0], [0.995], [0.0, -1.0]):
    out = enforce_invertible_ma(np.array(th))
    m = np.min(np.abs(np.roots(np.concatenate([[1.0], out])[::-1])))
    print(f"  theta={th} -> {np.round(out, 4)}  min|root|={m:.4f}  (R requires >= {1/0.99:.4f})")

print("== AR-12 global AR(p) pooling via pooled autocorrelation (R :279-293)")
rng = np.random.default_rng(9117)
lens = [120, 200, 80]
runs = np.repeat([1, 2, 3], lens)
err_g, err_r = [], []
for _ in range(40):
    Rg = np.vstack([sim_arma(L, 10, [0.5, 0.2], rng=rng) for L in lens])
    gphi = fit_noise(Rg, runs=runs, p=2).phi[0]
    # R 0.4.2 .ar_from_pooled_runs (p fixed): YW on frame-weighted mean autocorrelation
    from fmrimod.ar.acvf import acvf_from_pooled, pooled_acvf_segments, shrink_to_pd
    from fmrimod.ar.numhelpers import enforce_stationary_ar, levinson_durbin
    rhos, g0s = [], []
    for r in (1, 2, 3):
        pooled = pooled_acvf_segments(Rg[runs == r], np.ones(int(np.sum(runs == r))), 2)
        graw, _ = acvf_from_pooled(pooled, order=2, tol=-np.inf)
        rhos.append(graw / graw[0])
        g0s.append(graw[0])
    w = np.array(lens, float) / sum(lens)
    gp = float(np.dot(w, g0s)) * (w @ np.vstack(rhos))
    rphi = enforce_stationary_ar(levinson_durbin(shrink_to_pd(gp), 2)[0], 0.99)
    err_g.append(np.sum((gphi - [0.5, 0.2]) ** 2))
    err_r.append(np.sum((rphi - [0.5, 0.2]) ** 2))
print(f"  mean sq err: fmrimod coefficient-average={np.mean(err_g):.5f}  R pooled-acf={np.mean(err_r):.5f}")

print("== AR-13 sandwich rank-deficient design (R :357-367)")
rng = np.random.default_rng(9122)
x = rng.standard_normal(100)
Xr = np.column_stack([np.ones(100), x, 2 * x])
y = 1 + x + rng.standard_normal(100)
try:
    out = sandwich_from_whitened_resid(Xr, y[:, None])
    print("  fmrimod se =", np.round(out["se"][:, 0], 4), "df =", out["df"], "(R: se[3] = NA, se[1:2] = lm SEs)")
except Exception as e:  # noqa: BLE001
    print("  fmrimod raised", type(e).__name__, e)

print("== AR-14 AFNI MA(1) from voxels, not the mean (R :255-266)")
rng = np.random.default_rng(9115)
n, v = 300, 100
g = lfilter([1.0], [1.0, -0.95], rng.standard_normal(n))
Ra = rng.standard_normal((n, v)) + 0.05 * g[:, None]
pa = afni_restricted_plan(Ra, p=3, roots={"a": 0, "r1": 0, "t1": 0})
print("  fmrimod theta =", np.round(pa.theta[0], 3), "(R 0.4.2 expects |theta| < 0.06)")
pv = afni_restricted_plan(Ra, p=3, roots={"a": 0.5, "r1": 0.6, "t1": 1, "vrt": 0.7}, estimate_ma1=False)
print("  vrt=0.7, estimate_ma1=False: fmrimod theta =", pv.theta[0], "(R: ARMA(3,3) theta, rho_k = 0.7 rho_AR(k))")
try:
    afni_restricted_plan(Ra, roots={"a": -0.2, "r1": 0.3, "t1": 1})
    print("  negative a accepted (clamped) (R: error '>= 0')")
except Exception as e:  # noqa: BLE001
    print("  negative a ->", e)

print("== AR-15 whiten_apply plan-censor fallback / run-count checks (R :63-100)")
rng = np.random.default_rng(9104)
Rc = sim_arma(200, 4, [0.5], rng=rng)
Xc = np.column_stack([np.ones(200), rng.standard_normal(200)])
pc = fit_noise(Rc, p=1, censor=np.array([39, 40, 149]))
short = whiten_apply(pc, Xc[:150], Rc[:150]).Y
short_nocens = whiten_apply(pc, Xc[:150], Rc[:150], censor=np.array([], dtype=int)).Y
print("  plan.censor applied to 150-row data: rows changed =",
      np.where(np.any(np.abs(short - short_nocens) > 1e-12, axis=1))[0].tolist(), "(R: none; censor not inherited)")
R3 = sim_arma(300, 3, [0.5], rng=rng)
pr = fit_noise(R3, runs=np.repeat([1, 2, 3], 100), pooling="run", p=1)
try:
    whiten_apply(pr, np.ones((200, 1)), R3[:200], runs=np.repeat([1, 2], 100))
    print("  3-run per-run plan on 2-run data: accepted silently (R: error \"3 'phi' sets\")")
except Exception as e:  # noqa: BLE001
    print("  ->", e)

print("== AR-16 whiten_with_phi default exact_first (R :719-726)")
Yw = sim_arma(40, 2, [0.6], rng=np.random.default_rng(9305))
o = whiten_with_phi(np.ones((40, 1)), Yw, phi=np.array([0.6]))
print("  default Y[0] / Y_raw[0] =", np.round(o.Y[0] / Yw[0], 4), "(R 0.4.2 default: sqrt(1-0.36) = 0.8)")

print("== AR-17 design-residual orthogonality check (0.3.3 ca27f27; test-acvf-bias.R)")
rng = np.random.default_rng(707)
n = 120
t = np.linspace(-1, 1, n)
Xd = np.column_stack([np.ones(n), t, t ** 2, t ** 3, rng.standard_normal((n, 4))])
Yd = rng.standard_normal((n, 5))
try:
    fit_noise(Yd, method="ar", p=1, design=Xd)
    print("  raw (unprojected) series + design accepted (R: fmriAR_design_residual_mismatch)")
except Exception as e:  # noqa: BLE001
    print("  ->", e)

print("== AR-18 censor validation (R :675-683)")
R50 = sim_arma(50, 2, [0.3], rng=rng)
for c in (np.array([2.5]), np.array([3.0, np.nan]), np.array([True, False])):
    try:
        fit_noise(R50, p=1, censor=c)
        print("  censor", c, "accepted silently")
    except Exception as e:  # noqa: BLE001
        print("  censor", c, "->", type(e).__name__, e)
