"""AR-12 exact-formula gap, ARMA p='auto' semantics, AR-19 HR burn-in."""

import sys
import warnings

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fmriar_common import sim_arma  # noqa: E402

from fmrimod.ar import fit_noise, hr_arma  # noqa: E402
from fmrimod.ar.acvf import acvf_from_pooled, pooled_acvf_segments, shrink_to_pd  # noqa: E402
from fmrimod.ar.hr_arma import _arma_innovations, _lag_matrix  # noqa: E402
from fmrimod.ar.numhelpers import (  # noqa: E402
    enforce_invertible_ma,
    enforce_stationary_ar,
    levinson_durbin,
)

warnings.simplefilter("ignore")
rng = np.random.default_rng(9117)

# AR-12 fixed p = 2, runs of 120/200/80
lens = [120, 200, 80]
runs = np.repeat([1, 2, 3], lens)
diffs = []
for _ in range(10):
    Y = np.vstack([sim_arma(L, 10, [0.5, 0.2], rng=rng) for L in lens])
    fphi = fit_noise(Y, runs=runs, p=2).phi[0]
    rhos, g0 = [], []
    for r in (1, 2, 3):
        pooled = pooled_acvf_segments(Y[runs == r], np.ones(int(np.sum(runs == r))), 2)
        g, _ = acvf_from_pooled(pooled, order=2, tol=-np.inf)
        rhos.append(g / g[0])
        g0.append(g[0])
    w = np.array(lens, float) / sum(lens)
    gp = float(w @ g0) * (w @ np.vstack(rhos))
    rphi = enforce_stationary_ar(levinson_durbin(shrink_to_pd(gp), 2)[0], 0.99)
    diffs.append(np.max(np.abs(fphi - rphi)))
print(f"AR-12 p=2 global: max|fmrimod - R .ar_from_pooled_runs| over 10 fixtures = {max(diffs):.4f} (median {np.median(diffs):.4f})")

# ARMA p='auto': R 0.4.0 searches 0..p_max by BIC; fmrimod fixes p = min(2, p_max)
W = rng.standard_normal((240, 30))
pl = fit_noise(W, method="arma", p="auto", q=1)
print("ARMA p='auto', q=1 on white noise: fmrimod order =", pl.order, "(R 0.4.2: BIC over p in 0..6, q fixed 1)")

# AR-19: R hr_arma_R (0.4.0) with burn-in skip vs fmrimod hr_arma, same p_big & step-1
def r_hr_arma_R(y, p, q, p_big):
    y = y - y.mean()
    n = y.size
    g = np.array([np.sum(y[: n - k] * y[k:]) / n for k in range(p_big + 1)])
    phi_big, _ = levinson_durbin(g, p_big)
    ehat = _arma_innovations(y, phi_big, np.array([]))
    m = max(p, q)
    burn = max(m, p_big + q) if q > 0 else m
    if n - burn < 10 * (p + q + 1):
        burn = m
    idx = slice(burn, n)
    Z = np.hstack([_lag_matrix(y, p)[idx], _lag_matrix(ehat, q)[idx]])
    coef = np.linalg.lstsq(Z, y[idx], rcond=None)[0]
    return enforce_stationary_ar(coef[:p], 0.99), enforce_invertible_ma(coef[p:])


y = sim_arma(300, 1, [0.5], [0.4], rng=rng)[:, 0]
f = hr_arma(y, 1, 1)
rp, rt = r_hr_arma_R(y, 1, 1, f["p_big"])
print(f"AR-19 n=300: fmrimod (phi, theta)=({f['phi'][0]:.4f}, {f['theta'][0]:.4f})  "
      f"R0.4.2 hr_arma_R=({rp[0]:.4f}, {rt[0]:.4f})  p_big={f['p_big']}")
