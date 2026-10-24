"""AR-12 under p='auto': global pooling selects the order once on the pooled
autocorrelation (R 0.4.2 .ar_from_pooled_runs) vs fmrimod's per-run
selection + zero-padded coefficient average (estimation.py:936-969).
Also AR-19: single-series hr_arma burn-in skip (0.4.0)."""

import sys
import warnings

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fmriar_common import r_bic_n, sim_arma  # noqa: E402

from fmrimod.ar import fit_noise, hr_arma  # noqa: E402
from fmrimod.ar.acvf import acvf_from_pooled, pooled_acvf_segments, shrink_to_pd  # noqa: E402
from fmrimod.ar.numhelpers import enforce_stationary_ar, levinson_durbin  # noqa: E402

warnings.simplefilter("ignore")


def r_global_auto(Y, runs, p_max=6):
    graw, lens, nbic, neff = [], [], [], []
    for r in np.unique(runs):
        M = Y[runs == r]
        n = M.shape[0]
        pooled = pooled_acvf_segments(M, np.ones(n), min(p_max, n - 1))
        g, _ = acvf_from_pooled(pooled, order=min(p_max, n - 1), tol=-np.inf)
        graw.append(g)
        lens.append(n)
        neff.append(n)
        nbic.append(r_bic_n(M, n, vbar=pooled.num[0]))
    L = min(g.size for g in graw)
    w = np.array(lens, float) / sum(lens)
    Rho = np.vstack([g[:L] / g[0] for g in graw])
    g = float(np.dot(w, [g[0] for g in graw])) * (w @ Rho)
    nb, n_eff = sum(nbic), sum(neff)
    best_bic, best = nb * np.log(g[0]) + np.log(nb), np.array([])
    for pp in range(1, min(L - 1, n_eff // 5) + 1):
        phi, s2 = levinson_durbin(shrink_to_pd(g[: pp + 1]), pp)
        phi = enforce_stationary_ar(phi, 0.99)
        if phi.size != pp:
            continue
        bic = nb * np.log(max(s2, 1e-12)) + (pp + 1) * np.log(nb)
        if bic < best_bic:
            best_bic, best = bic, phi
    return best


rng = np.random.default_rng(42)
lens = [100, 100, 100, 100]
runs = np.repeat([1, 2, 3, 4], lens)
fm, rr = [], []
for _ in range(30):
    Y = np.vstack([sim_arma(L, 20, [0.4, -0.1], rng=rng) for L in lens])
    fm.append(len(fit_noise(Y, runs=runs, p="auto").phi[0]))
    rr.append(len(r_global_auto(Y, runs)))
print("4 runs x 100, 20 vox, AR(2)(0.4,-0.1) p='auto': fmrimod order counts",
      np.bincount(fm, minlength=4)[:4], " R0.4.2 order counts", np.bincount(rr, minlength=4)[:4])

# AR-19: single-series HR, n = 150, ARMA(1,1) (0.5, 0.4) -- R test :239-252 (mean bias < 0.04)
est = []
for _ in range(200):
    y = sim_arma(150, 1, [0.5], [0.4], rng=rng)[:, 0]
    f = hr_arma(y, 1, 1)
    est.append([f["phi"][0], f["theta"][0]])
est = np.array(est)
print("hr_arma n=150 ARMA(1,1)(0.5,0.4): fmrimod mean (phi, theta) =", np.round(est.mean(0), 3),
      " |bias| =", np.round(np.abs(est.mean(0) - [0.5, 0.4]), 3), "(R 0.4.2 test: < 0.04)")
