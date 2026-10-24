"""AR-03: pooled, segment-aware Hannan-Rissanen ARMA (0.4.0, 617bd87).

fmrimod estimation.py:826-832 runs hr_arma on the voxel-MEAN series.
R 0.4.2 .hr_arma_pooled (R/hr_pooled.R) pools the HR normal equations over
voxels. Fixture = test-audit-regressions.R:207-222.
"""

import sys

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fmriar_common import sim_arma  # noqa: E402

from fmrimod.ar import fit_noise  # noqa: E402
from fmrimod.ar.acvf import acvf_from_pooled, pooled_acvf_segments  # noqa: E402
from fmrimod.ar.numhelpers import enforce_stationary_ar, levinson_durbin  # noqa: E402


def r_enforce_invertible_ma(theta, tol=1e-8, bound=0.99):
    theta = np.asarray(theta, float)
    q = theta.size
    if q == 0:
        return theta
    if not np.all(np.isfinite(theta)):
        return np.zeros(q)
    r = np.roots(np.concatenate([[1.0], theta])[::-1])
    changed = False
    for i in range(r.size):
        if abs(r[i]) <= 1 + tol:
            r[i] = 1 / np.conj(r[i])
            changed = True
        if abs(r[i]) < 1 / bound:
            r[i] = r[i] * ((1 / bound) / abs(r[i]))
            changed = True
    if not changed:
        return theta
    c = np.array([1.0 + 0j])
    for z in r:
        c = np.convolve(c, [1.0, -1.0 / z])
    return np.real(c[1:])


def r_hr_arma_pooled(mat, p, q):
    """Single unit, single segment, iter = 0 transcription of .hr_arma_pooled."""
    mat = mat - mat.mean(axis=0)
    n, V = mat.shape
    k = p + q
    p_big = min(max(8, k + 5, int(np.ceil(10 * np.log10(n)))), 40)
    p_big = min(p_big, n - 1)
    while p_big > max(1, k) and max(0, n - max(p, p_big + q)) < 10 * (k + 1):
        p_big -= 1
    pooled = pooled_acvf_segments(mat, np.ones(n, int), p_big)
    g, _ = acvf_from_pooled(pooled, order=p_big)
    pb = min(p_big, g.size - 1)
    phi_big, _ = levinson_durbin(g[: pb + 1], pb)
    E = lfilter(np.concatenate([[1.0], -phi_big]), [1.0], mat, axis=0)  # zero pre-sample
    start = max(p, pb + q)
    G = np.zeros((k, k))
    b = np.zeros(k)
    for t in range(start, n):
        z = np.vstack([mat[t - 1 - np.arange(p)], E[t - 1 - np.arange(q)]])  # k x V
        G += z @ z.T
        b += z @ mat[t]
    coef = np.linalg.solve(G, b)
    return enforce_stationary_ar(coef[:p], 0.99), r_enforce_invertible_ma(coef[p:])


def rho1_arma11(phi, theta):
    return (1 + phi * theta) * (phi + theta) / (1 + 2 * phi * theta + theta ** 2)


rng = np.random.default_rng(9112)
n, v = 400, 200
e = rng.standard_normal(n)
g = lfilter([1.0], [1.0, -0.9], e)
R = rng.standard_normal((n, v)) + 0.3 * g[:, None]
Rc = R - R.mean(0)
lag1 = np.mean((Rc[1:] * Rc[:-1]).sum(0) / (Rc * Rc).sum(0))

pl = fit_noise(R, method="arma", p=1, q=1)
ph, th = float(pl.phi[0][0]), float(pl.theta[0][0])
phr, thr = r_hr_arma_pooled(R, 1, 1)
print(f"voxel mean lag-1 acf = {lag1:.3f}")
print(f"fmrimod (voxel-mean HR): phi={ph:.3f} theta={th:.3f} implied rho1={rho1_arma11(ph, th):.3f}  sigma2={pl.sigma2[0]}")
print(f"R 0.4.2 (pooled HR):     phi={phr[0]:.3f} theta={thr[0]:.3f} implied rho1={rho1_arma11(phr[0], thr[0]):.3f}")

# Parameter recovery on clean ARMA(1,1), 30 voxels, n = 400 (test-audit-regressions.R:224)
est_f, est_r = [], []
for _ in range(10):
    Y = sim_arma(400, 30, [0.5], [0.4], rng=rng)
    pf = fit_noise(Y, method="arma", p=1, q=1)
    est_f.append([pf.phi[0][0], pf.theta[0][0]])
    a, b = r_hr_arma_pooled(Y, 1, 1)
    est_r.append([a[0], b[0]])
est_f, est_r = np.array(est_f), np.array(est_r)
rmse = lambda E: np.sqrt(np.mean((E - [0.5, 0.4]) ** 2, axis=0))  # noqa: E731
print(f"ARMA(1,1) 30x400 RMSE (phi, theta): fmrimod={np.round(rmse(est_f), 3)}  R pooled={np.round(rmse(est_r), 3)}")
