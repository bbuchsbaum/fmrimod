"""AR-04: parcel pooling from voxel-pooled autocovariance (0.4.1, 5f5fc32).

fmrimod estimation.py:578 estimates each parcel from parcel_means().
R 0.4.2 .ms_estimate_scale(resid_valid, labels, ...) pools voxel ACVFs.
Fixture = test-audit-regressions.R:493-508 and :510-518.
"""

import sys

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fmriar_common import sim_arma  # noqa: E402

from fmrimod.ar import fit_noise  # noqa: E402
from fmrimod.ar.acvf import acvf_from_pooled, pooled_acvf_segments  # noqa: E402

rng = np.random.default_rng(9201)
n = 300
pf = np.repeat([1, 2, 3, 4], 20)
R = sim_arma(n, pf.size, [0.3], rng=rng)
for k in range(1, 5):
    R[:, pf == k] += 0.5 * lfilter([1.0], [1.0, -0.9], rng.standard_normal(n))[:, None]


def lag1(M):
    M = M - M.mean(0)
    return (M[1:] * M[:-1]).sum(0) / (M * M).sum(0)


plan = fit_noise(R, pooling="parcel", parcels=pf, p=1)
est = np.array([plan.phi_by_parcel[str(k)][0] for k in range(1, 5)])
vox = np.array([lag1(R[:, pf == k]).mean() for k in range(1, 5)])
mean_series = np.array([lag1(R[:, pf == k].mean(1, keepdims=True))[0] for k in range(1, 5)])
# R 0.4.2 with p = 1 fixed: Yule-Walker on the voxel-pooled ACVF of each parcel
r_est = []
for k in range(1, 5):
    pooled = pooled_acvf_segments(R[:, pf == k], np.ones(n, int), 1, center_id=np.ones(n, int))
    g, _ = acvf_from_pooled(pooled, order=1)
    r_est.append(g[1] / g[0])
r_est = np.array(r_est)
print("voxel lag-1     :", np.round(vox, 3))
print("parcel-mean lag1:", np.round(mean_series, 3))
print("fmrimod phi     :", np.round(est, 3), " max|fmrimod - vox| =", round(float(np.max(np.abs(est - vox))), 3))
print("R0.4.2 phi      :", np.round(r_est, 3), " max|R - vox| =", round(float(np.max(np.abs(r_est - vox))), 3))

# Single parcel holding every voxel must reproduce global pooling
R2 = sim_arma(250, 12, [0.5, 0.2], rng=np.random.default_rng(9202))
cens = np.array([29, 30, 139])
g = fit_noise(R2, p=2, censor=cens).phi[0]
pc = fit_noise(R2, p=2, censor=cens, pooling="parcel", parcels=np.ones(12, int)).phi_by_parcel["1"]
print("single-parcel vs global (p=2): global", np.round(g, 4), " parcel", np.round(pc, 4),
      " max diff", float(np.max(np.abs(g - pc))))

# Multiscale unit invariance (0.4.0 ms_dispersion fix), test-audit-regressions.R:296-309
rng = np.random.default_rng(9118)
R3 = sim_arma(200, 64, [0.4], rng=rng) * rng.uniform(0.5, 2, 64)[None, :]
ps = {"coarse": np.repeat([1, 2], 32), "medium": np.repeat([1, 2, 3, 4], 16),
      "fine": np.repeat(np.arange(1, 9), 8)}
for mode in ("pacf_weighted", "acvf_pooled"):
    a = fit_noise(R3, pooling="parcel", parcels=ps["fine"], parcel_sets=ps, p=1, multiscale=mode)
    b = fit_noise(R3 * 100, pooling="parcel", parcels=ps["fine"], parcel_sets=ps, p=1, multiscale=mode)
    da = max(abs(a.phi_by_parcel[k][0] - b.phi_by_parcel[k][0]) for k in a.phi_by_parcel)
    print(f"multiscale {mode}: max|phi(R) - phi(100 R)| = {da:.3g}")

# acvf_pooled with p='auto' on white noise must give white filters (R :799-807)
R4 = np.random.default_rng(9310).standard_normal((300, 32))
ps4 = {"coarse": np.repeat([1, 2], 16), "medium": np.repeat([1, 2, 3, 4], 8),
       "fine": np.repeat(np.arange(1, 9), 4)}
pl4 = fit_noise(R4, pooling="parcel", parcels=ps4["fine"], parcel_sets=ps4, p="auto",
                multiscale="acvf_pooled")
print("acvf_pooled p=auto white noise: fmrimod order =", pl4.order, "(R 0.4.2 expects p = 0)")
