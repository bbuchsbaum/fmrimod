"""AR-01: BIC penalty (0.4.0) + design-effect sample size (0.4.2).

fmrimod: estimation.py:895-906 scores 2*n*log(s2) + k*log(n) on frames.
R 0.4.2: N*log(s2) + k*log(N), N = frames * V_eff (Kish design effect).
"""

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fmriar_common import r_select_ar_run, sim_arma  # noqa: E402

from fmrimod.ar import fit_noise  # noqa: E402

rng = np.random.default_rng(9102)

# (a) single voxel AR(1), n = 200 (R test-audit-regressions.R:32-43)
fm, rr = [], []
for _ in range(300):
    y = sim_arma(200, 1, [0.5], rng=rng)
    fm.append(len(fit_noise(y, p="auto").phi[0]))
    rr.append(len(r_select_ar_run(y)))
fm, rr = np.array(fm), np.array(rr)
print(f"(a) 1 voxel AR(1) n=200: P(p>1) fmrimod={np.mean(fm > 1):.3f}  R0.4.2={np.mean(rr > 1):.3f}")

# (b) pooled voxels with a weak second lag: 30 voxels x 300 frames, AR(2) (0.5, -0.1)
fm, rr = [], []
for _ in range(40):
    Y = sim_arma(300, 30, [0.5, -0.1], rng=rng)
    fm.append(len(fit_noise(Y, p="auto").phi[0]))
    rr.append(len(r_select_ar_run(Y)))
fm, rr = np.array(fm), np.array(rr)
print(f"(b) 30 vox AR(2)(0.5,-0.1) n=300: P(p==2) fmrimod={np.mean(fm == 2):.3f}  R0.4.2={np.mean(rr == 2):.3f}")
print("    fmrimod orders:", np.bincount(fm, minlength=4)[:4], " R orders:", np.bincount(rr, minlength=4)[:4])

# (c) R's 0.4.2 fixture: 60 voxels x 240 frames, 2 runs, pooling='run', AR(2) (0.5, -0.2)
fm, rr = [], []
runs = np.repeat([1, 2], 120)
for _ in range(15):
    Y = sim_arma(240, 60, [0.5, -0.2], rng=rng)
    pl = fit_noise(Y, runs=runs, pooling="run", p="auto")
    fm.append(pl.order[0])
    rr.append(max(len(r_select_ar_run(Y[runs == r])) for r in (1, 2)))
print(f"(c) R fixture 60x240 run-pooled AR(2)(0.5,-0.2): fmrimod orders={fm}  R={rr}")
