"""REG-A: voxelwise AR whitening (R 3c2a9c2 + 6565fb6 'fit voxelwise AR through
one fit_voxel closure'; test-fmri-lm-voxelwise-noise-contract.R).

R: each voxel's X and Y are whitened with that voxel's phi (per-voxel GLS).
fmrimod ar/whitening.py:125-143 whitens X with mean(phi) and Y with phi_v,
so the solve is not GLS for any voxel whose phi differs from the mean.
"""
import warnings

import numpy as np
import pandas as pd

import fmrimod as fm
from fmrimod.ar.whitening import ar_whiten
from fmrimod.design import RealizedDesign
from fmrimod.model.config import AROptions, FmriLmConfig

warnings.simplefilter("ignore")
rng = np.random.default_rng(20260822)
n = 180
phis = [0.05, 0.25, 0.45, 0.65]
X = np.column_stack([np.ones(n), np.linspace(-1, 1, n), np.sin(np.arange(n) / 4.0)])
E = np.zeros((n, 4))
for v, p in enumerate(phis):
    z = rng.normal(size=n)
    for t in range(n):
        E[t, v] = z[t] + (p * E[t - 1, v] if t else 0.0)
Y = X @ np.array([[10.0] * 4, [2.0] * 4, [1.0] * 4]) + E
ev = pd.DataFrame({"onset": [0.0], "condition": ["A"], "duration": [0.0], "run": [1]})
ds = fm.fmri_dataset(Y, tr=1.0, events=ev, run_length=[n])
fit = fm.fmri_lm(RealizedDesign.from_array(X, ["int", "lin", "s"]), ds,
                 config=FmriLmConfig(ar=AROptions(struct="ar1", voxelwise=True)))
phi_hat = np.asarray(fit.ar_params).reshape(1, -1)
print("voxelwise phi_hat:", np.round(phi_hat.ravel(), 3))
for v in range(4):
    pv = phi_hat[:, v]
    Xw = ar_whiten(X, pv); yw = ar_whiten(Y[:, [v]], pv)
    b_gls = np.linalg.lstsq(Xw, yw, rcond=None)[0].ravel()
    print(f"voxel {v}: fmrimod beta={np.round(fit.betas[:, v], 4)}  per-voxel GLS={np.round(b_gls, 4)}")
