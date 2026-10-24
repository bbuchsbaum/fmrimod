"""REG-A: AR refit drops the run preprocessing (R 60f8430 'joint AR/robust
combinations that cannot apply requested volume weights now fail instead of
fitting an unweighted model'; 3c2a9c2 voxelwise contract rejects weights /
soft projection).  fmrimod ar/integration.py:84-96 re-reads raw
get_run_data()/design_matrix_array(), so volume weights and soft-subspace
projection applied in fit_run_ols are silently discarded by the GLS refit."""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fmrireg_a_fixture import two_run

import fmrimod as fm
from fmrimod.model.config import (
    AROptions,
    FmriLmConfig,
    SoftSubspaceOptions,
    VolumeWeightOptions,
)

warnings.simplefilter("ignore")
ds, Y, ev = two_run(seed=3, phi=0.4, n=100)
rng = np.random.default_rng(0)
w = rng.uniform(0.2, 1.0, 200); w[[5, 6, 7, 120]] = 0.0
N = rng.normal(size=(200, 6))
ar = FmriLmConfig(ar=AROptions(struct="ar1"))
base = fm.fmri_lm("hrf(condition)", ds, config=ar)
for label, cfg in [
    ("ar1 + volume weights", FmriLmConfig(ar=AROptions(struct="ar1"),
        volume_weights=VolumeWeightOptions(enabled=True, weights=w))),
    ("ar1 + soft subspace", FmriLmConfig(ar=AROptions(struct="ar1"),
        soft_subspace=SoftSubspaceOptions(enabled=True, nuisance_matrix=N, lam=1.0))),
]:
    f = fm.fmri_lm("hrf(condition)", ds, config=cfg)
    print(f"{label:22s}: max|dbeta vs plain ar1| = {np.max(np.abs(f.betas - base.betas)):.3g}  dfres={f.residual_df}")
iid_w = fm.fmri_lm("hrf(condition)", ds, config=FmriLmConfig(volume_weights=VolumeWeightOptions(enabled=True, weights=w)))
iid = fm.fmri_lm("hrf(condition)", ds)
print("(control) iid + volume weights changes betas by", f"{np.max(np.abs(iid_w.betas - iid.betas)):.3g}")
from fmrimod.ar.whitening import ar_whiten_matrix
from fmrimod.dataset.data_access import get_run_data

f = fm.fmri_lm("hrf(condition)", ds, config=FmriLmConfig(ar=AROptions(struct="ar1"),
    volume_weights=VolumeWeightOptions(enabled=True, weights=w)))
phis = np.asarray(f.ar_params).reshape(2, -1)
for r in range(2):
    Xr = f.model.design_matrix_array(run=r); Yr = get_run_data(ds, r)
    Xw, Yw = ar_whiten_matrix(Xr, Yr, phis[r])
    b_raw = np.linalg.pinv(Xw) @ Yw
    sw = np.sqrt(w[r * 100:(r + 1) * 100])[:, None]
    b_wtd = np.linalg.pinv(Xw * sw) @ (Yw * sw)
    print(f"run {r}: fmrimod run betas == unweighted GLS: {np.allclose(f.run_results[r].betas, b_raw, atol=1e-8)}; "
          f"max|fmrimod - weighted GLS| = {np.max(np.abs(f.run_results[r].betas - b_wtd)):.3g}")
