"""REG-A: AROptions fields the fmri_lm AR path ignores.

fmri_lm -> ar.integration.iterative_gls -> estimate_ar(order, voxelwise,
censor, noise_pools); q/method/pooling/parcels/global_ar never reach the
whitening.  R 3c2a9c2 (ARMA inference, q>0 routed to fmriAR 'arma') and
168980d (global pooling over runs) give these options real semantics.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fmrireg_a_fixture import two_run

import fmrimod as fm
from fmrimod.model.config import AROptions, FmriLmConfig

warnings.simplefilter("ignore")
ds, Y, ev = two_run(seed=11, phi=0.5, n=120)

def fit(ar):
    return fm.fmri_lm("hrf(condition)", ds, config=FmriLmConfig(ar=ar))

base = fit(AROptions(struct="ar1"))
print("ar1 ar_params:", np.round(np.asarray(base.ar_params).ravel(), 4))
for label, ar in [
    ("global_ar=True", AROptions(struct="ar1", global_ar=True)),
    ("pooling='run'", AROptions(struct="ar1", pooling="run")),
    ("struct='arma', q=1", AROptions(struct="arma", p=1, q=1)),
    ("method='arma', q=1", AROptions(struct="ar1", method="arma", q=1)),
]:
    f = fit(ar)
    print(f"{label:22s} max|dbeta| vs ar1 = {np.max(np.abs(f.betas - base.betas)):.3e}  "
          f"ar_params={np.round(np.asarray(f.ar_params).ravel(), 4)}")

# what a pooled (global) AR(1) whitening would give: mean of per-run phis
phis = np.asarray(base.ar_params).ravel()
print("per-run phi used for whitening (global_ar ignored):", phis)
