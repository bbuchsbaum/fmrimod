"""REG-A: AR + robust composition (R review D1/D2, fb00ab1; test-review-D-robust.R
'robust reestimate_phi recovers phi and matches the plain AR fit').

fmri_lm runs iterative_gls (whitened X_w, whitened residuals) and then
robust_refit, which pairs run_X = X_w with Y = get_run_data() (raw, unwhitened).
R: robust weights are applied to the whitened X *and* whitened Y; with no
outliers the AR+huber fit must stay close to the AR fit.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fmrireg_a_fixture import two_run

import fmrimod as fm
from fmrimod.model.config import AROptions, FmriLmConfig, RobustOptions

warnings.simplefilter("ignore")
ds, Y, ev = two_run(seed=21, phi=0.5, n=150, V=20)
ar = fm.fmri_lm("hrf(condition)", ds, config=FmriLmConfig(ar=AROptions(struct="ar1")))
arh = fm.fmri_lm("hrf(condition)", ds, config=FmriLmConfig(ar=AROptions(struct="ar1"),
                                                         robust=RobustOptions(type="huber")))
ols = fm.fmri_lm("hrf(condition)", ds)
se = ar.se()[:2]
print("event betas: max|AR+huber - AR| / SE(AR) =", float(np.max(np.abs(arh.betas[:2] - ar.betas[:2]) / se)))
print("event betas: max|OLS - AR| / SE(AR)      =", float(np.max(np.abs(ols.betas[:2] - ar.betas[:2]) / se)))
print("intercepts AR:", np.round(ar.betas[2:, :3], 3).tolist())
print("intercepts AR+huber:", np.round(arh.betas[2:, :3], 3).tolist())
re = fm.fmri_lm("hrf(condition)", ds, config=FmriLmConfig(ar=AROptions(struct="ar1"),
                robust=RobustOptions(type="huber", reestimate_phi=True)))
print("reestimate_phi=True changes nothing:", np.array_equal(re.betas, arh.betas))
