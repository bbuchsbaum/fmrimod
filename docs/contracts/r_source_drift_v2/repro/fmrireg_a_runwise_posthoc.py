"""REG-A: contrasts on multi-run runwise fits whose runs are whitened (R final
pass 0c5819f; test-review-F-integration.R 'post-hoc contrasts on multi-run
runwise fits: exact for OLS, refused when runs are whitened').

fmrimod FmriLm.contrast always uses XtXinv = inv(sum_r Xw_r'Xw_r) (strategies.
_pool_run_results) with sigma^2 = sum RSS / sum df and the IV-pooled betas.
R's in-fit runwise contrast pools per-run contrast estimates by inverse
variance; R refuses the 'global (X'X)^-1 x pooled sigma' path for AR/weighted
runs (SE ratio ~0.65 there).
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

def compare(fit, label):
    c = np.array([1.0, -1.0, 0, 0])
    res = fit.contrast(c, name="AvB")
    est_r, w_r = [], []
    for rr, pj in zip(fit.run_results, fit.projections):
        e = c @ rr.betas
        v = (c @ pj.XtXinv @ c) * rr.sigma2
        est_r.append(e); w_r.append(1.0 / v)
    est_r, w_r = np.array(est_r), np.array(w_r)
    est_meta = (est_r * w_r).sum(0) / w_r.sum(0)
    se_meta = np.sqrt(1.0 / w_r.sum(0))
    print(f"{label}: SE ratio fmrimod/meta median={np.median(res.se / se_meta):.3f} "
          f"range=({np.min(res.se / se_meta):.3f},{np.max(res.se / se_meta):.3f}); "
          f"max|est diff|/se_meta={np.max(np.abs(res.estimate - est_meta) / se_meta):.3f}")

# runs with different noise levels and AR so per-run sigmas/phis differ
rng = np.random.default_rng(7)
ds, Y, ev = two_run(seed=7, phi=0.5, n=150, V=30)
Y2 = Y.copy(); Y2[150:] = 100 + (Y[150:] - 100) * 2.5   # run 2 noisier
ds2 = fm.fmri_dataset(Y2, tr=2.0, events=ev, run_length=[150, 150])
compare(fm.fmri_lm("hrf(condition)", ds2), "OLS runwise")
compare(fm.fmri_lm("hrf(condition)", ds2, config=FmriLmConfig(ar=AROptions(struct="ar1"))), "AR1 runwise")
compare(fm.fmri_lm("hrf(condition)", ds), "OLS runwise, equal-variance runs")
compare(fm.fmri_lm("hrf(condition)", ds, config=FmriLmConfig(ar=AROptions(struct="ar1"))), "AR1 runwise, equal-variance runs")
