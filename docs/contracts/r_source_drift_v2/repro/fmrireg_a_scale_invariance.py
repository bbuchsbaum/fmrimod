"""REG-A: scale invariance of runwise pooled betas / t (R review C7 / B10).

R (0f9138b, 004b0f3) made every t / pooling path scale-free. fmrimod's
_pool_run_results drops runs whose se^2 <= machine eps (absolute), so data in
small units falls back to an unweighted mean of run betas.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fmrireg_a_fixture import two_run

import fmrimod as fm

warnings.simplefilter("ignore")
for scale in (1.0, 1e-4, 1e-8, 1e-10, 1e-12):
    ds, Y, ev = two_run(scale=scale, seed=3)
    fit = fm.fmri_lm("hrf(condition)", ds)
    b = fit.betas[:2] / scale
    t = fit.tstat()[:2]
    print(f"scale={scale:g}  beta/scale[0,:3]={np.round(b[0,:3], 6)}  t[0,:3]={np.round(t[0,:3], 6)}")
    if scale == 1.0:
        b_ref, t_ref = b.copy(), t.copy()
    else:
        print("   max|dbeta|/scale:", float(np.max(np.abs(b - b_ref))), " max|dt|:", float(np.max(np.abs(t - t_ref))))
    rr = fit.run_results
    pj = fit.projections
    se2 = [np.diag(p.XtXinv)[0] * r.sigma2[:3] for r, p in zip(rr, pj)]
    print("   run se^2 for beta0 (first 3 vox):", [np.array2string(s, precision=3) for s in se2])
