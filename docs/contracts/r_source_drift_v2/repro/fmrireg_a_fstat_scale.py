"""REG-A: F statistic / tstat() absolute thresholds (R review C7, 0f9138b; B10 004b0f3).

R test-review-C-con-stats.R 'C7: fast t, F and beta statistics are invariant
to the scale of Y' uses s in {1e-8, 1e-4, 1e4}.  fmrimod contrast_f uses
`sigma2 > 1e-15` (fmrimod/glm/contrasts.py:726-731) and FmriLm.tstat uses
`se > 1e-15` (fmrimod/glm/fmri_lm.py:581).
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fmrireg_a_fixture import two_run

import fmrimod as fm
from fmrimod.glm.contrasts import contrast_f, contrast_t

warnings.simplefilter("ignore")
rng = np.random.default_rng(5)
n = 60
X = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
Y0 = (X @ np.array([0.5, 1, -0.5, 0.25]))[:, None] + rng.normal(size=(n, 5))
L = np.array([[0, 1, 0, 0], [0, 0, 1, 0.0]])
for s in (1.0, 1e-4, 1e-8, 1e4):
    Y = Y0 * s
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    df = n - 4
    sig = np.sqrt(((Y - X @ B) ** 2).sum(0) / df)
    XtXinv = np.linalg.inv(X.T @ X)
    F = contrast_f(L, B, XtXinv, sig, df).stat
    t = contrast_t(np.array([0, 1, -1, 0.0]), B, XtXinv, sig, df).stat
    print(f"s={s:g}  F={np.round(F, 4)}  t={np.round(t, 4)}")

# end-to-end: F contrast through fmri_lm on the 2-run fixture
for s in (1.0, 1e-8):
    ds, Y, ev = two_run(scale=s, seed=3)
    fit = fm.fmri_lm("hrf(condition)", ds)
    res = fit.contrast(np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]), name="F")
    print(f"fmri_lm s={s:g}  F[:4]={np.round(res.stat[:4], 4)}  p[:4]={np.round(res.p_value[:4], 4)}")

# suffstats path: sigma floored at sqrt(eps) (fmrimod/glm/matrix.py:176)
from fmrimod.glm.matrix import fit_glm_from_suffstats

for s in (1.0, 1e-8):
    Y = Y0 * s
    fit = fit_glm_from_suffstats(None, X.T @ X, X.T @ Y, (Y * Y).sum(0), n - 4)
    print(f"suffstats s={s:g}  t[1,:3]={np.round(fit.tstat()[1, :3], 4)}")

# AR estimate: estimate_ar_yule_walker returns phi=0 when autocorr[0] < 1e-15
from fmrimod.model.config import AROptions, FmriLmConfig

for s in (1.0, 1e-9):
    ds, Y, ev = two_run(scale=s, seed=11, phi=0.5, n=120)
    f = fm.fmri_lm("hrf(condition)", ds, config=FmriLmConfig(ar=AROptions(struct="ar1")))
    print(f"AR1 s={s:g}: ar_params={np.round(np.ravel(f.ar_params), 4)}  t[0,:3]={np.round(f.tstat()[0,:3], 3)}")
