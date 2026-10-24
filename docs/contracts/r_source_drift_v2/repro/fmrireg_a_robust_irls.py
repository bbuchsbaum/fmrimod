"""REG-A: robust IRLS vs R review D2 (fb00ab1; test-review-D-robust.R).

R contract: each IRLS step weights the ORIGINAL X and Y by sqrt(w) for one
row-weight vector w; the returned weights are the ones used for the final
betas; residual df = sum(w > 0) - rank (lm.wfit).
fmrimod robust/irls.py:62-110 reuses the previous iteration's *weighted*
X (run_X = new_run_X) but the raw Y, so iteration k fits X*w1*...*wk against
Y*wk; df = n - rank; returned weights are the (n, V) voxel weights, not the
row weights actually applied.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import fmrimod as fm
from fmrimod.design import RealizedDesign
from fmrimod.model.config import FmriLmConfig, RobustOptions
from fmrimod.robust.estimators import bisquare_weights, mad_scale

warnings.simplefilter("ignore")
rng = np.random.default_rng(4)
n, V = 120, 5
X = np.column_stack([np.sin(np.arange(n) / 5.0), np.cos(np.arange(n) / 7.0), np.ones(n)])
Y = X @ np.array([[3.0] * V, [-2.0] * V, [50.0] * V]) + rng.normal(size=(n, V))
spikes = [6, 39, 40, 94]
Y[spikes] += 25.0
ev = pd.DataFrame({"onset": [0.0], "condition": ["A"], "duration": [0.0], "run": [1]})
ds = fm.fmri_dataset(Y, tr=2.0, events=ev, run_length=[n])
cfg = FmriLmConfig(robust=RobustOptions(type="bisquare", max_iter=2))
fit = fm.fmri_lm(RealizedDesign.from_array(X, ["a", "b", "int"]), ds, config=cfg)

# Replay fmrimod's two iterations to recover the row weights it applied.
B0 = np.linalg.lstsq(X, Y, rcond=None)[0]
r0 = Y - X @ B0
w1 = np.mean(np.sqrt(bisquare_weights(r0, mad_scale(r0, axis=0), c=4.685)), axis=1)
Xw1, Yw1 = X * w1[:, None], Y * w1[:, None]
B1 = np.linalg.lstsq(Xw1, Yw1, rcond=None)[0]
r1 = Yw1 - Xw1 @ B1                      # weighted residuals (as fmrimod)
w2 = np.mean(np.sqrt(bisquare_weights(r1, mad_scale(r1, axis=0), c=4.685)), axis=1)
B_bug = np.linalg.lstsq(Xw1 * w2[:, None], Y * w2[:, None], rcond=None)[0]
B_wls2 = np.linalg.lstsq(X * w2[:, None], Y * w2[:, None], rcond=None)[0]
print("fmrimod betas == replay with compounded X weights:", np.allclose(fit.betas, B_bug, atol=1e-8))
print("fmrimod betas vs consistent WLS(w2): max|d| =", float(np.max(np.abs(fit.betas - B_wls2))))
print("fmrimod slope estimates (true 3, -2):", np.round(fit.betas[:2, 0], 4),
      " consistent WLS:", np.round(B_wls2[:2, 0], 4))
print("rows with zero applied weight:", int(np.sum(w2 == 0)), " fmrimod dfres:", fit.residual_df,
      " lm.wfit df would be:", int(np.sum(w2 > 0)) - 3)
print("returned weights shape:", np.shape(fit.robust_weights), "(R: length-n row weights used for final betas)")

# R transcription (fmri_robust_fitting.R robust_iterative_fitter, scale_scope="run")
def r_irls(X, Y, c=4.685, max_iter=2, tol=1e-4):
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    for it in range(max_iter):
        Bp = B
        res = np.abs(Y - X @ B)
        sig = 1.4826 * np.median(np.median(res, axis=1))
        u = np.median(res / sig, axis=1)
        w = np.where(np.abs(u) <= c, (1 - (u / c) ** 2) ** 2, 0.0)
        sw = np.sqrt(w)
        B = np.linalg.lstsq(X * sw[:, None], Y * sw[:, None], rcond=None)[0]
        if it + 1 >= 1 and np.max(np.abs(B - Bp)) / max(1, np.max(np.abs(Bp))) <= tol:
            break
    df = int(np.sum(w > 0)) - np.linalg.matrix_rank(X)
    return B, w, df
B_r, w_r, df_r = r_irls(X, Y)
B_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
print("R-transcribed IRLS slopes:", np.round(B_r[:2, 0], 4), " zero-weight rows:", int(np.sum(w_r == 0)), " df:", df_r)
print("OLS slopes:", np.round(B_ols[:2, 0], 4))
print("max|fmrimod - R| over all betas:", float(np.max(np.abs(fit.betas - B_r))))
