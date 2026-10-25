"""REG-A: zero volume weights and residual df (R 60f8430 #226, review D3 fb00ab1).

R: zero-weight volumes are excluded from residual df (lm.wfit), runwise
full-length weight vectors are sliced per run.  fmrimod apply_volume_weights
(glm/preprocess.py:17) scales rows by sqrt(w) and keeps df = n - rank.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fmrireg_a_fixture import two_run

import fmrimod as fm
from fmrimod.model.config import FmriLmConfig, VolumeWeightOptions

warnings.simplefilter("ignore")
ds, Y, ev = two_run(seed=8, n=60)
rng = np.random.default_rng(30)
w = rng.uniform(0.3, 1, 120)
w[[2, 3, 49, 60, 61, 100]] = 0
fit = fm.fmri_lm("hrf(condition)", ds, config=FmriLmConfig(
    volume_weights=VolumeWeightOptions(enabled=True, weights=w)))
X0 = fit.model.design_matrix_array(run=0); X1 = fit.model.design_matrix_array(run=1)
rk = lambda A: np.linalg.matrix_rank(A)
df_lm = sum(int(np.sum(w[s] > 0)) - rk(Xr[w[s] > 0]) for s, Xr in ((slice(0, 60), X0), (slice(60, 120), X1)))
print("fmrimod residual_df:", fit.residual_df, " lm.wfit per-run df sum:", df_lm)
# per-run WLS betas are still right (sqrt-weighting), only df/sigma differ
sig_ratio = np.sqrt((fit.residual_df) / df_lm)
print("=> fmrimod sigma too small by factor", round(float(sig_ratio), 4), "(t inflated by same factor)")
