"""REG-A: residual df after soft subspace projection (R E7 5edfd16 + c24c720;
test-review-E-soft.R, test-review-F-integration.R).

R: dfres = n - sum(g) - rank(PX) + sum(g ||Q'u_j||^2), g = 2s - s^2,
s = d^2/(d^2+lambda).  fmrimod keeps dfres = n - rank(PX), so sigma^2 is
biased low under the null.
"""
import warnings

import numpy as np
import pandas as pd

import fmrimod as fm
from fmrimod.design import RealizedDesign
from fmrimod.model.config import FmriLmConfig, SoftSubspaceOptions

warnings.simplefilter("ignore")
rng = np.random.default_rng(21)
n, V, k, lam = 200, 3000, 20, 50.0
N = rng.normal(size=(n, k))
X = np.column_stack([np.sin(np.arange(n) / 6.0), np.cos(np.arange(n) / 9.0), np.ones(n)])
Y = rng.normal(size=(n, V))
ev = pd.DataFrame({"onset": [0.0], "condition": ["A"], "duration": [0.0], "run": [1]})
ds = fm.fmri_dataset(Y, tr=2.0, events=ev, run_length=[n])
cfg = FmriLmConfig(soft_subspace=SoftSubspaceOptions(enabled=True, nuisance_matrix=N, lam=lam))
fit = fm.fmri_lm(RealizedDesign.from_array(X, ["a", "b", "int"]), ds, config=cfg)

U, d, _ = np.linalg.svd(N, full_matrices=False)
s = d ** 2 / (d ** 2 + lam); g = 2 * s - s ** 2
Xp = X - U @ (s[:, None] * (U.T @ X))
Q, _ = np.linalg.qr(Xp)
df_exact = n - g.sum() - np.linalg.matrix_rank(Xp) + np.sum(g * np.sum((Q.T @ U) ** 2, axis=0))
print(f"fmrimod residual_df={fit.residual_df}  R exact df={df_exact:.3f}")
print(f"mean sigma^2 (truth 1): fmrimod={np.mean(fit.sigma ** 2):.4f}  "
      f"with R df={np.mean(fit.sigma ** 2) * fit.residual_df / df_exact:.4f}")
