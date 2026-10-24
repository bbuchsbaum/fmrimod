"""REG-A: fmri_lm shared-AR estimate lacks the OLS residual-bias (design)
correction R applies (fmriAR_adapter.R .estimate_ar_via_fmriAR(design=X,
correction_max_lag=.ar_correction_lag_budget()), 23c275f / 168980d).

R reference = fmrimod.ar.estimation.fit_noise(resid, design=X,
correction_max_lag=budget) -- the wave-1 port of fmriAR 0.3.3's correction --
with R's lag budget max(p, min(25, max(5, 2p+1), floor(rdf/3))).
fmrimod fmri_lm path = ar.integration.iterative_gls -> estimate_ar (plain
pooled-ACVF Yule-Walker on OLS residuals, no design).
"""
import warnings

import numpy as np
import pandas as pd

import fmrimod as fm
from fmrimod.ar.estimation import fit_noise
from fmrimod.design import RealizedDesign
from fmrimod.model.config import AROptions, FmriLmConfig

warnings.simplefilter("ignore")
rng = np.random.default_rng(20261024)
n, V, phi_true, n_rep = 100, 40, 0.4, 40
t = np.linspace(-1, 1, n)
# events: two boxcar-ish regressors + intercept + Legendre-like poly drift deg 5
ev1 = (np.sin(np.arange(n) * 2 * np.pi / 20) > 0.6).astype(float)
ev2 = (np.cos(np.arange(n) * 2 * np.pi / 14) > 0.6).astype(float)
drift = np.column_stack([t ** k for k in range(1, 6)])
X = np.column_stack([ev1, ev2, np.ones(n), drift])
cols = ["e1", "e2", "int"] + [f"d{k}" for k in range(1, 6)]
rdf = n - np.linalg.matrix_rank(X)
budget = int(max(1, min(25, max(5, 3), rdf // 3)))

est_fm, est_r = [], []
for rep in range(n_rep):
    E = np.zeros((n, V))
    z = rng.normal(size=(n, V))
    for i in range(n):
        E[i] = z[i] + (phi_true * E[i - 1] if i else 0.0)
    Y = X @ rng.normal(size=(X.shape[1], V)) + E
    ev = pd.DataFrame({"onset": [0.0], "condition": ["A"], "duration": [0.0], "run": [1]})
    ds = fm.fmri_dataset(Y, tr=2.0, events=ev, run_length=[n])
    fit = fm.fmri_lm(RealizedDesign.from_array(X, cols), ds,
                     config=FmriLmConfig(ar=AROptions(struct="ar1")))
    est_fm.append(float(np.ravel(fit.ar_params)[0]))
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    R = Y - X @ B
    plan = fit_noise(resid=R, method="ar", p=1, p_max=1, pooling="global",
                     exact_first="ar1", design=X, correction_max_lag=budget)
    est_r.append(float(plan.phi[0][0]))

print(f"true phi={phi_true}  n={n} rank={np.linalg.matrix_rank(X)} lag budget={budget}")
print(f"fmrimod fmri_lm ar_params mean={np.mean(est_fm):.4f} sd={np.std(est_fm):.4f}")
print(f"R-equivalent design-corrected mean={np.mean(est_r):.4f} sd={np.std(est_r):.4f}")
