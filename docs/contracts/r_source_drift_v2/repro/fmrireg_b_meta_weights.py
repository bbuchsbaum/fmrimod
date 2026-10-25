"""REG-B: fmri_meta weights='equal'/'custom' (R review B2, 0f99918).

R (fixed): equal/custom weights are *relative*; Var(y_i) = sigma^2 / w_i with
sigma^2 = Q_fe / df estimated per feature (== lm(weights=w)); equal weights
give the one-sample t-test. tau2/Q/I2 are NA.

fmrimod HEAD: _solve_meta_wls uses w as absolute inverse variances, so the SE
does not depend on the data scale.
"""
import warnings

import numpy as np
import pandas as pd

import fmrimod
from fmrimod.dataset.group_data import group_data_from_csv

warnings.simplefilter("ignore")
rng = np.random.default_rng(21)
S, P = 13, 4
x = rng.normal(size=S)


def make(scale):
    beta = scale * (rng.normal(size=(S, P)) + 0.4 + 0.3 * x[:, None])
    se = rng.uniform(0.1, 2, size=(S, P))
    rows = []
    for i in range(S):
        for j in range(P):
            rows.append({"subject": f"s{i}", "roi": f"r{j}", "beta": beta[i, j],
                         "se": se[i, j], "x": x[i]})
    df = pd.DataFrame(rows)
    gd = group_data_from_csv(df, effect_cols={"beta": "beta", "se": "se"},
                             roi_col="roi", covariate_cols=["x"])
    return beta, se, gd


print("== equal weights vs one-sample t-test (R B2 contract) ==")
for scale in (1e-3, 1.0, 250.0):
    beta, se, gd = make(scale)
    fit = fmrimod.fmri_meta(gd, formula="~ 1", method="fe", weights="equal")
    ref_se = beta.std(axis=0, ddof=1) / np.sqrt(S)
    ref_t = beta.mean(axis=0) / ref_se
    print(f"scale={scale:g}: fmrimod se={fit.se[:, 0].round(6)} | R/t.test se={ref_se.round(6)}")
    print(f"            fmrimod z={fit.z[:, 0].round(4)} | R t={ref_t.round(4)}  tau2={fit.tau2}")

print("\n== custom weights vs lm(weights=w) ==")
beta, se, gd = make(1.0)
w = np.linspace(0.5, 3, S)
fit = fmrimod.fmri_meta(gd, formula="~ 1 + x", method="fe", weights="custom", weights_custom=w)
fit10 = fmrimod.fmri_meta(gd, formula="~ 1 + x", method="fe", weights="custom", weights_custom=10 * w)
X = np.column_stack([np.ones(S), x])
for j in range(P):
    W = np.diag(w)
    XtWX_inv = np.linalg.inv(X.T @ W @ X)
    b = XtWX_inv @ X.T @ W @ beta[:, j]
    r = beta[:, j] - X @ b
    sigma2 = float(r @ W @ r) / (S - 2)
    ref = np.sqrt(np.diag(XtWX_inv) * sigma2)
    print(f"roi{j}: fmrimod se={fit.se[j].round(5)} (x10 weights: {fit10.se[j].round(5)}) | lm se={ref.round(5)}")
