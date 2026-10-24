"""REG-A: sketch-and-solve inference (R cdb5751; test-sketch-inference.R).

R: sketch-and-solve fits report the conditional covariance
(Xs'Xs)^-1 Xs'SS'Xs (Xs'Xs)^-1, sigma^2 = ||r_s||^2 / tr(P SS') and
Satterthwaite df (~ m - p); before the fix type-I at nominal .05 was ~0.42-0.45
(m=40, T=200).  fmrimod lowrank/engine.py:156-166 + glm/engines/sketch.py:98-105:
sigma^2 = ||r_s||^2/(n - rank), XtXinv = full-data (X'X)^-1, df = n - rank.
"""
import inspect
import warnings

import numpy as np
import pandas as pd

import fmrimod as fm
from fmrimod.design import RealizedDesign
from fmrimod.glm import SketchEngineOptions

warnings.simplefilter("ignore")
print(inspect.signature(SketchEngineOptions))
rng = np.random.default_rng(1)
T, V = 200, 4000
X = np.column_stack([np.sin(np.arange(T) / 4.0), np.cos(np.arange(T) / 11.0), np.ones(T),
                     np.linspace(-1, 1, T)])
Y = rng.normal(size=(T, V))            # null for every column
ev = pd.DataFrame({"onset": [0.0], "condition": ["A"], "duration": [0.0], "run": [1]})
ds = fm.fmri_dataset(Y, tr=2.0, events=ev, run_length=[T])
ols = fm.fmri_lm(RealizedDesign.from_array(X, ["a", "b", "int", "lin"]), ds)
print("OLS type-I (|t| test, alpha .05):", float(np.mean(ols.contrast(np.array([1.0, 0, 0, 0])).p_value < 0.05)))
for kind in ("gaussian", "srht", "countsketch"):
    for m in (40, 100):
        eng = SketchEngineOptions(sketch_kind=kind, sketch_size=m, seed=3)
        f = fm.fmri_lm(RealizedDesign.from_array(X, ["a", "b", "int", "lin"]), ds, engine=eng)
        p = f.contrast(np.array([1.0, 0, 0, 0])).p_value
        print(f"{kind:11s} m={m:3d}: type-I={np.mean(p < 0.05):.3f}  mean sigma^2={np.mean(f.sigma**2):.3f} (truth 1)  df={f.residual_df}")

# R transcription (sketch_time.R .lowrank_sketch_solve) on the same sketch matrix
from scipy import stats as sps

from fmrimod.lowrank.sketch import make_sketch

for kind in ("gaussian", "srht", "countsketch"):
    for m in (40, 100):
        S = make_sketch(T, m, kind=kind, rng=np.random.default_rng(3))
        Xs, Zs = S @ X, S @ Y
        Q, Rr = np.linalg.qr(Xs)
        Ginv = np.linalg.inv(Rr.T @ Rr)
        B = np.linalg.solve(Rr, Q.T @ Zs)
        res = Zs - Q @ (Q.T @ Zs)
        K = S @ S.T
        cov_u = Ginv @ (Xs.T @ K @ Xs) @ Ginv
        KQ = K @ Q; QKQ = Q.T @ KQ
        kappa = np.trace(K) - np.trace(QKQ)
        trPKPK = np.sum(K * K) - 2 * np.sum(KQ * KQ) + np.sum(QKQ * QKQ)
        df = kappa ** 2 / trPKPK
        s2 = np.sum(res ** 2, axis=0) / kappa
        t = B[0] / np.sqrt(s2 * cov_u[0, 0])
        p = 2 * sps.t.sf(np.abs(t), df)
        print(f"R-fixed {kind:11s} m={m:3d}: type-I={np.mean(p < 0.05):.3f}  df={df:.1f}  mean sigma^2={np.mean(s2):.3f}")
