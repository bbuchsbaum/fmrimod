"""REG-A: factorial contrast generator row order (R review C3, 91bb402;
test-review-C-gen-contrast.R).

fmrimod/contrast/factorial.py:94 still uses reduce(np.kron, blocks) -- the
pre-fix R code -- so rows have the FIRST factor varying slowest instead of
following the rows of `des` (expand.grid order, first factor fastest).
"""
import numpy as np
import pandas as pd

from fmrimod.contrast.factorial import (
    generate_interaction_contrast,
    generate_main_effect_contrast,
)

# expand.grid(Time = 1:4, Cond = c("face","scene")): first factor fastest
time_lv, cond_lv = ["1", "2", "3", "4"], ["face", "scene"]
rows = [(t, c) for c in cond_lv for t in time_lv]
des = pd.DataFrame(rows, columns=["Time", "Cond"])
des["Time"] = pd.Categorical(des["Time"], categories=time_lv)
des["Cond"] = pd.Categorical(des["Cond"], categories=cond_lv)
a = np.array([0, 1, 3, 7.0]); b = np.array([0, 10.0])
ti = des["Time"].cat.codes.to_numpy(); ci = des["Cond"].cat.codes.to_numpy()
mu = a[ti] + b[ci]

M = generate_main_effect_contrast(des, "Time")
print("fmrimod  M'mu/2 =", M.T @ mu / 2, " expected -diff(a) =", -np.diff(a))
MC = generate_main_effect_contrast(des, "Cond")
print("fmrimod MC'mu/4 =", MC.T @ mu / 4, " expected -diff(b) =", -np.diff(b))

# R HEAD transcription: row-wise Kronecker indexed by each factor's level in des
def r_fixed(des, factors):
    C = np.ones((len(des), 1))
    for name in des.columns:
        L = len(des[name].cat.categories)
        blk = -np.diff(np.eye(L), axis=0).T if name in factors else np.ones((L, 1))
        Rk = blk[des[name].cat.codes.to_numpy()]
        C = np.column_stack([C * Rk[:, [j]] for j in range(Rk.shape[1])])
    return C
Mr = r_fixed(des, ["Time"])
print("R-fixed  M'mu/2 =", Mr.T @ mu / 2)
ab = np.zeros((4, 2)); ab[2, 1] = 5
mu_ab = mu + ab[ti, ci]
I = generate_interaction_contrast(des, ["Time", "Cond"]); Ir = r_fixed(des, ["Time", "Cond"])
d = ab[:, 0] - ab[:, 1]
print("interaction: fmrimod", I.T @ mu_ab, " R-fixed", Ir.T @ mu_ab, " expected", -np.diff(d))
