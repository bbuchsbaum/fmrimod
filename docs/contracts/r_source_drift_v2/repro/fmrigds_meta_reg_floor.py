"""GDS: inverse-variance validity rule for meta:fe_reg / meta:re_reg (5d85f06, 7c7ecae).

Fixed R (verb-reduce.R "Variance floor"; test-variance-floor.R; test-review-numeric.R:256):
  a subject enters a sample only if beta and var are finite AND var > 0;
  0 < var < eps is floored to eps; meta:fe_reg needs max(min_subjects, p + 1)
  usable subjects (min_subjects default 2).
"""
import inspect

import numpy as np
import pandas as pd

from fmrimod.group import (
    SampleLabelSpace,
    group_dataset,
    meta_fe,
    meta_fe_reg,
    meta_re_reg,
)


def ds(beta, var, x):
    n = len(beta)
    return group_dataset(
        {"beta": np.asarray(beta, float).reshape(1, n, 1), "var": np.asarray(var, float).reshape(1, n, 1)},
        space=SampleLabelSpace(["v1"]),
        subjects=[f"s{i}" for i in range(n)],
        contrasts=["c1"],
        col_data=pd.DataFrame({"x": x}, index=[f"s{i}" for i in range(n)]),
    )


def r_fe_reg(y, v, X, eps=1e-12):
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    w = 1 / np.maximum(v[ok], eps)
    Xo, yo = X[ok], y[ok]
    A = np.linalg.inv((Xo * w[:, None]).T @ Xo)
    return A @ (Xo.T @ (w * yo))


rng = np.random.default_rng(3)
n = 6
x = rng.normal(size=n)
X = np.column_stack([np.ones(n), x])
y = rng.normal(size=n)
v = rng.uniform(0.2, 1.0, size=n)
v[1] = 0.0      # zero variance: R excludes subject 2
v[4] = -0.5     # negative variance: R excludes subject 5

print("== meta:fe_reg / meta:re_reg with var = 0 and var < 0 subjects ==")
fe = meta_fe_reg(ds(y, v, x), formula="~ x")
print("fmrimod fe_reg coef:", fe.assay("coef:Intercept").ravel()[0], fe.assay("coef:x").ravel()[0],
      " df_res:", fe.assay("df_res").ravel()[0])
print("R      fe_reg coef:", *r_fe_reg(y, v, X), " df_res:", int(np.sum(v > 0)) - 2)
print("fmrimod fe_reg is pulled to subjects 2/5 (weight 1e12):", y[1], y[4])
re = meta_re_reg(ds(y, v, x), formula="~ x")
print("fmrimod re_reg coef:x:", re.assay("coef:x").ravel()[0], " tau2:", re.assay("tau2").ravel()[0])

print("\n== meta:fe (intercept) applies the rule correctly ==")
fe0 = meta_fe(ds(y, v, x))
ok = v > 0
print("fmrimod beta_g:", fe0.assay("beta_g").ravel()[0],
      " R:", float(np.sum(y[ok] / v[ok]) / np.sum(1 / v[ok])),
      " n_eff assay present:", "n_eff" in fe0.assays)

print("\n== min_subjects on meta:fe_reg (R test-review-numeric.R:256) ==")
print("meta_fe_reg signature:", inspect.signature(meta_fe_reg))
